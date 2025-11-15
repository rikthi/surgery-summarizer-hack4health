import asyncio
import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import dep_install

# Ensure a console handler at INFO level so module `LOG.info` messages are visible.
root_logger = logging.getLogger()
if not root_logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

# Also make sure common server loggers are not quieter than INFO
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

LOG = logging.getLogger("backend.startup")

# Project root (two levels up from this file)
project_root = Path(__file__).resolve().parent.parent

# At import time we avoid creating venvs or running pip (can be slow).
# The installer will run during the FastAPI startup event so the uvicorn
# worker process performs the installation and the server blocks until
# dependencies are ready.

# Make sure the project root is on sys.path so imports like `backend.routes`
# work reliably when uvicorn's reloader changes working directories.
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

try:
    # Preferred import when running from repo root: `uvicorn backend.main:app`
    from backend.routes import router
except Exception:
    LOG.debug("Import backend.routes failed; trying local import fallback", exc_info=True)
    # Fallback: try importing `routes` as a top-level module (works when
    # running from inside the `backend` directory: `cd backend; uvicorn main:app`)
    try:
        from routes import router
    except Exception:
        LOG.exception("Failed to import router from both 'backend.routes' and 'routes'")
        raise

app = FastAPI(title="Surgical Summarizer Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def _ensure_model_deps_on_startup():
    """Ensure model dependencies are installed into `backend/venv` before serving.

    This runs in the uvicorn worker process and will block startup until
    installation completes (or fails). This is intentional so the app has
    required packages available when handling requests.
    """
    skip = False
    try:
        skip_env = dep_install.os.environ.get("SKIP_MODEL_AUTO_INSTALL", "0")
        skip = skip_env and skip_env != "0"
    except Exception:
        skip = False

    if skip:
        LOG.info("Skipping model auto-install per SKIP_MODEL_AUTO_INSTALL")
        return

    LOG.info("Starting synchronous model dependency install into backend/venv (this may take several minutes)")
    loop = asyncio.get_event_loop()
    # Run blocking install in executor but await it so startup waits.
    await loop.run_in_executor(None, dep_install.ensure_and_install_model_deps, project_root)
    # After installation, patch sys.path so imports resolve from backend venv.
    try:
        dep_install.ensure_backend_venv_and_patch_sys_path(project_root)
    except Exception:
        LOG.exception("Failed to patch sys.path with backend venv site-packages")
    LOG.info("Model dependency install completed on startup")
    loop.create_task(_warm_model_after_startup())


async def _warm_model_after_startup(delay_seconds: float = 2.0) -> None:
    await asyncio.sleep(delay_seconds)
    try:
        try:
            from backend.model_inference import warm_start_model  # type: ignore
        except Exception:  # pragma: no cover - fallback for relative import usage
            from .model_inference import warm_start_model  # type: ignore

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, warm_start_model)
    except Exception:
        LOG.exception("Background model warm-start failed")