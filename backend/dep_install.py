from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("backend.deps")


def _python_exe_for_venv(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _site_packages_for_venv(venv_path: Path) -> Path:
    """Return the site-packages path for a venv on this platform/version.

    This is a best-effort helper; the exact path depends on platform and
    python minor version.
    """
    venv_path = Path(venv_path)
    if os.name == "nt":
        return venv_path / "Lib" / "site-packages"
    # POSIX: venv/lib/pythonX.Y/site-packages
    return venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def ensure_venv(venv_path: Path) -> Path:
    """Ensure a virtualenv exists at `venv_path`. Returns the python executable path.

    Creating a venv is cheap compared to downloading packages; we create it
    if missing so callers can (optionally) patch `sys.path` to use the venv's
    site-packages during the running process.
    """
    venv_path = Path(venv_path)
    python_exe = _python_exe_for_venv(venv_path)
    if python_exe.exists():
        LOG.info("Found existing venv at %s", venv_path)
        return python_exe

    LOG.info("Creating venv at %s", venv_path)
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
    python_exe = _python_exe_for_venv(venv_path)
    if not python_exe.exists():
        raise RuntimeError(f"Failed to create venv at {venv_path}")
    return python_exe


def add_venv_site_packages_to_sys_path(venv_path: Path) -> bool:
    """Add the venv's site-packages directory to `sys.path` (front).

    Returns True if a path was added, False otherwise.
    """
    sp = _site_packages_for_venv(venv_path)
    if not sp.exists():
        LOG.debug("site-packages not found at %s", sp)
        return False
    sp_str = str(sp)
    if sp_str in sys.path:
        LOG.debug("site-packages already on sys.path: %s", sp)
        return True
    sys.path.insert(0, sp_str)
    LOG.info("Added venv site-packages to sys.path: %s", sp)
    return True


def get_backend_venv_path(project_root: Path) -> Path:
    return Path(project_root) / "backend" / "venv"


def pip_install_requirements(python_exe: Path, requirements: Path) -> int:
    """Install requirements using the given python executable. Returns exit code."""
    if not requirements.exists():
        LOG.warning("No requirements file at %s", requirements)
        return 0

    cmd = [str(python_exe), "-m", "pip", "install", "-r", str(requirements)]
    LOG.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    LOG.info(proc.stdout)
    if proc.returncode != 0:
        LOG.error(proc.stderr)
    return proc.returncode


def ensure_backend_venv_and_patch_sys_path(project_root: Path) -> None:
    """Ensure `backend/venv` exists and add its site-packages to sys.path.

    This is intended to be called early (before importing modules that may
    require model packages) so the running process can import packages
    installed into the backend venv.
    """
    venv_path = get_backend_venv_path(project_root)
    # Prefer not to create the venv synchronously during module import/startup
    # (that can be slow). Only add site-packages if the venv already exists.
    sp = _site_packages_for_venv(venv_path)
    if sp.exists():
        added = add_venv_site_packages_to_sys_path(venv_path)
        if not added:
            LOG.debug("backend venv site-packages not added: %s", venv_path)
    else:
        LOG.info("backend venv not present yet; skipping sys.path patch: %s", venv_path)


def ensure_and_install_model_deps(project_root: Path) -> None:
    """Ensure backend venv exists and install `model/requirements.txt` into it.

    This is safe to call repeatedly; it will create the venv only if missing and
    will run pip install each time (you can skip via env var).
    """
    skip = os.environ.get("SKIP_MODEL_AUTO_INSTALL", "0")
    if skip and skip != "0":
        LOG.info("Skipping automatic model dependency installation (SKIP_MODEL_AUTO_INSTALL set)")
        return

    model_dir = Path(project_root) / "model"
    requirements = model_dir / "requirements.txt"
    venv_path = get_backend_venv_path(project_root)

    try:
        python_exe = ensure_venv(venv_path)
    except Exception as exc:
        LOG.exception("Failed to create backend venv: %s", exc)
        return

    try:
        rc = pip_install_requirements(python_exe, requirements)
        if rc == 0:
            LOG.info("Model dependencies installed successfully into %s", venv_path)
        else:
            LOG.error("pip returned non-zero exit code %d", rc)
    except Exception:
        LOG.exception("Error installing model requirements")
