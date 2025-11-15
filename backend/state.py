from pathlib import Path

FILES: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
RESULTS: dict[str, dict] = {}

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CLIP_DIR = BASE_DIR / "clips"

UPLOAD_DIR.mkdir(exist_ok=True)
CLIP_DIR.mkdir(exist_ok=True)
