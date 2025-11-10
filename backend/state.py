from pathlib import Path

FILES: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
RESULTS: dict[str, dict] = {}

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
SLICE_DIR = BASE_DIR / "slices"

UPLOAD_DIR.mkdir(exist_ok=True)
SLICE_DIR.mkdir(exist_ok=True)
