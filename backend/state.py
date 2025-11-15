from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable

LOG = logging.getLogger("backend.state")

FILES: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
RESULTS: dict[str, dict] = {}

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CLIP_DIR = BASE_DIR / "clips"

PERSIST_ROOTS: list[Path] = [BASE_DIR / "state", UPLOAD_DIR / "_state"]
STATE_JOB_DIRS: list[Path] = []

UPLOAD_DIR.mkdir(exist_ok=True)
CLIP_DIR.mkdir(exist_ok=True)

for root in PERSIST_ROOTS:
	jobs_dir = root / "jobs"
	jobs_dir.mkdir(parents=True, exist_ok=True)
	STATE_JOB_DIRS.append(jobs_dir)

# Maintain backward compatibility with earlier single-dir constant importers
STATE_JOBS_DIR = STATE_JOB_DIRS[0]


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	with tmp_path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=2)
	tmp_path.replace(path)


def persist_job_snapshot(job_id: str) -> None:
	job = JOBS.get(job_id)
	if not job:
		return
	file_info = FILES.get(job.get("file_id"))
	payload = {
		"job": job,
		"file": file_info,
		"result": RESULTS.get(job_id),
	}
	for jobs_dir in STATE_JOB_DIRS:
		target = jobs_dir / f"{job_id}.json"
		try:
			_write_json_atomic(target, payload)
		except Exception:
			LOG.exception("Failed to persist job snapshot for %s at %s", job_id, target)


def _iter_snapshot_paths() -> Iterable[Path]:
	seen: set[str] = set()
	for jobs_dir in STATE_JOB_DIRS:
		if not jobs_dir.exists():
			continue
		for snapshot in jobs_dir.glob("*.json"):
			key = snapshot.stem
			if key in seen:
				continue
			seen.add(key)
			yield snapshot


def _load_persisted_jobs() -> None:
	loaded_jobs = 0
	for snapshot in _iter_snapshot_paths():
		try:
			with snapshot.open("r", encoding="utf-8") as handle:
				payload = json.load(handle)
		except Exception:
			LOG.warning("Unable to read snapshot %s", snapshot.name, exc_info=True)
			continue

		job = payload.get("job")
		file_info = payload.get("file")
		result = payload.get("result")

		if not job or "job_id" not in job:
			continue

		job_id = job["job_id"]
		JOBS[job_id] = job
		if file_info and "file_id" in file_info:
			FILES[file_info["file_id"]] = file_info
		if result:
			RESULTS[job_id] = result
		loaded_jobs += 1

	if loaded_jobs:
		LOG.info("Loaded %d persisted job(s) from disk", loaded_jobs)


_load_persisted_jobs()


def refresh_jobs_from_disk() -> None:
	"""Re-load any job snapshots from disk into in-memory stores."""

	_load_persisted_jobs()
