from __future__ import annotations

import asyncio
import logging
from pathlib import Path

LOG = logging.getLogger("backend.tasks")

from .state import FILES, JOBS, RESULTS
from .utils import now_iso


async def run_processing(job_id: str) -> None:
    job = JOBS.get(job_id)
    if not job:
        return

    job["status"] = "processing"
    job["updated_at"] = now_iso()

    file_info = FILES.get(job["file_id"])
    if not file_info:
        job["status"] = "failed"
        job["error"] = "File metadata missing"
        job["updated_at"] = now_iso()
        return

    video_path = Path(file_info["path"])

    try:
        job["progress"] = 10
        job["updated_at"] = now_iso()

        loop = asyncio.get_event_loop()

        # Import model-heavy modules lazily so the application can start even
        # if backend venv packages are not yet installed. These imports will
        # raise if packages are missing when inference actually runs.
        from .model_inference import infer_procedure_phases
        from .llm_stub import build_llm_stub_response
        from .video_clips import generate_phase_clips
        LOG.info("Starting model inference for job %s", job_id)
        phase_result = await loop.run_in_executor(
            None,
            infer_procedure_phases,
            video_path,
        )
        LOG.info("Model inference complete for job %s: %d segments", job_id, len(phase_result.segments))

        job["progress"] = 60
        job["updated_at"] = now_iso()

        LOG.info("Creating per-phase clips for job %s", job_id)
        phase_clips = await loop.run_in_executor(
            None,
            generate_phase_clips,
            video_path,
            phase_result.longest_segments,
            job_id,
        )

        job["progress"] = 80
        job["updated_at"] = now_iso()

        llm_stub = build_llm_stub_response(phase_result)

        job["progress"] = 90
        job["updated_at"] = now_iso()

        RESULTS[job_id] = {
            "summary_text": llm_stub["summary_text"],
            "phase_predictions": llm_stub["phase_predictions"],
            "phase_segments": llm_stub["phase_segments"],
            "phase_segments_filtered": llm_stub["phase_segments_filtered"],
            "phase_segments_longest": llm_stub["phase_segments_longest"],
            "phase_clips": phase_clips,
            "source_file": {
                "file_id": job["file_id"],
                "path": file_info["path"],
                "name": file_info.get("name"),
                "size_bytes": file_info.get("size_bytes"),
                "content_type": file_info.get("content_type"),
            },
        }

        job["progress"] = 100
        job["status"] = "completed"
        job["updated_at"] = now_iso()
    except Exception as exc:  # pragma: no cover - surfaced to API layer
        job["status"] = "failed"
        job["error"] = str(exc)
        job["updated_at"] = now_iso()
