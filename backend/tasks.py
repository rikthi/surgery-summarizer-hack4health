from __future__ import annotations

import asyncio
from pathlib import Path

from .llm_stub import build_llm_stub_response
from .state import FILES, JOBS, RESULTS
from .utils import now_iso
from .video_processing import extract_video_slices


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

        analysis = await asyncio.get_event_loop().run_in_executor(
            None,
            extract_video_slices,
            video_path,
            job_id,
        )

        job["progress"] = 65
        job["updated_at"] = now_iso()

        llm_stub = build_llm_stub_response(analysis)

        job["progress"] = 90
        job["updated_at"] = now_iso()

        RESULTS[job_id] = {
            "summary_text": llm_stub["summary_text"],
            "highlights": llm_stub["highlights"],
            "frame_slices": llm_stub["frame_slices"],
            "video_metadata": llm_stub["video_metadata"],
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
