from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .schemas import ProcessRequest
from .state import CLIP_DIR, FILES, JOBS, RESULTS, UPLOAD_DIR, persist_job_snapshot, refresh_jobs_from_disk
from .tasks import run_processing
from .utils import now_iso

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Backend online", "endpoints": ["/upload", "/process"]}


async def save_upload_to_disk(upload: UploadFile, destination: Path) -> int:
    size = 0
    with destination.open("wb") as buffer:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            buffer.write(chunk)
    await upload.close()
    return size

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(..., description="Video file (mp4/mov/etc.)"),
    patient_id: Optional[str] = Form(None),
    procedure_type: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video uploads are allowed.")

    extension = Path(file.filename or "upload.mp4").suffix or ".mp4"
    file_id = f"file_{uuid4().hex[:12]}"
    destination = UPLOAD_DIR / f"{file_id}{extension}"

    size_bytes = await save_upload_to_disk(file, destination)
    if size_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    FILES[file_id] = {
        "file_id": file_id,
        "path": str(destination),
        "name": file.filename,
        "size_bytes": size_bytes,
        "content_type": file.content_type,
        "metadata": {
            "patient_id": patient_id or "",
            "procedure_type": procedure_type or "",
            "notes": notes or "",
        },
    }

    return JSONResponse({
        "file_id": file_id,
        "name": file.filename,
        "size_bytes": size_bytes,
        "metadata": FILES[file_id]["metadata"],
    })


@router.post("/process")
async def start_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    if req.file_id not in FILES:
        raise HTTPException(status_code=404, detail="file_id not found. Upload first and pass the returned file_id.")

    file_info = FILES.get(req.file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="File metadata missing for supplied file_id")

    job_id = f"job_{uuid4().hex[:12]}"
    JOBS[job_id] = {
        "job_id": job_id,
        "file_id": req.file_id,
        "file_name": file_info.get("name") or Path(file_info["path"]).name,
        "status": "queued",
        "progress": 0,
        "model": req.model or "baseline_v1",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "error": None,
    }

    persist_job_snapshot(job_id)

    background_tasks.add_task(run_processing, job_id)

    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/history")
async def list_job_history(limit: int = 5):
    refresh_jobs_from_disk()
    limit = max(1, min(limit, 25))
    ordered = sorted(JOBS.values(), key=lambda entry: entry["created_at"], reverse=True)
    payload = []
    for job in ordered[:limit]:
        job_summary = {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job.get("progress", 0),
            "file_id": job["file_id"],
            "file_name": job.get("file_name"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "model": job.get("model"),
            "error": job.get("error"),
        }
        result = RESULTS.get(job["job_id"])
        if result:
            job_summary["result"] = {
                "summary_text": result.get("summary_text"),
                "phase_clips": result.get("phase_clips", []),
                "source_file": result.get("source_file"),
            }
        payload.append(job_summary)
    return {"jobs": payload}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job.get("progress", 0),
        "file_id": job["file_id"],
        "file_name": job.get("file_name"),
        "updated_at": job["updated_at"],
        "error": job.get("error"),
    }


@router.get("/jobs/{job_id}/result")
async def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job not completed. Current status: {job['status']}")
    result = RESULTS.get(job_id)
    if not result:
        raise HTTPException(status_code=500, detail="Result missing for completed job")
    return {"job_id": job_id, "result": result}


@router.get("/jobs/{job_id}/source")
async def get_job_source(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    file_info = FILES.get(job["file_id"])
    if not file_info:
        raise HTTPException(status_code=404, detail="Original file metadata missing")

    source_path = Path(file_info["path"]).resolve()
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Original file unavailable")

    filename = file_info.get("name") or source_path.name
    media_type = file_info.get("content_type") or "video/mp4"
    return FileResponse(source_path, media_type=media_type, filename=filename)


@router.get("/jobs/{job_id}/clips/{clip_name}")
async def get_phase_clip(job_id: str, clip_name: str, download: bool = False):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Job not found or not completed")

    clip_dir = (CLIP_DIR / job_id).resolve()
    target = (clip_dir / clip_name).resolve()
    try:
        target.relative_to(clip_dir)
    except ValueError:
        raise HTTPException(status_code=404, detail="Clip not found")

    if not target.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    filename = clip_name if download else None
    return FileResponse(target, media_type="video/mp4", filename=filename)
