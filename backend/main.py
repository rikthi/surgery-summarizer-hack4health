from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi import BackgroundTasks
from pydantic import BaseModel
from uuid import uuid4
from datetime import datetime, timezone
import asyncio
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()
from typing import Optional
import os

# importing ImageKit SDK
try:
    from imagekitio import ImageKit
    from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
except Exception as e:
    # Defer import error to runtime response so the app can still start
    ImageKit = None


app = FastAPI()

 
FILES = {}     # file_id -> {url, name, size_bytes, ...}
JOBS = {}      # job_id -> {status, file_id, progress, created_at, updated_at, error}
RESULTS = {}   # job_id -> result dict

def now_iso():
    return datetime.now(timezone.utc).isoformat()

# Loading ImageKit configuration from environment variables
IMAGEKIT_PUBLIC_KEY = (os.getenv("IMAGEKIT_PUBLIC_KEY") or "").strip()
IMAGEKIT_PRIVATE_KEY = (os.getenv("IMAGEKIT_PRIVATE_KEY") or "").strip()
IMAGEKIT_URL_ENDPOINT = (os.getenv("IMAGEKIT_URL") or "").strip()

if IMAGEKIT_URL_ENDPOINT and not IMAGEKIT_URL_ENDPOINT.startswith("http"):
    IMAGEKIT_URL_ENDPOINT = IMAGEKIT_URL_ENDPOINT.strip()

if IMAGEKIT_PUBLIC_KEY and IMAGEKIT_PRIVATE_KEY and IMAGEKIT_URL_ENDPOINT and ImageKit:
    imagekit = ImageKit(
        public_key=IMAGEKIT_PUBLIC_KEY,
        private_key=IMAGEKIT_PRIVATE_KEY,
        url_endpoint=IMAGEKIT_URL_ENDPOINT,
    )
else:
    imagekit = None


@app.get("/")
async def root():
    return {"message": "OK", "upload": "/upload"}


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(..., description="Video file (mp4/mov/etc.)"),
    patient_id: Optional[str] = Form(None),
    procedure_type: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    """
    Receives a video via form-data and uploads it to ImageKit.
    """

    # Guardrails for ImageKit SDK availability
    if imagekit is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "ImageKit SDK not available or environment variables missing. "
                "Ensure imagekitio is installed and IMAGEKIT_PUBLIC_KEY, IMAGEKIT_PRIVATE_KEY, IMAGEKIT_URL are set."
            ),
        )

    # Validate if its a video file
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video uploads are allowed.")

    # Read bytes 
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ImageKit upload options
    tags = [t for t in ["surgery", procedure_type] if t]

    # Minimal options to avoid Invalid upload options
    options = UploadFileRequestOptions(
        use_unique_file_name=True,
        folder="surgery-raw",
        tags=tags if tags else None,
        # Note: omit is_private_file, response_fields, and custom_metadata for MVP
    )

    # Upload to ImageKit
    try:
        res = imagekit.upload_file(
            file=data,
            file_name=file.filename or "video.mp4",
            options=options,
        )
    except Exception as e:
        detail_msg = f"ImageKit upload failed: {e}"
        if hasattr(e, 'args') and len(e.args) > 0 and e.args[0]:
            detail_msg = f"ImageKit upload failed: {e.args[0]}"
        raise HTTPException(
            status_code=502,
            detail=detail_msg
        )

    # Normalize SDK response
    payload = {
        "file_id": getattr(res, "file_id", None),
        "url": getattr(res, "url", None),
        "thumbnail_url": getattr(res, "thumbnail_url", None),
        "name": getattr(res, "name", file.filename),
        "size_bytes": getattr(res, "size", None),
        "format": getattr(res, "file_type", None),
        "height": getattr(res, "height", None),
        "width": getattr(res, "width", None),
        "metadata": {
            "patient_id": patient_id or "",
            "procedure_type": procedure_type or "",
            "notes": notes or "",
        },
    }
    # Store file info for downstream processing
    if payload.get("file_id"):
        FILES[payload["file_id"]] = payload

    return JSONResponse(payload)


# --- Processing API---

# Pydantic model for /process requests
class ProcessRequest(BaseModel):
    file_id: str
    model: str | None = None


# Background task to simulate AI processing and populate a result
async def run_processing(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return
    job["status"] = "processing"
    job["updated_at"] = now_iso()

    # Retrieve file info (e.g., URL) for the job
    file_info = FILES.get(job["file_id"], {})
    video_url = file_info.get("url")

    try:
        # Simulating incremental progress
        for p in (10, 25, 40, 65, 85, 100):
            await asyncio.sleep(0.6)  # simulate work
            job["progress"] = p
            job["updated_at"] = now_iso()

        # Fake AI result (to be replaced)
        RESULTS[job_id] = {
            "summary_text": "Key steps detected: trocar placement, dissection, clipping, extraction.",
            "timeline": [
                {"t": 120.0, "label": "Trocar placement"},
                {"t": 540.0, "label": "Dissection start"},
                {"t": 1320.0, "label": "Clip & cut"},
                {"t": 2100.0, "label": "Specimen extraction"}
            ],
            "keyframes": [],
            "confidence": 0.87,
            "source_file": {
                "file_id": job["file_id"],
                "url": video_url,
            },
        }
        job["status"] = "completed"
        job["updated_at"] = now_iso()
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["updated_at"] = now_iso()


# POST /process: create a job and start background task
@app.post("/process")
async def start_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    # Validate file exists
    if req.file_id not in FILES:
        raise HTTPException(status_code=404, detail="file_id not found. Upload first and pass the returned file_id.")

    job_id = f"job_{uuid4().hex[:12]}"
    JOBS[job_id] = {
        "job_id": job_id,
        "file_id": req.file_id,
        "status": "queued",
        "progress": 0,
        "model": req.model or "baseline_v1",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "error": None,
    }

    # Start background processing
    background_tasks.add_task(run_processing, job_id)

    return {"job_id": job_id, "status": "queued"}

 # checking job status
@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    # Return a subset safe for clients
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job.get("progress", 0),
        "file_id": job["file_id"],
        "updated_at": job["updated_at"],
        "error": job.get("error"),
    }


#gets final summary result
@app.get("/jobs/{job_id}/result")
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