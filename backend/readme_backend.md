# Backend Overview

## Processing Pipeline
1. **Upload video** &mdash; `POST /upload` accepts a local video file and saves it under `backend/uploads/`, returning a `file_id`.
2. **Create job** &mdash; `POST /process` starts a background task bound to the uploaded file and returns a `job_id`.
3. **Slice frames** &mdash; `extract_video_slices` samples frames every _N_ seconds (configurable) and stores JPEG previews in `backend/slices/{job_id}`.
4. **Generate summary** &mdash; `run_processing` currently calls `build_llm_stub_response(analysis)` to simulate an AI summary.
5. **Persist result** &mdash; Completed jobs are cached in memory and made available through `GET /jobs/{job_id}` and `GET /jobs/{job_id}/result`.

## Key Modules
- **FastAPI** &mdash; REST API surface
- **OpenCV / NumPy** &mdash; Video decoding and frame handling (install with `pip install opencv-python numpy`)
- **AsyncIO + BackgroundTasks** &mdash; Non-blocking job execution

## Where To Integrate AI Call
- Replace the stub inside [`build_llm_stub_response`](llm_stub.py) with your model invocation. Its sole parameter, `analysis`, is the dictionary produced by `extract_video_slices` and contains:
	- `analysis["slices"]`: list of sampled frames. Each entry includes:
		- `image_base64`: base64 JPEG preview ready to send to a multimodal LLM
		- `timestamp`: human readable `HH:MM:SS`
		- `time_seconds`: float seconds-from-start (useful for video scrubbing)
		- `path`: absolute path to the cached JPEG slice on disk
	- `analysis["fps"]`, `analysis["frame_count"]`, `analysis["duration_seconds"]`, and `analysis["sample_interval_seconds"]` for additional context.
- Suggested pattern:
	1. Construct your LLM prompt using `analysis["slices"]` (images) plus metadata.
	2. Replace the stub return value with the model response (summary, highlights, etc.).
	3. Keep the output structure (`summary_text`, `highlights`, `frame_slices`, `video_metadata`) so the frontend continues to function.

## Performance Notes
- Frame sampling uses targeted frame seeks to avoid decoding the entire video, greatly reducing processing time for long recordings.
- JPEG encoding and base64 conversion run in a thread pool, enabling multi-core utilization without altering API shape.
- Previous slice files are cleared before each run to keep disk usage bounded.

## Configuration Tips
- Ensure the backend runs with access to a GPU-friendly build of OpenCV if available; otherwise CPU decoding is used.
- For extremely long videos, adjust `sample_every_seconds` or `max_slices` in `extract_video_slices` to tune throughput vs. fidelity.