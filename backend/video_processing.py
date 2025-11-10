from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import base64
import math
import os

from fastapi import HTTPException

from .state import SLICE_DIR
from .utils import human_timestamp

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None
    np = None


def ensure_video_tooling_available() -> None:
    if cv2 is None or np is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Video processing dependencies are missing. Install with "
                "`pip install opencv-python numpy` on backend."
            ),
        )


def extract_video_slices(
    video_path: Path,
    job_id: str,
    sample_every_seconds: int = 10,
    max_slices: int = 12,
) -> dict:
    """Return sampled frames and lightweight previews for the uploaded video.

    The returned payload includes a ``slices`` list where each entry already
    carries an ``image_base64`` field suitable for direct dispatch to the LLM.
    """

    ensure_video_tooling_available()

    fps, total_frames, duration = probe_video_metrics(video_path)
    slice_folder = SLICE_DIR / job_id
    slice_folder.mkdir(exist_ok=True)
    clear_previous_slices(slice_folder)

    sample_every_seconds = max(sample_every_seconds, 1)
    max_slices = max(max_slices, 1)

    target_timestamps = generate_target_timestamps(duration, sample_every_seconds, max_slices)
    target_frames = tuple(
        (
            min(int(round(timestamp * fps)), max(total_frames - 1, 0))
            if total_frames
            else int(round(timestamp * fps))
        )
        for timestamp in target_timestamps
    )

    try:
        raw_frames = collect_frames_random_access(video_path, target_frames, target_timestamps)
    except RuntimeError:
        raw_frames = collect_frames_sequential(
            video_path,
            fps,
            sample_every_seconds,
            max_slices,
        )

    slices = build_slice_payloads(
        raw_frames,
        slice_folder,
    )

    return {
        "slices": slices,
        "fps": float(fps),
        "frame_count": int(total_frames),
        "duration_seconds": float(duration),
        "sample_interval_seconds": sample_every_seconds,
    }


def probe_video_metrics(video_path: Path) -> tuple[float, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to open uploaded video for processing")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    duration = total_frames / fps if fps else 0.0
    return float(fps or 30.0), total_frames, float(duration)


def generate_target_timestamps(
    duration: float,
    sample_every_seconds: int,
    max_slices: int,
) -> Sequence[float]:
    candidates = {0.0}
    if duration > 0:
        steps = max(1, int(math.ceil(duration / sample_every_seconds)))
        for idx in range(1, steps + 1):
            if len(candidates) >= max_slices:
                break
            candidates.add(min(duration, idx * sample_every_seconds))
    ordered = sorted(candidates)
    return tuple(ordered[:max_slices])


def collect_frames_random_access(
    video_path: Path,
    target_frames: Sequence[int],
    target_timestamps: Sequence[float],
) -> Sequence[tuple[int, float, Any]]:
    if len(target_frames) != len(target_timestamps):
        raise RuntimeError("Target frame metadata mismatch")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to open uploaded video for processing")

    raw_frames: list[tuple[int, float, Any]] = []
    try:
        for index, (frame_idx, timestamp) in enumerate(zip(target_frames, target_timestamps, strict=True)):
            if frame_idx < 0:
                frame_idx = 0
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx)):
                raise RuntimeError("Random access not supported by codec")
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to decode frame during random access")
            raw_frames.append((index, float(timestamp), frame))
    finally:
        cap.release()

    return raw_frames


def collect_frames_sequential(
    video_path: Path,
    fps: float,
    sample_every_seconds: int,
    max_slices: int,
) -> Sequence[tuple[int, float, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to open uploaded video for processing")

    frame_interval = max(int(fps * sample_every_seconds), 1)
    frame_index = 0
    captured: list[tuple[int, float, Any]] = []

    try:
        while cap.isOpened() and len(captured) < max_slices:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame_index % frame_interval == 0:
                timestamp_sec = frame_index / fps if fps else 0.0
                captured.append((len(captured), float(timestamp_sec), frame))
            frame_index += 1
    finally:
        cap.release()

    return tuple(captured)


def clear_previous_slices(slice_folder: Path) -> None:
    for existing in slice_folder.glob("*.jpg"):
        try:
            existing.unlink()
        except FileNotFoundError:
            continue


def build_slice_payloads(
    raw_frames: Iterable[tuple[int, float, Any]],
    slice_folder: Path,
) -> list[dict]:
    frames = list(raw_frames)
    if not frames:
        return []

    max_workers = min(len(frames), max(os.cpu_count() or 1, 1))

    def prepare(entry: tuple[int, float, Any]) -> dict:
        index, timestamp_sec, frame = entry
        resized = resize_frame_for_preview(frame)
        jpg_bytes = encode_frame_to_jpeg(resized)
        slice_path = slice_folder / f"slice_{index:03d}.jpg"
        slice_path.write_bytes(jpg_bytes)
        return {
            "index": index,
            "time_seconds": float(timestamp_sec),
            "timestamp": human_timestamp(timestamp_sec),
            "image_base64": base64.b64encode(jpg_bytes).decode("utf-8"),
            "path": str(slice_path),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        slices = list(executor.map(prepare, frames))

    return sorted(slices, key=lambda entry: entry["index"])


def resize_frame_for_preview(frame):
    height, width = frame.shape[:2]
    max_dim = 320
    scale = min(max_dim / max(height, width), 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (math.floor(width * scale), math.floor(height * scale)))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def encode_frame_to_jpeg(frame) -> bytes:
    success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        raise ValueError("Failed to encode frame to JPEG")
    return buffer.tobytes()
