from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

import logging

LOG = logging.getLogger("backend.video")

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None


def ensure_video_tooling_available() -> None:
    if cv2 is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Video processing dependencies are missing. Install with "
                "`pip install opencv-python` on backend."
            ),
        )


def probe_video_metrics(video_path: Path) -> tuple[float, int, float]:
    """Extract FPS, frame count, and duration from a video file."""
    ensure_video_tooling_available()
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
