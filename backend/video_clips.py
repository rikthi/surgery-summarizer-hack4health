from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence

from . import video_processing as vp
from .state import CLIP_DIR

LOG = logging.getLogger("backend.clips")
FFMPEG_PATH = shutil.which("ffmpeg")


def slugify_phase(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "phase"


def humanize_duration(start: float, end: float) -> float:
    return max(end - start, 0.0) + 1.0


def generate_phase_clips(video_path: Path, segments: Sequence[Dict[str, Any]], job_id: str) -> List[Dict[str, Any]]:
    if not segments:
        return []

    clip_root = CLIP_DIR / job_id
    clip_root.mkdir(parents=True, exist_ok=True)
    for existing in clip_root.glob("*"):
        try:
            existing.unlink()
        except FileNotFoundError:
            continue

    fps, total_frames, duration = vp.probe_video_metrics(video_path)
    LOG.info(
        "Preparing %d phase clips (fps=%.2f, frames=%d, duration=%.2fs)",
        len(segments),
        fps,
        total_frames,
        duration,
    )

    clips: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments):
        start = max(float(segment["start_second"]), 0.0)
        end = max(float(segment["end_second"]), start)
        duration_seconds = humanize_duration(start, end)
        slug = slugify_phase(segment.get("phase", f"phase-{index}"))
        filename = f"{index:02d}_{slug}.mp4"
        clip_path = clip_root / filename

        success = False
        if FFMPEG_PATH:
            success = _extract_with_ffmpeg(video_path, start, duration_seconds, clip_path)
        if not success:
            success = _extract_with_cv2(video_path, start, end, fps, clip_path)

        if not success:
            LOG.warning("Failed to generate clip for phase %s (%s)", segment.get("phase"), filename)
            continue

        clip_record = {
            **segment,
            "duration_seconds": duration_seconds,
            "file_name": filename,
            "video_path": str(clip_path),
            "video_url": f"/jobs/{job_id}/clips/{filename}",
            "download_url": f"/jobs/{job_id}/clips/{filename}?download=1",
        }
        clips.append(clip_record)
        LOG.debug("Generated clip %s", clip_path)

    return clips


def _extract_with_ffmpeg(video_path: Path, start: float, duration: float, destination: Path) -> bool:
    cmd = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        str(destination),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, check=False)
    except Exception as exc:  # pragma: no cover - depends on runtime ffmpeg availability
        LOG.debug("ffmpeg invocation failed: %s", exc)
        return False

    if completed.returncode != 0:
        LOG.debug("ffmpeg exited with %s: %s", completed.returncode, completed.stderr.decode("utf-8", "ignore"))
        return False

    return destination.exists() and destination.stat().st_size > 0


def _extract_with_cv2(video_path: Path, start: float, end: float, fps: float, destination: Path) -> bool:
    vp.ensure_video_tooling_available()
    cv2 = vp.cv2
    if cv2 is None:
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = max(int(math.floor(start * fps)), 0)
    end_frame = max(int(math.ceil(end * fps)), start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(destination), fourcc, fps, (width, height))

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        current = start_frame
        while current <= end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            writer.write(frame)
            current += 1
    finally:
        writer.release()
        cap.release()

    return destination.exists() and destination.stat().st_size > 0