from __future__ import annotations

import importlib
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .state import BASE_DIR
from .utils import human_timestamp

LOG = logging.getLogger("backend.model")

MODEL_DIR = BASE_DIR.parent / "model"
PREDICTIONS_DIR = MODEL_DIR / "Predictions"
MIN_SEGMENT_LENGTH = 12  # seconds
DEFAULT_IMPL = os.environ.get("MODEL_INFER_IMPL", "fast").lower()
DEFAULT_SAMPLE_FPS = float(os.environ.get("MODEL_SAMPLE_FPS", "1"))
CHUNK_SIZE_OVERRIDE = os.environ.get("MODEL_CHUNK_SIZE")


def _resolve_module_name() -> str:
    if DEFAULT_IMPL in {"slow", "chunked"}:
        return "model.infer_lstm"
    return "model.infer_lstm_fast"


@lru_cache(maxsize=1)
def _load_infer_module():
    module_name = _resolve_module_name()
    LOG.info("Using inference module: %s", module_name)
    return importlib.import_module(module_name)


@lru_cache(maxsize=1)
def _load_model_and_labels():
    module = _load_infer_module()
    model, int_to_label = module.load_model_and_labels()
    label_lookup = {int(idx): label for idx, label in int_to_label.items()}
    return model, int_to_label, label_lookup


def _persist_prediction_file(video_path: Path, per_second: Sequence[Dict[str, Any]]) -> None:
    """Write per-second predictions to model/Predictions for inspection."""

    if not per_second:
        return

    try:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        LOG.exception("Unable to create predictions directory at %s", PREDICTIONS_DIR)
        return

    base_name = video_path.stem or "prediction"
    output_path = PREDICTIONS_DIR / f"{base_name}-PREDICTED-phases.txt"

    try:
        with output_path.open("w", encoding="utf-8") as outfile:
            for entry in per_second:
                outfile.write(f"{entry['second']}\t{entry['phase']}\n")
        LOG.info("Saved per-second predictions to %s", output_path)
    except Exception:
        LOG.exception("Failed adding per-second predictions file at %s", output_path)


def group_segments(per_second: Sequence[int], label_lookup: Dict[int, str]) -> List[Dict[str, Any]]:
    if not per_second:
        return []

    segments: List[Dict[str, Any]] = []
    current_label = per_second[0]
    start_idx = 0

    for idx in range(1, len(per_second)):
        if per_second[idx] != current_label:
            segment = _build_segment(current_label, start_idx, idx - 1, label_lookup)
            segments.append(segment)
            current_label = per_second[idx]
            start_idx = idx

    segments.append(_build_segment(current_label, start_idx, len(per_second) - 1, label_lookup))
    return segments


def _build_segment(label_idx: int, start_second: int, end_second: int, label_lookup: Dict[int, str]) -> Dict[str, Any]:
    duration = (end_second - start_second) + 1
    return {
        "phase": label_lookup.get(label_idx, str(label_idx)),
        "start_second": start_second,
        "end_second": end_second,
        "duration_seconds": duration,
        "start_timestamp": human_timestamp(start_second),
        "end_timestamp": human_timestamp(end_second),
    }


def filter_segments(segments: Sequence[Dict[str, Any]], min_length: int = MIN_SEGMENT_LENGTH) -> List[Dict[str, Any]]:
    return [segment for segment in segments if segment["duration_seconds"] >= min_length]


def combine_phase_segments(
    segments: Sequence[Dict[str, Any]],
    min_length: int = MIN_SEGMENT_LENGTH,
) -> List[Dict[str, Any]]:
    """Group qualifying segments per phase and merge them for downstream clip generation."""

    if not segments:
        return []

    grouped: dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        if segment["duration_seconds"] >= min_length:
            grouped[segment["phase"]].append(segment)

    combined: List[Dict[str, Any]] = []
    for phase, phase_segments in grouped.items():
        ordered = sorted(phase_segments, key=lambda seg: seg["start_second"])
        start_second = ordered[0]["start_second"]
        end_second = ordered[-1]["end_second"]
        combined.append(
            {
                "phase": phase,
                "start_second": start_second,
                "end_second": end_second,
                "start_timestamp": human_timestamp(start_second),
                "end_timestamp": human_timestamp(end_second),
                "duration_seconds": sum(seg["duration_seconds"] for seg in ordered),
                "segments": ordered,
            }
        )

    return sorted(combined, key=lambda seg: seg["start_second"])


@dataclass
class PhaseInferenceResult:
    per_second: List[Dict[str, Any]]
    segments: List[Dict[str, Any]]
    filtered_segments: List[Dict[str, Any]]
    combined_segments: List[Dict[str, Any]]

    @property
    def has_content(self) -> bool:
        return bool(self.segments)


def infer_procedure_phases(video_path: Path) -> PhaseInferenceResult:
    module = _load_infer_module()
    model, int_to_label, label_lookup = _load_model_and_labels()

    predict_kwargs: Dict[str, Any] = {
        "sample_fps": DEFAULT_SAMPLE_FPS,
        "save_output": False,
        "return_data": True,
        "show_progress": False,
    }
    if module.__name__.endswith("infer_lstm") and CHUNK_SIZE_OVERRIDE:
        predict_kwargs["chunk_size"] = int(CHUNK_SIZE_OVERRIDE)

    LOG.info("Starting inference for %s using %s", video_path, module.__name__)
    import time
    t0 = time.time()
    result = module.predict_video(model, str(video_path), int_to_label, **predict_kwargs)
    elapsed = time.time() - t0
    try:
        pred_count = len(result.get("per_second_indices", [])) if result else 0
    except Exception:
        pred_count = 0
    LOG.info(
        "Inference finished for %s using %s: predictions=%d time=%.2fs",
        video_path,
        module.__name__,
        pred_count,
        elapsed,
    )

    if not result or not result.get("per_second_indices"):
        raise RuntimeError("Inference module returned no per-second predictions")

    per_second_indices = [int(idx) for idx in result["per_second_indices"]]
    per_second = [
        {
            "second": idx,
            "phase_index": label_idx,
            "phase": label_lookup.get(label_idx, str(label_idx)),
        }
        for idx, label_idx in enumerate(per_second_indices)
    ]

    segments = group_segments(per_second_indices, label_lookup)
    filtered = filter_segments(segments)
    selection_source = filtered or segments
    combined_segments = combine_phase_segments(selection_source)

    _persist_prediction_file(video_path, per_second)

    return PhaseInferenceResult(
        per_second=per_second,
        segments=segments,
        filtered_segments=filtered,
        combined_segments=combined_segments,
    )


def warm_start_model() -> None:
    """Load the configured inference module/model so TensorFlow spins up early."""

    try:
        _load_model_and_labels()
        LOG.info("Model warm-start complete")
    except Exception:
        LOG.exception("Model warm-start failed")
    # Log TensorFlow GPU/CPU availability if TF is installed
    try:
        import importlib

        tf = importlib.import_module("tensorflow")
        try:
            tf_version = getattr(tf, "__version__", "<unknown>")
            # Newer TF API: list_physical_devices
            gpus = []
            try:
                gpus = tf.config.list_physical_devices("GPU") or []
            except Exception:
                # Fallback for older TF
                try:
                    if tf.test.is_gpu_available():
                        gpus = ["GPU (legacy-detect)"]
                except Exception:
                    gpus = []

            if gpus:
                LOG.info("TensorFlow %s detected GPUs: %s", tf_version, [getattr(d, 'name', str(d)) for d in gpus])
            else:
                LOG.info("TensorFlow %s running without GPUs (CPU-only)", tf_version)
        except Exception:
            LOG.exception("Unable to inspect TensorFlow devices")
    except Exception:
        LOG.debug("TensorFlow not importable during warm-start; skipping GPU/CPU log")