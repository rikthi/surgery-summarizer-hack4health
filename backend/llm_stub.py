from __future__ import annotations

from typing import Any, Dict, List

from .model_inference import PhaseInferenceResult


def build_llm_stub_response(phase_result: PhaseInferenceResult) -> dict:
    """Compose the response payload from model predictions."""

    segments_for_summary = phase_result.filtered_segments or phase_result.segments
    longest_segments = phase_result.longest_segments
    summary_lines = build_summary_lines(segments_for_summary)

    return {
        "summary_text": "\n".join(summary_lines),
        "phase_predictions": phase_result.per_second,
        "phase_segments": phase_result.segments,
        "phase_segments_filtered": phase_result.filtered_segments,
        "phase_segments_longest": longest_segments,
    }


def build_summary_lines(segments: List[Dict[str, Any]]) -> List[str]:
    if not segments:
        return [
            "Model ran successfully but no stable surgical phases exceeded the minimum length threshold.",
            "Review raw per-second predictions for finer detail.",
        ]

    # Group segments by phase and keep only the longest instance of each phase
    phase_best: Dict[str, Dict[str, Any]] = {}
    for segment in segments:
        phase = segment["phase"]
        if phase not in phase_best or segment["duration_seconds"] > phase_best[phase]["duration_seconds"]:
            phase_best[phase] = segment

    # Maintain original order by keeping first occurrence order
    seen_phases: List[str] = []
    phase_to_segment: Dict[str, Dict[str, Any]] = {}
    for segment in segments:
        phase = segment["phase"]
        if phase not in seen_phases:
            seen_phases.append(phase)
            phase_to_segment[phase] = phase_best[phase]

    lines = ["Detected surgical phases:"]
    for phase in seen_phases:
        segment = phase_to_segment[phase]
        lines.append(
            "- {phase}: {start} → {end} (~{duration}s)".format(
                phase=segment["phase"],
                start=segment["start_timestamp"],
                end=segment["end_timestamp"],
                duration=segment["duration_seconds"],
            )
        )
    return lines



