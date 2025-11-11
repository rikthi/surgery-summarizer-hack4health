def build_llm_stub_response(analysis: dict) -> dict:
    """Construct a placeholder payload that mirrors the expected LLM output.

    Parameters
    ----------
    analysis: dict
        Output from ``extract_video_slices``. The ``analysis['slices']`` list
        already contains base64-encoded preview images suitable for grounding
        multimodal LLM prompts.
    """
    frame_slices = analysis.get("slices", [])
    highlights = [
        {
            "timestamp": slice_info["timestamp"],
            "time_seconds": slice_info["time_seconds"],
            "label": f"Stub highlight at {slice_info['timestamp']}",
        }
        for slice_info in frame_slices[:5]
    ]

    summary_lines = [
        "LLM summary placeholder:",
        "- Critical anatomical landmarks detected.",
        "- No complications flagged in stub.",
        "- Replace this block with real model output.",
    ]

    return {
        "summary_text": "\n".join(summary_lines),
        "highlights": highlights,
        "frame_slices": frame_slices,
        "video_metadata": {
            "duration_seconds": analysis.get("duration_seconds", 0.0),
            "frame_count": analysis.get("frame_count", 0),
            "fps": analysis.get("fps", 0.0),
            "sample_interval_seconds": analysis.get("sample_interval_seconds", 0),
        },
    }
