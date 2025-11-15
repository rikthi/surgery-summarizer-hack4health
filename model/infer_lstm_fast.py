from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import logging
import time

MODULE_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODULE_DIR / "model_phase2_lstm.keras"
LABEL_MAP_PATH = MODULE_DIR / "label_map.json"
PREDICTIONS_DIR = MODULE_DIR / "predictions"
SEQ_LEN = 8
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SMOOTH_WINDOW = 3
SAMPLE_FPS = 1   # Only keep 1 frame per second


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, IMG_SIZE)
    return tf.keras.applications.efficientnet.preprocess_input(frame_resized)


def temporal_smooth(probabilities: np.ndarray, window: int = 5) -> np.ndarray:
    smoothed = np.zeros_like(probabilities)
    for i in range(len(probabilities)):
        start = max(0, i - window)
        end = min(len(probabilities), i + window + 1)
        smoothed[i] = np.mean(probabilities[start:end], axis=0)
    return smoothed


def load_model_and_labels(
    model_path: Optional[str | Path] = None,
    label_map_path: Optional[str | Path] = None,
):
    model_path = Path(model_path) if model_path else MODEL_PATH
    label_map_path = Path(label_map_path) if label_map_path else LABEL_MAP_PATH
    model = tf.keras.models.load_model(str(model_path))
    with label_map_path.open("r", encoding="utf-8") as f:
        label_data = json.load(f)
    return model, label_data["int_to_label"]


def predict_video(
    model,
    video_path: str | Path,
    int_to_label: Dict[str, str],
    sample_fps: float = SAMPLE_FPS,
    save_output: bool = True,
    output_path: Optional[str | Path] = None,
    return_data: bool = False,
    show_progress: bool = True,
):
    video_path = str(video_path)
    video_name = os.path.basename(video_path)

    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + "-PREDICTED-phases.txt"

        sample_method = os.environ.get("MODEL_SAMPLE_METHOD", "seek").lower()
        LOG = logging.getLogger("model.infer_lstm_fast")
        LOG.info("Sampling method: %s (set MODEL_SAMPLE_METHOD to change)", sample_method)
    if save_output and os.path.exists(output_path):
        print(f"⏩ Skipping {video_name} (prediction already exists)")
        if return_data:
            raise RuntimeError("Predictions already exist on disk; set save_output=False to recompute in-memory.")
        return None

    LOG = logging.getLogger("model.infer_lstm_fast")
    LOG.info("Predicting %s", video_name)
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or sample_fps
    frame_interval = max(int(original_fps / sample_fps), 1)
    print(f"Original FPS: {original_fps:.2f} → sampling 1 frame every {frame_interval} frames")

    frames: list[np.ndarray] = []
    frame_id = 0
    decode_time = 0.0
    preproc_time = 0.0
    sample_method = os.environ.get("MODEL_SAMPLE_METHOD", "seek").lower()
    # If using seek-based sampling, total progress should match target frames rather than all frames
    import math
    targets_total = (total_frames + frame_interval - 1) // frame_interval if (sample_method == "seek" and total_frames) else total_frames
    pbar = tqdm(total=targets_total, desc="Sampling frames") if show_progress else None

    if sample_method == "seek" and total_frames:
        # Seek to each target frame index; this avoids reading every frame
        target_indices = list(range(0, total_frames, frame_interval))
        for idx in target_indices:
            t0 = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(min(idx, max(total_frames - 1, 0))))
            ret, frame = cap.read()
            decode_time += time.time() - t0
            if not ret or frame is None:
                LOG.debug("Seek failed at frame=%s; falling back to sequential capture", idx)
                # fallback to sequential mode
                sample_method = "sequential"
                break

            if pbar:
                pbar.update(1)

            t0 = time.time()
            frames.append(preprocess_frame(frame))
            preproc_time += time.time() - t0
            frame_id += 1

    if sample_method != "seek":
        # sequential capture fallback: visit every frame and keep those that match the interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            decode_time += time.time() - t0
            if not ret:
                break
            if frame_id % frame_interval == 0:
                t0 = time.time()
                frames.append(preprocess_frame(frame))
                preproc_time += time.time() - t0
            frame_id += 1
            if pbar:
                pbar.update(1)
    cap.release()
    if pbar:
        pbar.close()

        LOG.info("Sampling times - decoding: %.2fs ; preprocessing: %.2fs", decode_time, preproc_time)

        LOG.info("Total sampled frames: %d", len(frames))

    if len(frames) < SEQ_LEN:
        message = "Not enough frames for a full sequence."
        print(message)
        if return_data:
            raise RuntimeError(message)
        return None

    sequences = [np.stack(frames[i : i + SEQ_LEN]) for i in range(0, len(frames) - SEQ_LEN + 1)]
    X = np.asarray(sequences)

    LOG.info("Running model on %d sequences (batch_size=%d)", len(X), BATCH_SIZE)
    t0 = time.time()
    preds = model.predict(X, batch_size=BATCH_SIZE, verbose=0 if not show_progress else 1)
    dt = time.time() - t0
    LOG.info("Model predict finished: video=%s sequences=%d time=%.2fs", video_name, len(X), dt)

    smoothed_preds = temporal_smooth(preds, window=SMOOTH_WINDOW)
    predicted_classes = np.argmax(smoothed_preds, axis=1)
    final_predictions = [predicted_classes[0]] * (SEQ_LEN - 1) + predicted_classes.tolist()

    if save_output:
        with open(output_path, "w", encoding="utf-8") as f:
            for i, label_idx in enumerate(final_predictions):
                label_name = int_to_label[str(int(label_idx))]
                f.write(f"{i}\t{label_name}\n")
        print(f"Saved 1 FPS predictions → {output_path}")

    total_time = time.time() - start_time
    LOG.info("Prediction complete: video=%s total_seconds=%.2fs total_predictions=%d", video_name, total_time, len(final_predictions))

    if return_data:
        return {
            "per_second_indices": [int(idx) for idx in final_predictions],
            "per_second_labels": [int_to_label[str(int(idx))] for idx in final_predictions],
            "sequence_indices": [int(idx) for idx in predicted_classes.tolist()],
            "sample_fps": sample_fps,
        }

    return None


def main():
    if not MODEL_PATH.exists():
        print("Model not found.")
        return

    print("Loading model...")
    model, int_to_label = load_model_and_labels()

    print("Loading videos...")
    video_files = [
        os.path.join(PREDICTIONS_DIR, f)
        for f in os.listdir(PREDICTIONS_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    if not video_files:
        print("No videos found in 'predictions/'.")
        return

    for video_path in video_files:
        predict_video(model, video_path, int_to_label)


if __name__ == "__main__":
    main()
