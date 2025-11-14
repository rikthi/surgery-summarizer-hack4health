import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

MODEL_PATH = "model_phase2_lstm.keras"
LABEL_MAP_PATH = "label_map.json"
PREDICTIONS_DIR = "predictions"
SEQ_LEN = 8
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SMOOTH_WINDOW = 3
SAMPLE_FPS = 1   # Only keep 1 frame per second


def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, IMG_SIZE)
    return tf.keras.applications.efficientnet.preprocess_input(frame_resized)


def temporal_smooth(probabilities, window=5):
    smoothed = np.zeros_like(probabilities)
    for i in range(len(probabilities)):
        start = max(0, i - window)
        end = min(len(probabilities), i + window + 1)
        smoothed[i] = np.mean(probabilities[start:end], axis=0)
    return smoothed


def predict_video(model, video_path, int_to_label):


    output_path = os.path.splitext(video_path)[0] + "-PREDICTED-phases.txt"
    if os.path.exists(output_path):
        print(f"⏩ Skipping {os.path.basename(video_path)} (prediction already exists)")
        return


    print(f"\n🎥 Predicting {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / SAMPLE_FPS)
    print(f"Original FPS: {original_fps:.2f} → sampling 1 frame every {frame_interval} frames")

    frames = []
    frame_id = 0
    pbar = tqdm(total=total_frames, desc="Sampling frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            frames.append(preprocess_frame(frame))
        frame_id += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    print(f"Total sampled frames: {len(frames)}")

    if len(frames) < SEQ_LEN:
        print("Not enough frames for a full sequence.")
        return

    sequences = []
    for i in range(0, len(frames) - SEQ_LEN + 1):
        seq = np.stack(frames[i:i + SEQ_LEN])
        sequences.append(seq)
    X = np.array(sequences)

    print(f"Running model on {len(X)} sequences...")
    preds = model.predict(X, batch_size=BATCH_SIZE, verbose=1)

    smoothed_preds = temporal_smooth(preds, window=SMOOTH_WINDOW)
    predicted_classes = np.argmax(smoothed_preds, axis=1)

    final_predictions = [predicted_classes[0]] * (SEQ_LEN - 1) + predicted_classes.tolist()

    # Save output
    with open(output_path, "w") as f:
        for i, label_idx in enumerate(final_predictions):
            f.write(f"{i}\t{int_to_label[str(label_idx)]}\n")

    print(f"Saved 1 FPS predictions → {output_path}")


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading label map...")
    with open(LABEL_MAP_PATH, "r") as f:
        label_data = json.load(f)
    int_to_label = label_data["int_to_label"]

    video_files = [os.path.join(PREDICTIONS_DIR, f)
                   for f in os.listdir(PREDICTIONS_DIR)
                   if f.lower().endswith((".mp4", ".avi", ".mov"))]

    if not video_files:
        print("No videos found in 'predictions/'.")
        return

    for video_path in video_files:
        predict_video(model, video_path, int_to_label)


if __name__ == "__main__":
    main()
