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
BATCH_SIZE = 8       # smaller batch to save RAM
SMOOTH_WINDOW = 3
CHUNK_SIZE = 3000    # process 3000 frames at a time  (Might cause ram allocation issues!!)


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
    print(f"\n🎥 Predicting {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    predictions = []
    frames_buffer = []
    frame_index = 0

    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_buffer.append(preprocess_frame(frame))
        frame_index += 1
        pbar.update(1)


        if len(frames_buffer) >= CHUNK_SIZE:
            preds = predict_chunk(model, frames_buffer)
            predictions.extend(preds)
            frames_buffer = frames_buffer[-SEQ_LEN:]


    if len(frames_buffer) >= SEQ_LEN:
        preds = predict_chunk(model, frames_buffer)
        predictions.extend(preds)

    pbar.close()
    cap.release()


    predictions = np.array(predictions)
    smoothed = temporal_smooth(predictions, window=SMOOTH_WINDOW)
    predicted_classes = np.argmax(smoothed, axis=1)


    output_path = os.path.splitext(video_path)[0] + "-PREDICTED-phases.txt"
    with open(output_path, "w") as f:
        for i, class_idx in enumerate(predicted_classes):
            f.write(f"{i}\t{int_to_label[str(class_idx)]}\n")

    print(f"Saved predictions → {output_path}")


def predict_chunk(model, frames):

    seqs = []
    for i in range(0, len(frames) - SEQ_LEN + 1):
        seq = np.stack(frames[i:i + SEQ_LEN])
        seqs.append(seq)
    X = np.array(seqs)
    preds = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    return preds


def main():
    if not os.path.exists(MODEL_PATH):
        print(" Model not found.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading label map...")
    with open(LABEL_MAP_PATH, "r") as f:
        label_data = json.load(f)
    int_to_label = label_data["int_to_label"]

    videos = [os.path.join(PREDICTIONS_DIR, v)
              for v in os.listdir(PREDICTIONS_DIR)
              if v.lower().endswith((".mp4", ".avi", ".mov"))]

    if not videos:
        print("No videos found in 'predictions/'.")
        return

    for video_path in videos:
        predict_video(model, video_path, int_to_label)


if __name__ == "__main__":
    main()
