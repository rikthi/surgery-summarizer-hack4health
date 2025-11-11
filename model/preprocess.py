import os
import glob
import cv2
import numpy as np
from tqdm import tqdm


DATA_DIR = 'data'
OUTPUT_DIR = 'data_preprocessed'
IMG_SIZE = (224, 224)
FRAME_SUBSAMPLE_RATE = 30  # Grab 1 frame every 30 frames




def get_video_paths(data_dir):

    video_paths = []
    label_files = sorted(glob.glob(os.path.join(data_dir, "*-phase.txt")))

    if not label_files:
        print(f"Error: No '-phase.txt' files found in {data_dir}")
        return []

    for label_file in label_files:
        try:
            label_id_stem = os.path.basename(label_file).split('-phase.txt')[0]
            number_part = label_id_stem.replace('video', '')
            video_number_int = int(number_part)
            video_stem = f"video{video_number_int}"
        except ValueError:
            print(f"Warning: Could not parse video ID from {label_file}. Skipping.")
            continue

        video_path_glob = glob.glob(os.path.join(data_dir, f"{video_stem}.*"))
        video_path = [v for v in video_path_glob if not v.endswith('.txt')]

        if not video_path:
            print(f"Warning: No video file found for label {label_file}. Skipping.")
            continue

        video_paths.append((video_path[0], label_file))

    return video_paths


def extract_frames(video_path, label_file, video_id):



    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            try:
                frame_idx_str, phase_name = line.strip().split('\t')
                labels[int(frame_idx_str)] = phase_name
            except ValueError:
                pass

    if not labels:
        print(f"No labels found in {label_file}. Skipping video.")
        return


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    video_output_dir = os.path.join(OUTPUT_DIR, video_id)
    os.makedirs(video_output_dir, exist_ok=True)

    frame_count = 0
    saved_count = 0


    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Processing {video_id}")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break


        if frame_count % FRAME_SUBSAMPLE_RATE == 0:

            phase_name = labels.get(frame_count)
            if phase_name is not None:

                frame_resized = cv2.resize(frame, (IMG_SIZE[1], IMG_SIZE[0]))


                save_name = f"{frame_count:08d}_{phase_name}.jpg"
                save_path = os.path.join(video_output_dir, save_name)

                cv2.imwrite(save_path, frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                saved_count += 1

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"Finished {video_id}: Saved {saved_count} frames.")


def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at '{DATA_DIR}'")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_list = get_video_paths(DATA_DIR)

    if not video_list:
        print("No videos found to process.")
        return

    print(f"Found {len(video_list)} videos to pre-process.")

    for video_path, label_file in video_list:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        extract_frames(video_path, label_file, video_id)

    print("\n--- Pre-processing Complete! ---")
    print(f"All sampled frames are saved in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()