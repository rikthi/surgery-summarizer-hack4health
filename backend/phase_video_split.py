import os
import sys
from moviepy import VideoFileClip, concatenate_videoclips

PREDICTIONS_DIR = os.path.join("..", "model", "predictions")
RAW_OUTPUT_DIR = "phase_outputs_raw"
FINAL_OUTPUT_DIR = "phase_outputs_final"
MIN_SEGMENT_LENGTH = 12  # seconds


# ----------------------------
# Load Predictions
# ----------------------------
def load_predictions(pred_path):
    preds = []
    with open(pred_path, "r") as f:
        for line in f:
            sec_str, phase = line.strip().split("\t")
            preds.append((int(sec_str), phase))
    return preds


# ----------------------------
# Group contiguous segments
# ----------------------------
def group_segments(preds):
    segments = {}
    prev_phase = preds[0][1]
    start = preds[0][0]

    for i in range(1, len(preds)):
        sec, phase = preds[i]
        if phase != prev_phase:
            end = preds[i - 1][0]
            segments.setdefault(prev_phase, []).append((start, end))
            prev_phase = phase
            start = sec

    # Final segment
    last_sec, last_phase = preds[-1]
    segments.setdefault(last_phase, []).append((start, last_sec))

    return segments


# ----------------------------
# Filter segments too short
# ----------------------------
def filter_segments(segs):
    filtered = {}
    for phase, arr in segs.items():
        valid = [(s, e) for (s, e) in arr if (e - s + 1) >= MIN_SEGMENT_LENGTH]
        if valid:
            filtered[phase] = valid
    return filtered


# ----------------------------
# Extract raw clips (FAST + FIXED)
# ----------------------------
def extract_raw_clips(video_path, video_id, segments):
    raw_folder = os.path.join(RAW_OUTPUT_DIR, video_id)
    os.makedirs(raw_folder, exist_ok=True)

    print("\n📼 Loading video:", video_path)
    main_clip = VideoFileClip(video_path)
    video_duration = main_clip.duration  # ⬅ CRITICAL FIX

    for phase, runs in segments.items():
        print(f"\n🎬 Extracting clips for phase: {phase}")

        for i, (s, e) in enumerate(runs, start=1):

            # -----------------------
            # Prevent out-of-range error
            # -----------------------
            e = min(e, int(video_duration) - 1)
            s = max(0, s)

            print(f"   - Cutting clip {i}: {s}s → {e}s (video ends at {video_duration:.2f}s)")

            sub = main_clip.subclipped(s, e)

            out_path = os.path.join(raw_folder, f"{phase}_{i}.mp4")

            sub.write_videofile(
                out_path,
                codec="libx264",
                audio=False,
                fps=main_clip.fps
            )

    main_clip.close()
    print("\n✅ Raw clips extracted.")
    return raw_folder


# ----------------------------
# Merge raw clips per phase
# ----------------------------
def merge_phase_videos(video_id, raw_folder):
    final_folder = os.path.join(FINAL_OUTPUT_DIR, video_id)
    os.makedirs(final_folder, exist_ok=True)

    phase_groups = {}

    for f in sorted(os.listdir(raw_folder)):
        if not f.endswith(".mp4"):
            continue
        phase = f.split("_")[0]
        phase_groups.setdefault(phase, []).append(os.path.join(raw_folder, f))

    for phase, clip_paths in phase_groups.items():
        print(f"\n🧩 Merging {len(clip_paths)} clips for: {phase}")

        clips = [VideoFileClip(p) for p in clip_paths]
        merged = concatenate_videoclips(clips, method="compose")

        out_path = os.path.join(final_folder, f"{phase}.mp4")

        merged.write_videofile(
            out_path,
            codec="libx264",
            audio=False,
            fps=clips[0].fps
        )

        for c in clips:
            c.close()

    print("\n🎉 All merged phase videos created!")


# ----------------------------
# Main
# ----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python phase_video_split.py <video.mp4>")
        return

    # input video
    video_filename = sys.argv[1]
    video_path = os.path.join(PREDICTIONS_DIR, video_filename)

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # prediction file
    pred_path = os.path.join(PREDICTIONS_DIR, f"{video_id}-PREDICTED-phases.txt")
    if not os.path.exists(pred_path):
        print(f"❌ Prediction file not found: {pred_path}")
        return

    print("\n📄 Loading predictions...")
    preds = load_predictions(pred_path)

    print("🔍 Grouping segments...")
    grouped = group_segments(preds)

    print("🧹 Filtering short segments...")
    filtered = filter_segments(grouped)

    print("✂ Extracting raw phase clips...")
    raw_folder = extract_raw_clips(video_path, video_id, filtered)

    print("\n🔗 Merging phase clips...")
    merge_phase_videos(video_id, raw_folder)

    print("\n✅ COMPLETE!")


if __name__ == "__main__":
    main()
