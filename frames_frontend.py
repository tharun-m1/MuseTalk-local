import cv2
import os

VIDEO_FPS = 12  # Reference FPS from other scripts
OUTPUT_DIR = "silent_frames"

def extract_frames(video_path, output_dir=OUTPUT_DIR, target_fps=VIDEO_FPS):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(original_fps / target_fps)) if target_fps < original_fps else 1

    frame_idx = 0
    saved_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            filename = os.path.join(output_dir, f"{saved_idx:08d}.png")
            cv2.imwrite(filename, frame)
            saved_idx += 1
        frame_idx += 1
    cap.release()
    print(f"✅ Extracted {saved_idx-1} frames to {output_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python frames_frontend.py <video_filename>")
        exit(1)
    # Video is at the same level as this script
    video_path = sys.argv[1]
    extract_frames(video_path)
