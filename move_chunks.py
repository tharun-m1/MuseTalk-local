import os
import shutil

src_base = "cached_frames"
dst_base = "talking_frames"

# Move audio.wav from chunk_009 to all other chunk directories in talking_frames
src_audio = os.path.join(src_base, "chunk_009", "audio.wav")

if not os.path.exists(src_audio):
    print(f"Source audio not found: {src_audio}")
    exit(1)

for i in range(1, 21):  # 1 to 20 inclusive
    if i == 4:
        continue  # skip chunk_004 if needed
    dst_dir = os.path.join(dst_base, f"chunk_{i:03d}")
    dst_audio = os.path.join(dst_dir, "audio.wav")
    if not os.path.exists(dst_dir):
        print(f"Destination chunk does not exist: {dst_dir}")
        continue
    shutil.copy2(src_audio, dst_audio)
    print(f"Copied audio.wav to {dst_audio}")
