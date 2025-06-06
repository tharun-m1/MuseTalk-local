import cv2
import os
import argparse
from pathlib import Path
import time

def extract_frames_from_video(video_path, output_folder, target_fps=12, resize_to=(640, 360), jpeg_quality=85):
    """
    Extract frames from video and save them in the format expected by dummy_server.py
    
    Args:
        video_path: Path to the input video file
        output_folder: Folder to save extracted frames
        target_fps: Target FPS for frame extraction (default: 12)
        resize_to: Target size (width, height) to match WebRTC response (default: 640x360)
        jpeg_quality: JPEG compression quality (default: 85)
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return False
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    
    print(f"📹 Video Info:")
    print(f"   - Original FPS: {original_fps:.2f}")
    print(f"   - Total frames: {total_frames}")
    print(f"   - Duration: {duration:.2f} seconds")
    print(f"   - Target FPS: {target_fps}")
    print(f"   - Output size: {resize_to[0]}x{resize_to[1]}")
    
    # Calculate frame sampling interval
    if target_fps >= original_fps:
        # If target FPS is higher or equal, use all frames
        frame_interval = 1
        print(f"   - Using every frame (interval: 1)")
    else:
        # If target FPS is lower, skip frames
        frame_interval = int(original_fps / target_fps)
        print(f"   - Sampling every {frame_interval} frames")
    
    extracted_count = 0
    frame_number = 0
    
    print(f"\n🎬 Starting frame extraction...")
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Only process frames at the target interval
        if frame_number % frame_interval == 0:
            # Resize frame to match WebRTC response format
            resized_frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            
            # Generate filename with zero-padded numbering
            frame_filename = f"frame_{extracted_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            
            # Save frame with specified JPEG quality
            success = cv2.imwrite(
                frame_path, 
                resized_frame, 
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            
            if success:
                extracted_count += 1
                if extracted_count % 50 == 0:  # Progress update every 50 frames
                    print(f"   ✅ Extracted {extracted_count} frames...")
            else:
                print(f"   ❌ Failed to save frame: {frame_filename}")
        
        frame_number += 1
    
    cap.release()
    
    end_time = time.time()
    extraction_time = end_time - start_time
    
    print(f"\n✅ Frame extraction completed!")
    print(f"   - Extracted frames: {extracted_count}")
    print(f"   - Processing time: {extraction_time:.2f} seconds")
    print(f"   - Output folder: {output_folder}")
    print(f"   - Effective FPS: {extracted_count / duration:.2f}")
    
    # Verify some extracted frames
    print(f"\n🔍 Verification:")
    sample_files = list(Path(output_folder).glob("frame_*.jpg"))[:5]
    for sample_file in sample_files:
        file_size = sample_file.stat().st_size
        print(f"   - {sample_file.name}: {file_size} bytes")
    
    return True

def validate_video_file(video_path):
    """Validate if the video file exists and is readable."""
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file does not exist: {video_path}")
        return False
    
    # Try to open with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file: {video_path}")
        return False
    
    cap.release()
    return True

def main():
    parser = argparse.ArgumentParser(description='Extract frames from video for dummy lip-sync server')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', default='./pregenerated_frames', 
                       help='Output folder for frames (default: ./pregenerated_frames)')
    parser.add_argument('--fps', '-f', type=int, default=12, 
                       help='Target FPS for frame extraction (default: 12)')
    parser.add_argument('--width', '-w', type=int, default=640, 
                       help='Output frame width (default: 640)')
    parser.add_argument('--height', '-H', type=int, default=360, 
                       help='Output frame height (default: 360)')
    parser.add_argument('--quality', '-q', type=int, default=85, 
                       help='JPEG quality 1-100 (default: 85)')
    parser.add_argument('--clean', '-c', action='store_true',
                       help='Clean output folder before extraction')
    
    args = parser.parse_args()
    
    print("🎭 Video to Frames Extractor for Dummy Lip-Sync Server")
    print("=" * 60)
    
    # Validate input video
    if not validate_video_file(args.video_path):
        return 1
    
    # Clean output folder if requested
    if args.clean and os.path.exists(args.output):
        print(f"🧹 Cleaning output folder: {args.output}")
        import shutil
        shutil.rmtree(args.output)
    
    # Extract frames
    success = extract_frames_from_video(
        video_path=args.video_path,
        output_folder=args.output,
        target_fps=args.fps,
        resize_to=(args.width, args.height),
        jpeg_quality=args.quality
    )
    
    if success:
        print(f"\n🎉 Ready to use with dummy_server.py!")
        print(f"   Run: python dummy_server.py")
        return 0
    else:
        print(f"\n❌ Frame extraction failed!")
        return 1

if __name__ == "__main__":
    exit(main())