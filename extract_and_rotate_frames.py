#!/usr/bin/env python3
"""
Extract and rotate frames for debugging rotation-related bugs.
"""

import os
import json
import cv2
import imageio

# Working directory
work_dir = r"d:\myflie\ALL_CODE\swimming"
os.chdir(work_dir)

print("=" * 60)
print("Frame Extraction and Rotation Debug Script")
print("=" * 60)

results = {}

# ============================================================================
# Helper function to extract frame using imageio-ffmpeg
# ============================================================================
def extract_frame_with_imageio_ffmpeg(video_path, time_seconds):
    """Extract a frame at specific time using imageio-ffmpeg."""
    try:
        reader = imageio.get_reader(video_path, format='FFMPEG', mode='I')
        fps = reader.get_meta_data().get('fps', 30)
        frame_idx = int(time_seconds * fps)
        frame = reader.get_data(frame_idx)
        reader.close()
        return frame
    except Exception as e:
        print(f"    Error with imageio: {e}")
        return None

# ============================================================================
# 1. outputs_single directory
# ============================================================================
print("\n[1] Checking outputs_single directory...")
output_single_dir = os.path.join(work_dir, "outputs_single")
fused_single_path = os.path.join(output_single_dir, "fused_swim_v2.mp4")

if os.path.exists(fused_single_path):
    print(f"  Found: {fused_single_path}")
    frame_path = os.path.join(work_dir, "fusion_frame_0.png")

    # Extract frame 0 using imageio-ffmpeg
    frame = extract_frame_with_imageio_ffmpeg(fused_single_path, 0)

    if frame is not None:
        # Convert RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Save using cv2
        cv2.imwrite(frame_path, frame_bgr)

        results["fusion_frame_0"] = {
            "path": frame_path,
            "shape": frame.shape,
            "file_size": os.path.getsize(frame_path)
        }
        print(f"  Extracted frame 0 -> {frame_path}")
        print(f"  Shape: {frame.shape}")
        print(f"  File size: {os.path.getsize(frame_path)} bytes")
    else:
        print("  Failed to extract frame with imageio, trying cv2...")
        # Fallback to cv2
        cap = cv2.VideoCapture(fused_single_path)
        ret, frame_bgr = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(frame_path, frame_bgr)
            results["fusion_frame_0"] = {
                "path": frame_path,
                "shape": frame_bgr.shape,
                "file_size": os.path.getsize(frame_path)
            }
            print(f"  Extracted frame 0 (cv2) -> {frame_path}")
            print(f"  Shape: {frame_bgr.shape}")
else:
    print("  fused_swim_v2.mp4 NOT found")

# Check horizontal_debug in outputs_single
horizontal_debug_single = os.path.join(output_single_dir, "horizontal_debug")
if os.path.exists(horizontal_debug_single):
    files = os.listdir(horizontal_debug_single)
    print(f"  horizontal_debug: {len(files)} files")
    results["outputs_single_horizontal_debug"] = files[:10]  # First 10
else:
    print("  horizontal_debug NOT found in outputs_single")

# ============================================================================
# 2. outputs_v2 directory
# ============================================================================
print("\n[2] Checking outputs_v2 directory...")
output_v2_dir = os.path.join(work_dir, "outputs_v2")
fused_v2_path = os.path.join(output_v2_dir, "fused_swim_v2.mp4")

if os.path.exists(fused_v2_path):
    print(f"  Found: {fused_v2_path}")
    frame_path = os.path.join(work_dir, "outputs_v2_frame_0.png")

    # Extract frame 0 using imageio-ffmpeg
    frame = extract_frame_with_imageio_ffmpeg(fused_v2_path, 0)

    if frame is not None:
        # Convert RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Save using cv2
        cv2.imwrite(frame_path, frame_bgr)

        results["outputs_v2_frame_0"] = {
            "path": frame_path,
            "shape": frame.shape,
            "file_size": os.path.getsize(frame_path)
        }
        print(f"  Extracted frame 0 -> {frame_path}")
        print(f"  Shape: {frame.shape}")
        print(f"  File size: {os.path.getsize(frame_path)} bytes")
    else:
        print("  Failed to extract frame with imageio, trying cv2...")
        cap = cv2.VideoCapture(fused_v2_path)
        ret, frame_bgr = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(frame_path, frame_bgr)
            results["outputs_v2_frame_0"] = {
                "path": frame_path,
                "shape": frame_bgr.shape,
                "file_size": os.path.getsize(frame_path)
            }
            print(f"  Extracted frame 0 (cv2) -> {frame_path}")
            print(f"  Shape: {frame_bgr.shape}")
else:
    print("  fused_swim_v2.mp4 NOT found in outputs_v2")

# Check horizontal_debug in outputs_v2
horizontal_debug_v2 = os.path.join(output_v2_dir, "horizontal_debug")
if os.path.exists(horizontal_debug_v2):
    files = os.listdir(horizontal_debug_v2)
    print(f"  horizontal_debug: {len(files)} files")
    results["outputs_v2_horizontal_debug"] = files[:10]  # First 10
else:
    print("  horizontal_debug NOT found in outputs_v2")

# ============================================================================
# 3. video2_min frame extraction and rotation (bottom camera)
# ============================================================================
print("\n[3] Processing video2_min (bottom camera)...")
video2_path = r"d:\myflie\ALL_CODE\swimming\video_merge\video2_min\DJI_20260515181343_0009_D_min.mp4"

if os.path.exists(video2_path):
    print(f"  Found: {video2_path}")

    # Try imageio-ffmpeg first
    frame = extract_frame_with_imageio_ffmpeg(video2_path, 5)

    if frame is not None:
        # Convert RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        print("  Using cv2 fallback...")
        cap = cv2.VideoCapture(video2_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(5 * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret:
            print("  ERROR: Failed to read frame")
            frame_bgr = None

    if frame_bgr is not None:
        # Save original frame
        bot_original_path = os.path.join(work_dir, "bot_original_t5.png")
        cv2.imwrite(bot_original_path, frame_bgr)

        # Apply ROTATE_180
        frame_rotated = cv2.rotate(frame_bgr, cv2.ROTATE_180)

        # Save rotated frame
        bot_rotated_path = os.path.join(work_dir, "bot_rotated_t5.png")
        cv2.imwrite(bot_rotated_path, frame_rotated)

        results["bot_original_t5"] = {
            "path": bot_original_path,
            "shape": frame_bgr.shape,
            "file_size": os.path.getsize(bot_original_path)
        }
        results["bot_rotated_t5"] = {
            "path": bot_rotated_path,
            "shape": frame_rotated.shape,
            "file_size": os.path.getsize(bot_rotated_path)
        }

        print(f"  Original frame shape: {frame_bgr.shape}")
        print(f"  Rotated frame shape: {frame_rotated.shape}")
        print(f"  bot_original_t5.png: {os.path.getsize(bot_original_path)} bytes")
        print(f"  bot_rotated_t5.png: {os.path.getsize(bot_rotated_path)} bytes")
else:
    print("  Video file NOT found")

# ============================================================================
# 4. video1_min frame extraction and rotation (top camera)
# ============================================================================
print("\n[4] Processing video1_min (top camera)...")
video1_path = r"d:\myflie\ALL_CODE\swimming\video_merge\video1_min\DJI_20260515181348_0009_D_min.mp4"

if os.path.exists(video1_path):
    print(f"  Found: {video1_path}")

    # Try imageio-ffmpeg first
    frame = extract_frame_with_imageio_ffmpeg(video1_path, 5)

    if frame is not None:
        # Convert RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        print("  Using cv2 fallback...")
        cap = cv2.VideoCapture(video1_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(5 * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret:
            print("  ERROR: Failed to read frame")
            frame_bgr = None

    if frame_bgr is not None:
        # Save original frame
        top_original_path = os.path.join(work_dir, "top_original_t5.png")
        cv2.imwrite(top_original_path, frame_bgr)

        # Apply ROTATE_180
        frame_rotated = cv2.rotate(frame_bgr, cv2.ROTATE_180)

        # Save rotated frame
        top_rotated_path = os.path.join(work_dir, "top_rotated_t5.png")
        cv2.imwrite(top_rotated_path, frame_rotated)

        results["top_original_t5"] = {
            "path": top_original_path,
            "shape": frame_bgr.shape,
            "file_size": os.path.getsize(top_original_path)
        }
        results["top_rotated_t5"] = {
            "path": top_rotated_path,
            "shape": frame_rotated.shape,
            "file_size": os.path.getsize(top_rotated_path)
        }

        print(f"  Original frame shape: {frame_bgr.shape}")
        print(f"  Rotated frame shape: {frame_rotated.shape}")
        print(f"  top_original_t5.png: {os.path.getsize(top_original_path)} bytes")
        print(f"  top_rotated_t5.png: {os.path.getsize(top_rotated_path)} bytes")
else:
    print("  Video file NOT found")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY - All saved files:")
print("=" * 60)

for key, data in results.items():
    print(f"\n{key}:")
    print(f"  Path: {data['path']}")
    print(f"  Shape: {data['shape']}")
    print(f"  File size: {data['file_size']} bytes")

# Save results to JSON
with open(os.path.join(work_dir, "frame_extraction_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 60)
print("Done! Results saved to frame_extraction_results.json")
print("=" * 60)
