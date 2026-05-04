"""Video-based fall detection integration test.

Processes video files through the fall detection pipeline and reports results.
Supports both Transformer and Geometric methods.

Usage:
    # Test with Transformer model
    python tests/test_fall_video.py data/test_videos/fall_clip.mp4

    # Test with Geometric fallback
    python tests/test_fall_video.py data/test_videos/fall_clip.mp4 --method geometric

    # Test all videos in a directory
    python tests/test_fall_video.py data/test_videos/ --all

    # Adjust confidence threshold
    python tests/test_fall_video.py fall.mp4 --threshold 0.85

    # Save annotated output video
    python tests/test_fall_video.py fall.mp4 --output result.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Config


def load_detector(method: str, config: Config, threshold: float):
    """Load the appropriate fall detector based on method."""
    if method == "transformer":
        from src.recognizers.transformer_fall_detector import TransformerFallDetector
        config.fall_detection.transformer_confidence = threshold
        detector = TransformerFallDetector(config)
        if detector._interpreter is None:
            print("[WARN] Transformer model not found. Falling back to geometric.")
            method = "geometric"
        else:
            return detector, method

    if method == "geometric":
        from src.recognizers.fall_detector import FallDetector
        detector = FallDetector(config)
        return detector, method

    raise ValueError(f"Unknown method: {method}")


def process_video(
    video_path: str,
    method: str = "transformer",
    threshold: float = 0.90,
    output_path: str = None,
    frame_skip: int = 1,
    show_preview: bool = False,
):
    """Process a video file through fall detection.

    Returns:
        dict with detection results and timing metrics.
    """
    config = Config()
    detector, actual_method = load_detector(method, config, threshold)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    print(f"\n{'='*60}")
    print(f"  Video: {Path(video_path).name}")
    print(f"  Resolution: {width}x{height} @ {fps:.1f}fps")
    print(f"  Duration: {duration:.1f}s ({total_frames} frames)")
    print(f"  Method: {actual_method}")
    print(f"  Threshold: {threshold}")
    print(f"  Frame skip: {frame_skip}")
    print(f"{'='*60}\n")

    # Output video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Processing
    fall_events = []
    frame_times = []
    frame_idx = 0
    last_fall_time = 0.0
    cooldown = config.fall_detection.alert_cooldown

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Frame skipping
        if frame_idx % frame_skip != 0:
            if writer:
                writer.write(frame)
            continue

        # Convert BGR to RGB for detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t_start = time.perf_counter()
        result = detector.detect(rgb_frame)
        t_elapsed = time.perf_counter() - t_start
        frame_times.append(t_elapsed)

        # Extract result fields
        if actual_method == "transformer":
            is_fall = result.get("fall_detected", False)
            confidence = result.get("confidence", 0.0)
            state = result.get("state", "unknown")
            fall_prob = result.get("metrics", {}).get("fall_probability", 0.0)
        else:
            is_fall = result.get("fall_detected", False)
            confidence = result.get("confidence", 0.0)
            state = result.get("state", "unknown")
            fall_prob = confidence

        # Draw overlay
        video_time = frame_idx / fps
        status_color = (0, 255, 0)  # Green = safe

        if is_fall:
            status_color = (0, 0, 255)  # Red
            fall_events.append({
                "frame": frame_idx,
                "time": video_time,
                "confidence": confidence,
                "probability": fall_prob,
            })
            print(
                f"  🚨 FALL at {video_time:.1f}s "
                f"(frame {frame_idx}) "
                f"conf={confidence:.2%}"
            )
        elif state == "fallen":
            status_color = (0, 165, 255)  # Orange (within cooldown)

        # Annotate frame
        cv2.putText(
            frame,
            f"[{actual_method.upper()}] {state} | prob={fall_prob:.2%} | {t_elapsed*1000:.0f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2,
        )
        cv2.putText(
            frame,
            f"Frame {frame_idx}/{total_frames} | {video_time:.1f}s",
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        if writer:
            writer.write(frame)

        if show_preview:
            cv2.imshow("Fall Detection Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n  [Preview stopped by user]")
                break

    cap.release()
    if writer:
        writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    # Results
    avg_time = np.mean(frame_times) if frame_times else 0
    max_time = np.max(frame_times) if frame_times else 0
    processed_fps = 1.0 / avg_time if avg_time > 0 else 0

    results = {
        "video": str(video_path),
        "method": actual_method,
        "threshold": threshold,
        "total_frames": total_frames,
        "processed_frames": len(frame_times),
        "duration_seconds": duration,
        "fall_events": fall_events,
        "fall_count": len(fall_events),
        "avg_inference_ms": avg_time * 1000,
        "max_inference_ms": max_time * 1000,
        "processed_fps": processed_fps,
    }

    print(f"\n{'─'*60}")
    print(f"  Results: {Path(video_path).name}")
    print(f"{'─'*60}")
    print(f"  Falls detected:    {len(fall_events)}")
    print(f"  Frames processed:  {len(frame_times)}/{total_frames}")
    print(f"  Avg inference:     {avg_time*1000:.1f} ms")
    print(f"  Max inference:     {max_time*1000:.1f} ms")
    print(f"  Effective FPS:     {processed_fps:.1f}")

    if fall_events:
        print(f"\n  Fall Timeline:")
        for evt in fall_events:
            print(f"    → {evt['time']:.1f}s (frame {evt['frame']}, conf={evt['confidence']:.2%})")

    if output_path:
        print(f"\n  Output saved: {output_path}")

    print()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test fall detection on video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Video file path or directory (with --all)",
    )
    parser.add_argument(
        "--method", choices=["transformer", "geometric"], default="transformer",
        help="Detection method (default: transformer)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="Fall confidence threshold (default: 0.90)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save annotated output video",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=1,
        help="Process every Nth frame (default: 1 = all frames)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all video files in directory",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show live preview window (press Q to stop)",
    )
    parser.add_argument(
        "--extensions", nargs="+", default=[".mp4", ".avi", ".mov", ".mkv"],
        help="Video file extensions to scan (default: .mp4 .avi .mov .mkv)",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"[ERROR] Path not found: {input_path}")
        sys.exit(1)

    if input_path.is_dir():
        if not args.all:
            print(f"[ERROR] {input_path} is a directory. Use --all to process all videos.")
            sys.exit(1)

        videos = []
        for ext in args.extensions:
            videos.extend(input_path.glob(f"*{ext}"))
        videos.sort()

        if not videos:
            print(f"[ERROR] No video files found in {input_path}")
            sys.exit(1)

        print(f"\nFound {len(videos)} video(s) in {input_path}\n")

        all_results = []
        for vpath in videos:
            result = process_video(
                str(vpath),
                method=args.method,
                threshold=args.threshold,
                frame_skip=args.frame_skip,
                show_preview=args.preview,
            )
            if result:
                all_results.append(result)

        # Summary
        total_falls = sum(r["fall_count"] for r in all_results)
        avg_fps = np.mean([r["processed_fps"] for r in all_results])
        print(f"\n{'='*60}")
        print(f"  SUMMARY: {len(all_results)} videos processed")
        print(f"  Total falls detected: {total_falls}")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"{'='*60}\n")

    else:
        process_video(
            str(input_path),
            method=args.method,
            threshold=args.threshold,
            output=args.output,
            frame_skip=args.frame_skip,
            show_preview=args.preview,
        )


if __name__ == "__main__":
    main()
