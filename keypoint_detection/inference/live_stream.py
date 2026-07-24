"""
Real-time keypoint detection on a live camera stream.

Usage:
    python live_stream.py --model weights/best.pt --camera 0
"""
import argparse
import time
import cv2
import numpy as np
from .pose_detector import TennisCourtDetector
from .keypoints_pipeline import KeypointsPipeline
from ..utils.visualization import draw_keypoints, draw_info


def run_live_stream(
    model_path: str,
    camera_id: int = 0,
    conf_threshold: float = 0.6,
    target_fps: int = 30,
    device: str = "cpu",
    imgsz: int = 640,
    width: int = 640,
    height: int = 480,
):
    """
    Main entry point for live keypoint detection on camera stream.

    Args:
        model_path: Path to trained YOLOv8-Pose weights
        camera_id: Camera device index
        conf_threshold: Minimum detection confidence
        target_fps: Target frame rate
        device: Inference device ("cpu", "cuda", "mps")
        imgsz: Model input size
        width: Camera capture width
        height: Camera capture height
    """
    # Initialize detector and pipeline
    detector = TennisCourtDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
        imgsz=imgsz,
    )
    pipeline = KeypointsPipeline(detector, smooth=True)

    # Open camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return

    frame_interval = 1.0 / target_fps
    frame_count = 0
    fps_display = 0.0
    fps_timer = time.perf_counter()

    print(f"Starting live stream (camera {camera_id}, {width}x{height} @ {target_fps} FPS)")
    print("Press 'q' to quit")

    while True:
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        # Run pipeline
        result = pipeline.process_frame(frame)

        # Print PnP-ready coordinates
        if result["valid"]:
            pts = result["pole_points_2d"]
            print(f"\r[PnP Ready] ({pts[0, 0]:.0f},{pts[0, 1]:.0f}) "
                  f"({pts[1, 0]:.0f},{pts[1, 1]:.0f}) "
                  f"({pts[2, 0]:.0f},{pts[2, 1]:.0f}) "
                  f"({pts[3, 0]:.0f},{pts[3, 1]:.0f})", end="")
        else:
            print("\r[No valid keypoints detected]", end="")

        # Visualization
        vis_frame = frame.copy()
        draw_keypoints(vis_frame, result)
        draw_info(vis_frame, result, fps_display)
        cv2.imshow("Tennis Court Keypoints", vis_frame)

        # FPS calculation
        frame_count += 1
        elapsed_total = time.perf_counter() - fps_timer
        if elapsed_total >= 1.0:
            fps_display = frame_count / elapsed_total
            frame_count = 0
            fps_timer = time.perf_counter()

        # Frame rate limiting
        elapsed = time.perf_counter() - t_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("\nStream ended")
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live keypoint detection stream")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to YOLOv8-Pose .pt weights")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.6,
                        help="Confidence threshold (default: 0.6)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target FPS (default: 30)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Inference device (default: cpu)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Model input size (default: 640)")
    parser.add_argument("--width", type=int, default=640,
                        help="Camera capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Camera capture height (default: 480)")
    args = parser.parse_args()

    run_live_stream(
        model_path=args.model,
        camera_id=args.camera,
        conf_threshold=args.conf,
        target_fps=args.fps,
        device=args.device,
        imgsz=args.imgsz,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
