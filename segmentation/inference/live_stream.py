import cv2
import time
import argparse
from .segmentation_pipeline import SegmentationPipeline
from ..models.seg_detector import SegmentationModel
from ..utils.grid_calculator import GridCalculator


def run_live_segmentation(model_path: str, camera_id: int = 0,
                          backend: str = "onnx", target_fps: int = 30):
    """
    Real-time segmentation inference loop.

    Args:
        model_path: Path to ONNX/TRT model.
        camera_id: Camera device index.
        backend: Inference backend ("onnx" or "tensorrt").
        target_fps: Target frame rate.
    """
    # Initialize components
    model = SegmentationModel(model_path, backend=backend)
    
    # Initialize GridCalculator with default camera params
    # In production, these should match the IPM module's calibration
    grid_calc = GridCalculator(ground_roi=(-2.0, 2.0, 0.0, 10.0))
    
    pipeline = SegmentationPipeline(model, grid_calc)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return

    frame_interval = 1.0 / target_fps
    print(f"Starting live segmentation ({backend})...")

    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        # Process
        result = pipeline.process_frame(frame)

        # Print grid distances
        if result["grid_result"] and result["grid_result"]["metric_lines"]:
            n_lines = len(result["grid_result"]["metric_lines"])
            print(f"\rDetected {n_lines} line segments in metric space", end="")

        # Display
        cv2.imshow("Segmentation - Combined", result["overlays"]["combined"])
        cv2.imshow("Segmentation - Lines", result["overlays"]["lines"])
        
        if result["grid_result"] and result["grid_result"]["warped_mask"] is not None:
            cv2.imshow("Bird's Eye View", result["grid_result"]["warped_mask"])

        # FPS limiting
        elapsed = time.perf_counter() - t_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", type=str, default="onnx", choices=["onnx", "tensorrt"])
    args = parser.parse_args()
    run_live_segmentation(args.model, args.camera, args.backend)
