import numpy as np
from typing import List, Dict, Optional


class TennisCourtDetector:
    """YOLOv8-Pose wrapper for tennis court keypoint detection."""

    def __init__(self, model_path: str, conf_threshold: float = 0.6,
                 device: str = "cpu", imgsz: int = 640):
        """
        Args:
            model_path: Path to .pt weights (e.g., "runs/pose/train/weights/best.pt")
            conf_threshold: Minimum confidence to accept a detection (default 0.6)
            device: "cuda", "cpu", or "mps"
            imgsz: Input image size for inference
        """
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run inference on a single frame.

        Returns:
            List of detection dicts, each containing:
                - "bbox": [x1, y1, x2, y2] pixel coords
                - "bbox_conf": float, overall detection confidence
                - "keypoints": np.ndarray of shape (8, 3) — [x_px, y_px, visibility]
                - "kpt_confs": np.ndarray of shape (8,) — per-keypoint confidence
            Empty list if no detection above threshold.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.keypoints is None:
                continue

            # keypoints.xy: (num_dets, num_kpts, 2) — pixel coordinates
            # keypoints.conf: (num_dets, num_kpts) — per-kpt confidence
            # boxes.xyxy: (num_dets, 4) — bounding boxes
            # boxes.conf: (num_dets,) — bbox confidence

            kpts_xy = r.keypoints.xy[0].cpu().numpy()       # (8, 2)
            kpt_conf = r.keypoints.conf[0].cpu().numpy()    # (8,)
            bbox = r.boxes.xyxy[0].cpu().numpy()            # (4,)
            bbox_conf = float(r.boxes.conf[0])

            # Stack keypoints with visibility (from original data if available)
            kpt_vis = r.keypoints.vis[0].cpu().numpy() if r.keypoints.vis is not None else np.ones(8)
            keypoints = np.column_stack([kpts_xy, kpt_vis])  # (8, 3)

            detections.append({
                "bbox": bbox.tolist(),
                "bbox_conf": bbox_conf,
                "keypoints": keypoints,
                "kpt_confs": kpt_conf,
            })

        return detections
