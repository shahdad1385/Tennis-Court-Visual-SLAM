import numpy as np
from typing import Dict, Optional
from .pose_detector import TennisCourtDetector
from ..dataset.keypoints_schema import POLE_INDICES, GRID_INDICES


class KeypointSmoother:
    """Exponential moving average for temporal keypoint stability."""

    def __init__(self, alpha: float = 0.7, max_stale: int = 5):
        """
        Args:
            alpha: Smoothing factor (1.0 = no smoothing, 0.0 = frozen)
            max_stale: Max frames to hold last valid pose before reset
        """
        self.alpha = alpha
        self.max_stale = max_stale
        self.smoothed: Optional[np.ndarray] = None
        self.stale_count = 0

    def update(self, keypoints_2d: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Apply temporal smoothing to keypoint array.

        Args:
            keypoints_2d: (N, 2) array of pixel coordinates, or None if no detection

        Returns:
            Smoothed keypoints, or None if stale
        """
        if keypoints_2d is None:
            self.stale_count += 1
            if self.stale_count > self.max_stale:
                self.smoothed = None
            return self.smoothed

        self.stale_count = 0

        if self.smoothed is None or self.smoothed.shape != keypoints_2d.shape:
            self.smoothed = keypoints_2d.copy()
        else:
            self.smoothed = self.alpha * keypoints_2d + (1 - self.alpha) * self.smoothed

        return self.smoothed.copy()

    def reset(self):
        self.smoothed = None
        self.stale_count = 0


class KeypointsPipeline:
    """
    Full pipeline: camera frame -> filtered 2D keypoints -> PnP-ready arrays.

    Detects tennis court keypoints using YOLOv8-Pose, filters by confidence,
    and formats output for OpenCV solvePnP.
    """

    MIN_KP_CONF = 0.6  # Per-keypoint minimum confidence

    def __init__(self, detector: TennisCourtDetector, smooth: bool = True,
                 min_pole_kpts: int = 4):
        """
        Args:
            detector: Initialized TennisCourtDetector
            smooth: Enable temporal smoothing
            min_pole_kpts: Minimum pole keypoints required for valid detection
        """
        self.detector = detector
        self.min_pole_kpts = min_pole_kpts
        self.smoother = KeypointSmoother() if smooth else None

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single camera frame through the full pipeline.

        Returns:
            Dict with keys:
                - "pole_points_2d": (4, 2) float64 array or None — ready for solvePnP
                - "grid_points_2d": (N, 2) float64 array or None — visible grid points
                - "all_keypoints": (8, 3) float64 array or None — [x, y, vis]
                - "all_confs": (8,) float64 array or None — per-kpt confidence
                - "valid": bool — True if enough pole keypoints for PnP
                - "detection": raw detection dict or None
        """
        detections = self.detector.detect(frame)

        if not detections:
            return self._empty_result()

        # Select highest-confidence detection
        best = max(detections, key=lambda d: d["bbox_conf"])
        kpts = best["keypoints"]       # (8, 3)
        kpt_conf = best["kpt_confs"]   # (8,)

        # --- Filter pole keypoints ---
        pole_mask = [kpt_conf[i] >= self.MIN_KP_CONF for i in POLE_INDICES]
        visible_poles = [i for i, m in zip(POLE_INDICES, pole_mask) if m]
        pole_ok = len(visible_poles) >= self.min_pole_kpts

        pole_points = None
        if pole_ok:
            pole_points = kpts[POLE_INDICES, :2].astype(np.float64)

            # Enforce spatial ordering: left pole has smaller x than right pole
            if pole_points[0, 0] > pole_points[2, 0]:
                pole_points = pole_points[[2, 3, 0, 1]]

            # Enforce vertical ordering: pole top y < pole base y (image coords)
            if pole_points[0, 1] > pole_points[1, 1]:
                pole_points[[0, 1]] = pole_points[[1, 0]]
            if pole_points[2, 1] > pole_points[3, 1]:
                pole_points[[2, 3]] = pole_points[[3, 2]]

        # --- Filter grid keypoints ---
        grid_mask = [kpt_conf[i] >= self.MIN_KP_CONF for i in GRID_INDICES]
        visible_grid = [i for i, m in zip(GRID_INDICES, grid_mask) if m]

        grid_points = None
        if len(visible_grid) >= 2:
            grid_points = kpts[visible_grid, :2].astype(np.float64)

        # --- Temporal smoothing ---
        if self.smoother is not None and pole_points is not None:
            pole_points = self.smoother.update(pole_points)
        elif self.smoother is not None:
            self.smoother.update(None)

        return {
            "pole_points_2d": pole_points,
            "grid_points_2d": grid_points,
            "all_keypoints": kpts,
            "all_confs": kpt_conf,
            "valid": pole_ok,
            "detection": best,
        }

    def _empty_result(self) -> Dict:
        return {
            "pole_points_2d": None,
            "grid_points_2d": None,
            "all_keypoints": None,
            "all_confs": None,
            "valid": False,
            "detection": None,
        }
