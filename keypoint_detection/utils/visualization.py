"""
Visualization utilities for keypoint detection results.
"""
import cv2
import numpy as np
from typing import Dict
from ..dataset.keypoints_schema import (
    KEYPOINT_NAMES, POLE_INDICES, GRID_INDICES,
    KP_COLORS, SKELETON, SKELETON_COLOR,
)


def draw_keypoints(frame: np.ndarray, result: Dict) -> None:
    """
    Draw detected keypoints and skeleton on the frame (in-place).

    Args:
        frame: BGR image to draw on
        result: Output from KeypointsPipeline.process_frame()
    """
    kpts = result.get("all_keypoints")
    confs = result.get("all_confs")
    if kpts is None:
        return

    # Draw skeleton connections
    for idx_a, idx_b in SKELETON:
        if confs[idx_a] >= 0.6 and confs[idx_b] >= 0.6:
            pt_a = (int(kpts[idx_a, 0]), int(kpts[idx_a, 1]))
            pt_b = (int(kpts[idx_b, 0]), int(kpts[idx_b, 1]))
            cv2.line(frame, pt_a, pt_b, SKELETON_COLOR, 2, cv2.LINE_AA)

    # Draw keypoints
    for i in range(len(kpts)):
        x, y = int(kpts[i, 0]), int(kpts[i, 1])
        conf = confs[i]
        color = KP_COLORS[i]

        if conf >= 0.6:
            # Fully visible: filled circle
            cv2.circle(frame, (x, y), 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
        elif conf >= 0.3:
            # Low confidence: hollow circle
            cv2.circle(frame, (x, y), 6, color, 2, cv2.LINE_AA)
        # Below 0.3: don't draw

        # Label with keypoint name (only for high-confidence)
        if conf >= 0.6:
            name = KEYPOINT_NAMES[i].replace("_", " ")
            cv2.putText(frame, name, (x + 10, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def draw_info(frame: np.ndarray, result: Dict, fps: float = 0.0) -> None:
    """
    Draw status information overlay on the frame.

    Args:
        frame: BGR image to draw on
        result: Output from KeypointsPipeline.process_frame()
        fps: Current FPS
    """
    h, w = frame.shape[:2]

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Detection status
    if result["valid"]:
        status = "PnP READY"
        color = (0, 255, 0)
    else:
        status = "NO DETECTION"
        color = (0, 0, 255)

    cv2.putText(frame, status, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Keypoint count
    confs = result.get("all_confs")
    if confs is not None:
        n_visible = sum(1 for c in confs if c >= 0.6)
        cv2.putText(frame, f"Keypoints: {n_visible}/8", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Pole points if valid
    if result["valid"]:
        pts = result["pole_points_2d"]
        cv2.putText(frame, f"Pole pts: {pts.shape[0]}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_pnp_result(frame: np.ndarray, pose: Dict) -> None:
    """
    Draw estimated pose information on the frame.

    Args:
        frame: BGR image to draw on
        pose: Dict from solvepnp_bridge.rvec_tvec_to_pose_message()
    """
    pos = pose["position"]
    quat = pose["quaternion"]

    info_lines = [
        f"Pos: ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}) m",
        f"Quat: ({quat['x']:.3f}, {quat['y']:.3f}, {quat['z']:.3f}, {quat['w']:.3f})",
    ]

    h, w = frame.shape[:2]
    y_offset = h - 60

    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
