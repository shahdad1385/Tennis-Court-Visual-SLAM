"""
Bridge between keypoint detection pipeline and OpenCV solvePnP.

Provides functions to extract 2D keypoints from detection results
and solve for the camera/robot pose using PnP.
"""
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from ..dataset.keypoints_schema import OBJ_POINTS_POLES, OBJ_POINTS_GRID


def get_pnp_points_2d(
    pipeline_result: Dict,
    source: str = "poles",
) -> Optional[np.ndarray]:
    """
    Extract 2D keypoints as (N, 2) array ready for cv2.solvePnP.

    Args:
        pipeline_result: Output from KeypointsPipeline.process_frame()
        source: "poles" for net pole keypoints, "grid" for grid keypoints

    Returns:
        (N, 2) float64 array of pixel coordinates, or None if insufficient
    """
    if source == "poles":
        pts = pipeline_result.get("pole_points_2d")
        if pts is None or pts.shape != (4, 2):
            return None
        return pts.astype(np.float64)

    elif source == "grid":
        pts = pipeline_result.get("grid_points_2d")
        if pts is None or pts.shape[0] < 3:
            return None
        return pts.astype(np.float64)

    else:
        raise ValueError(f"Unknown source: {source}. Use 'poles' or 'grid'.")


def solve_pose(
    image_points_2d: np.ndarray,
    K: np.ndarray,
    obj_points: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    method: int = cv2.SOLVEPNP_ITERATIVE,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Solve PnP and return rotation and translation vectors.

    Args:
        image_points_2d: (N, 2) pixel coordinates from detector
        K: (3, 3) camera intrinsic matrix
        obj_points: (N, 3) 3D world coordinates. Defaults to OBJ_POINTS_POLES.
        dist_coeffs: Camera distortion coefficients. None = no distortion.
        method: PnP solving method (SOLVEPNP_ITERATIVE, SOLVEPNP_SQPNP, etc.)

    Returns:
        Tuple of (success, rvec, tvec)
        - rvec: (3, 1) rotation vector (axis-angle)
        - tvec: (3, 1) translation vector
    """
    if image_points_2d is None or image_points_2d.shape[0] < 3:
        return False, None, None

    if obj_points is None:
        if image_points_2d.shape[0] == 4:
            obj_points = OBJ_POINTS_POLES
        else:
            obj_points = OBJ_POINTS_GRID[:image_points_2d.shape[0]]

    # Ensure correct shapes for solvePnP
    obj_pts = obj_points.reshape(-1, 1, 3).astype(np.float64)
    img_pts = image_points_2d.reshape(-1, 1, 2).astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist_coeffs, flags=method
    )

    return success, rvec, tvec


def rvec_tvec_to_pose_message(
    rvec: np.ndarray,
    tvec: np.ndarray,
):
    """
    Convert OpenCV rvec/tvec to a quaternion + position for ROS messages.

    Args:
        rvec: (3, 1) rotation vector
        tvec: (3, 1) translation vector

    Returns:
        Dict with keys: position (x,y,z), quaternion (x,y,z,w)
    """
    from scipy.spatial.transform import Rotation

    # Convert rotation vector to rotation matrix
    rot_mat, _ = cv2.Rodrigues(rvec)

    # Invert to get World-from-Camera (robot pose in world frame)
    r_world_cam = rot_mat.T
    t_world_cam = -np.dot(r_world_cam, tvec)

    # Convert to quaternion
    r = Rotation.from_matrix(r_world_cam)
    quat = r.as_quat()  # [x, y, z, w]

    return {
        "position": {
            "x": float(t_world_cam[0]),
            "y": float(t_world_cam[1]),
            "z": float(t_world_cam[2]),
        },
        "quaternion": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3]),
        },
    }
