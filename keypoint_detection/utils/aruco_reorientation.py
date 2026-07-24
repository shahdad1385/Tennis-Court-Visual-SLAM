"""
ArUco Marker Detection for AMR Pitch Reorientation.

Detects ArUco markers (DICT_6X6_250) placed around the pitch perimeter
to help the robot reorient toward the main pitch area when main visual
features (poles/grid) are not visible.
"""
import cv2
import numpy as np
import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ArUcoDetection:
    """Data class for a single ArUco marker detection."""
    marker_id: int
    corners: np.ndarray       # 4x2 pixel coordinates
    rvec: Optional[np.ndarray] = None  # (3,1) rotation vector
    tvec: Optional[np.ndarray] = None  # (3,1) translation vector
    yaw_angle: float = 0.0    # Relative yaw to pitch center (radians)
    distance: float = 0.0     # Distance to marker (meters)


# =========================================
# Configuration Constants
# =========================================

# ArUco dictionary type
ARUCO_DICT = cv2.aruco.DICT_6X6_250

# Physical marker length in meters
MARKER_LENGTH = 0.2  # 20cm markers

# Tag ID to pitch-relative yaw lookup table (radians)
# Maps marker ID to the yaw angle from that marker to the pitch center
# Markers are placed around the pitch perimeter facing outward
TAG_YAW_LOOKUP = {
    0: 0.0,           # Marker at pitch center reference (0 degrees)
    1: math.pi / 4,   # 45 degrees
    2: math.pi / 2,   # 90 degrees
    3: 3 * math.pi / 4,  # 135 degrees
    4: math.pi,       # 180 degrees (behind)
    5: -3 * math.pi / 4,  # -135 degrees
    6: -math.pi / 2,  # -90 degrees
    7: -math.pi / 4,  # -45 degrees
}

# Rotation speed for reorientation (rad/s)
REORIENTATION_SPEED = 0.3  # ~17 deg/s

# Detection confidence threshold
MIN_MARKER_DISTANCE = 0.1   # Minimum distance to consider valid (meters)
MAX_MARKER_DISTANCE = 5.0   # Maximum distance to consider valid (meters)


class ArUcoDetector:
    """
    ArUco marker detector for AMR pitch reorientation.
    
    Detects markers, estimates pose, and computes relative yaw
    to guide the robot back toward the main pitch area.
    """
    
    def __init__(self, 
                 camera_matrix: np.ndarray,
                 dist_coeffs: Optional[np.ndarray] = None,
                 marker_length: float = MARKER_LENGTH,
                 tag_lookup: Optional[Dict[int, float]] = None):
        """
        Args:
            camera_matrix: 3x3 camera intrinsic matrix K
            dist_coeffs: Camera distortion coefficients (4 or 5 elements)
            marker_length: Physical marker side length in meters
            tag_lookup: Dict mapping marker_id -> yaw angle to pitch center
        """
        self.K = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.marker_length = marker_length
        self.tag_lookup = tag_lookup if tag_lookup is not None else TAG_YAW_LOOKUP
        
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        self.get_logger = print  # Fallback logger
    
    def detect(self, frame: np.ndarray) -> List[ArUcoDetection]:
        """
        Detect ArUco markers and estimate their poses.
        
        Args:
            frame: BGR camera image
            
        Returns:
            List of ArUcoDetection objects with pose information
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        detections = []
        
        if ids is None or len(ids) == 0:
            return detections
        
        for i, marker_id in enumerate(ids.flatten()):
            # Get marker corners (4x2 pixel coordinates)
            marker_corners = corners[i][0]
            
            # Estimate pose using solvePnP
            obj_points = self._get_marker_object_points()
            img_points = marker_corners.reshape(-1, 1, 2).astype(np.float64)
            
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, self.K, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                continue
            
            # Compute distance to marker
            distance = np.linalg.norm(tvec)
            
            # Validate distance
            if distance < MIN_MARKER_DISTANCE or distance > MAX_MARKER_DISTANCE:
                continue
            
            # Compute relative yaw to pitch center using lookup table
            yaw_to_pitch = self.tag_lookup.get(marker_id, 0.0)
            
            # Compute robot's yaw from marker's perspective
            # tvec gives marker position in camera frame
            marker_yaw_cam = math.atan2(tvec[0][0], tvec[2][0])
            
            # Total yaw correction needed
            # Robot needs to rotate by: marker_yaw_cam + yaw_to_pitch
            total_yaw = marker_yaw_cam + yaw_to_pitch
            
            # Normalize to [-pi, pi]
            total_yaw = self._normalize_angle(total_yaw)
            
            detection = ArUcoDetection(
                marker_id=int(marker_id),
                corners=marker_corners,
                rvec=rvec,
                tvec=tvec,
                yaw_angle=total_yaw,
                distance=float(distance)
            )
            
            detections.append(detection)
        
        return detections
    
    def compute_reorientation_velocity(self, 
                                        detections: List[ArUcoDetection],
                                        target_speed: float = REORIENTATION_SPEED
                                       ) -> Tuple[float, float]:
        """
        Compute cmd_vel angular velocity to reorient toward pitch.
        
        Uses the closest/highest-confidence marker detection to
        determine the rotation direction and speed.
        
        Args:
            detections: List of ArUcoDetection objects
            target_speed: Maximum rotation speed (rad/s)
            
        Returns:
            Tuple of (linear_x, angular_z) velocities
        """
        if not detections:
            # No markers detected - rotate to search
            return (0.0, target_speed)
        
        # Find best detection (closest marker)
        best = min(detections, key=lambda d: d.distance)
        
        # Compute angular velocity proportional to yaw error
        yaw_error = best.yaw_angle
        
        # Proportional control with deadzone
        deadzone = math.radians(5.0)  # 5 degree deadzone
        
        if abs(yaw_error) < deadzone:
            # Aligned with pitch
            return (0.0, 0.0)
        
        # Proportional gain
        kp = 0.5
        
        # Compute angular velocity
        angular_z = kp * yaw_error
        
        # Clamp to target speed
        angular_z = max(-target_speed, min(target_speed, angular_z))
        
        return (0.0, angular_z)
    
    def draw_detections(self, frame: np.ndarray, 
                         detections: List[ArUcoDetection]) -> np.ndarray:
        """
        Draw detected markers and pose information on the frame.
        
        Args:
            frame: Input BGR image
            detections: List of ArUcoDetection objects
            
        Returns:
            Annotated frame
        """
        vis = frame.copy()
        
        for det in detections:
            # Draw marker corners
            pts = det.corners.astype(int).reshape(-1, 2)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            
            # Draw center point
            center = pts.mean(axis=0).astype(int)
            cv2.circle(vis, tuple(center), 5, (0, 0, 255), -1)
            
            # Draw ID
            cv2.putText(vis, f"ID:{det.marker_id}", 
                       (center[0] + 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Draw distance
            cv2.putText(vis, f"{det.distance:.2f}m",
                       (center[0] + 10, center[1] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw yaw angle
            yaw_deg = math.degrees(det.yaw_angle)
            cv2.putText(vis, f"yaw:{yaw_deg:.1f}deg",
                       (center[0] + 10, center[1] + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Draw axis visualization
            if det.rvec is not None and det.tvec is not None:
                cv2.drawFrameAxes(vis, self.K, self.dist_coeffs,
                                 det.rvec, det.tvec, self.marker_length * 0.5)
        
        return vis
    
    def _get_marker_object_points(self) -> np.ndarray:
        """
        Generate 3D object points for a single ArUco marker.
        
        Returns:
            (4, 1, 3) array of corner coordinates in marker frame
            Origin at marker center, marker lies in XY plane
        """
        half = self.marker_length / 2.0
        obj_points = np.array([
            [-half,  half, 0.0],   # Top-left
            [ half,  half, 0.0],   # Top-right
            [ half, -half, 0.0],   # Bottom-right
            [-half, -half, 0.0],   # Bottom-left
        ], dtype=np.float64)
        
        return obj_points.reshape(4, 1, 3)
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


class ArUcoReorientationNode:
    """
    ROS 2 compatible node for ArUco-based reorientation.
    
    Can be used standalone or integrated with the SMACH state machine.
    """
    
    def __init__(self, 
                 camera_matrix: np.ndarray,
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.detector = ArUcoDetector(camera_matrix, dist_coeffs)
        self.is_reorienting = False
        self.last_detection_time = None
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a camera frame for ArUco detection.
        
        Args:
            frame: BGR camera image
            
        Returns:
            Dict with keys:
                - "detections": List of ArUcoDetection
                - "cmd_vel": (linear_x, angular_z) tuple
                - "reorienting": bool indicating if reorientation is active
                - "visualization": Annotated frame
        """
        # Detect markers
        detections = self.detector.detect(frame)
        
        # Compute velocity command
        linear_x, angular_z = self.detector.compute_reorientation_velocity(detections)
        
        # Determine if we're actively reorienting
        self.is_reorienting = len(detections) > 0 and abs(angular_z) > 0.01
        
        if detections:
            self.last_detection_time = time.time()
        
        # Draw visualization
        vis = self.detector.draw_detections(frame, detections)
        
        return {
            "detections": detections,
            "cmd_vel": (linear_x, angular_z),
            "reorienting": self.is_reorienting,
            "visualization": vis,
            "num_markers": len(detections),
        }
    
    def should_transition_to_locating(self, 
                                       vision_lost_duration: float = 2.0,
                                       reorientation_timeout: float = 10.0
                                      ) -> bool:
        """
        Check if we should transition to LOCATING_PITCH state.
        
        Args:
            vision_lost_duration: How long vision has been lost
            reorientation_timeout: Max time to attempt reorientation
            
        Returns:
            True if should transition to LOCATING_PITCH
        """
        if self.last_detection_time is None:
            return vision_lost_duration > reorientation_timeout
        
        time_since_detection = time.time() - self.last_detection_time
        return time_since_detection > reorientation_timeout


def create_default_detector() -> ArUcoDetector:
    """
    Create an ArUcoDetector with default camera parameters.
    
    Returns:
        Configured ArUcoDetector instance
    """
    # Default camera matrix (640x480, should be calibrated)
    K = np.array([
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    return ArUcoDetector(K)


def run_standalone(camera_id: int = 0):
    """
    Run ArUco detection standalone for testing.
    
    Args:
        camera_id: Camera device index
    """
    detector = create_default_detector()
    
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return
    
    print("ArUco Reorientation Test")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and compute velocity
        result = detector.detect(frame)
        linear_x, angular_z = detector.compute_reorientation_velocity(result)
        
        # Draw detections
        vis = detector.draw_detections(frame, result)
        
        # Add velocity info overlay
        cv2.putText(vis, f"vel: ({linear_x:.2f}, {angular_z:.2f})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"markers: {len(result)}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("ArUco Reorientation", vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_standalone()
