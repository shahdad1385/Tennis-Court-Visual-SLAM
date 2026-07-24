"""
ROS 2 Node for PnP Pose Estimation.

Subscribes to /vision/landmarks_2d (2D keypoints from YOLOv8-Pose)
and publishes /vision/robot_pose (6DoF pose in world frame).

Pipeline Integration:
  /camera/image_raw -> [yolo_pose_node] -> /vision/landmarks_2d -> [pnp_node] -> /vision/robot_pose
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class PnpPoseNode(Node):
    def __init__(self):
        super().__init__('pnp_pose_node')
        
        # Publisher for the estimated robot pose in the world frame
        self.publisher_ = self.create_publisher(PoseWithCovarianceStamped, '/vision/robot_pose', 10)
        
        # Subscription for 2D landmark keypoints from YOLOv8-Pose node
        # Message format: PoseWithCovarianceStamped with encoded 2D keypoints
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/vision/landmarks_2d',
            self.landmarks_callback,
            10)
            
        # Camera Intrinsics (Example values - MUST be calibrated for your specific camera)
        self.declare_parameter('camera_matrix', [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0])
        cam_mat_flat = self.get_parameter('camera_matrix').value
        self.K = np.array(cam_mat_flat, dtype=np.float64).reshape((3, 3))
        
        # 3D Object Points in World Frame (Pitch Coordinates)
        # Origin (0,0,0) is defined at the center of the net on the ground.
        # X: Lateral (Right is positive), Y: Forward (Away from camera is positive), Z: Up
        # Standard net width: ~10.97m, Height: ~1.07m
        self.obj_points = np.array([
            [-5.485, 0.0, 0.0],   # 1. Left Pole Base
            [-5.485, 0.0, 1.07],  # 2. Left Pole Top
            [ 5.485, 0.0, 0.0],   # 3. Right Pole Base
            [ 5.485, 0.0, 1.07]   # 4. Right Pole Top
        ], dtype=np.float64)
        
        # State tracking
        self.last_pose = None
        self.consecutive_failures = 0
        
        self.get_logger().info("PnP Pose Node Initialized")
        self.get_logger().info(f"  Object points: {self.obj_points.tolist()}")
        self.get_logger().info(f"  Camera matrix diagonal: {np.diag(self.K).tolist()}")

    def landmarks_callback(self, msg):
        """
        Callback for /vision/landmarks_2d topic.
        
        Parses 2D keypoints from the message and solves PnP.
        
        Message format (PoseWithCovarianceStamped):
          - position.x = Left Pole Base x
          - position.y = Left Pole Base y
          - position.z = Left Pole Top x
          - orientation.x = Left Pole Top y
          - orientation.y = Right Pole Base x
          - orientation.z = Right Pole Base y
          - orientation.w = Right Pole Top x
          - covariance[0] = Right Pole Top y
        """
        try:
            # Parse 2D keypoints from message
            image_points = self.parse_landmarks(msg)
            
            if image_points is None:
                self.consecutive_failures += 1
                if self.consecutive_failures > 10:
                    self.get_logger().warn(
                        f"No valid keypoints for {self.consecutive_failures} frames")
                return
            
            self.consecutive_failures = 0
            
            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points, 
                image_points, 
                self.K, 
                None, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
                
            if success:
                self.publish_pose(rvec, tvec)
            else:
                self.get_logger().debug("PnP solve failed")
            
        except Exception as e:
            self.get_logger().error(f"Landmarks callback error: {e}")
    
    def parse_landmarks(self, msg) -> np.ndarray:
        """
        Parse 2D keypoints from PoseWithCovarianceStamped message.
        
        Args:
            msg: PoseWithCovarianceStamped with encoded keypoints
            
        Returns:
            (4, 2) array of pixel coordinates, or None if invalid
        """
        # Decode keypoints from message
        # Format from yolo_pose_node:
        #   position = (x1, y1, x2)
        #   orientation = (y2, x3, y3, x4)
        #   covariance[0] = y4
        
        try:
            x1 = msg.pose.pose.position.x
            y1 = msg.pose.pose.position.y
            x2 = msg.pose.pose.position.z
            y2 = msg.pose.pose.orientation.x
            x3 = msg.pose.pose.orientation.y
            y3 = msg.pose.pose.orientation.z
            x4 = msg.pose.pose.orientation.w
            y4 = msg.pose.covariance[0]
            
            # Validate coordinates are within reasonable image bounds
            points = np.array([
                [x1, y1],  # Left Pole Base
                [x2, y2],  # Left Pole Top
                [x3, y3],  # Right Pole Base
                [x4, y4],  # Right Pole Top
            ], dtype=np.float64)
            
            # Basic validation
            if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                return None
            
            # Check if points are within image bounds (assuming 640x480)
            if np.any(points < 0) or np.any(points[:, 0] > 640) or np.any(points[:, 1] > 480):
                self.get_logger().debug(f"Keypoints out of bounds: {points}")
                return None
            
            return points
            
        except Exception as e:
            self.get_logger().error(f"Failed to parse landmarks: {e}")
            return None
    
    def publish_pose(self, rvec, tvec):
        """
        Convert PnP result to pose message and publish.
        
        Args:
            rvec: (3,1) rotation vector from solvePnP
            tvec: (3,1) translation vector from solvePnP
        """
        # Convert Rotation Vector to Rotation Matrix
        rot_mat, _ = cv2.Rodrigues(rvec)
        
        # solvePnP returns transform from World to Camera
        # We want Camera pose in World frame, so invert
        r_world_cam = rot_mat.T
        t_world_cam = -np.dot(r_world_cam, tvec)
        
        # Convert to quaternion
        r = Rotation.from_matrix(r_world_cam)
        quat = r.as_quat()  # [x, y, z, w]
        
        # Create Pose Message
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        
        # Fill position
        pose_msg.pose.pose.position.x = float(t_world_cam[0])
        pose_msg.pose.pose.position.y = float(t_world_cam[1])
        pose_msg.pose.pose.position.z = float(t_world_cam[2])
        
        # Fill orientation
        pose_msg.pose.pose.orientation.x = float(quat[0])
        pose_msg.pose.pose.orientation.y = float(quat[1])
        pose_msg.pose.pose.orientation.z = float(quat[2])
        pose_msg.pose.pose.orientation.w = float(quat[3])
        
        # Covariance (tunable based on detection quality)
        pose_msg.pose.covariance = [0.01] * 36
        
        self.last_pose = pose_msg
        self.publisher_.publish(pose_msg)
        
        self.get_logger().info(
            f"PnP Solved: pos=({t_world_cam[0]:.2f}, {t_world_cam[1]:.2f}, {t_world_cam[2]:.2f})")


def main(args=None):
    rclpy.init(args=args)
    node = PnpPoseNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
