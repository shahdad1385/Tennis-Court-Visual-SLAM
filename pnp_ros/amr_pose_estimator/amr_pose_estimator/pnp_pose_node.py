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
        
        # Subscription for 2D landmark keypoints (expects a list of 8 floats: x1,y1,x2,y2,x3,y3,x4,y4)
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped, # Placeholder, ideally a custom message or Float32MultiArray
            '/vision/landmarks_2d',
            self.landmarks_callback,
            10)
            
        # Camera Intrinsics (Example values - MUST be calibrated for your specific camera)
        # Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        self.declare_parameter('camera_matrix', [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0])
        cam_mat_flat = self.get_parameter('camera_matrix').value
        self.K = np.array(cam_mat_flat, dtype=np.float64).reshape((3, 3))
        
        # 3D Object Points in World Frame (Pitch Coordinates)
        # Origin (0,0,0) is defined at the center of the net on the ground.
        # X: Lateral (Right is positive), Y: Forward (Away from camera is positive), Z: Up
        # Assuming a standard net width of ~10.97m and height of ~1.07m
        self.obj_points = np.array([
            [-5.485, 0.0, 0.0],   # 1. Left Pole Base
            [-5.485, 0.0, 1.07],  # 2. Left Pole Top
            [ 5.485, 0.0, 0.0],   # 3. Right Pole Base
            [ 5.485, 0.0, 1.07]   # 4. Right Pole Top
        ], dtype=np.float64)
        
        self.get_logger().info("PnP Pose Node Initialized.")

    def landmarks_callback(self, msg):
        # NOTE: In a real implementation, parse the 2D points from the message
        # For this example, we assume the message contains the 4 (x,y) pairs.
        # Here we simulate 2D points for demonstration:
        image_points = np.array([
            [120, 400], # Left Pole Base
            [120, 180], # Left Pole Top
            [520, 400], # Right Pole Base
            [520, 180]  # Right Pole Top
        ], dtype=np.float64)
        
        # Solve PnP using the Iterative method
        # rvec: Rotation vector (axis-angle representation)
        # tvec: Translation vector (position of the object origin in the camera frame)
        success, rvec, tvec = cv2.solvePnP(
            self.obj_points, 
            image_points, 
            self.K, 
            None, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
            
        if success:
            self.get_logger().info(f"PnP Solved. Translation (Camera in World): {tvec.flatten()}")
            
            # Convert Rotation Vector to Rotation Matrix
            rot_mat, _ = cv2.Rodrigues(rvec)
            
            # Convert Rotation Matrix to Quaternion
            # scipy's Rotation class handles this robustly
            r = Rotation.from_matrix(rot_mat)
            quat = r.as_quat() # Returns [x, y, z, w]
            
            # Create Pose Message
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'world'
            
            # solvePnP returns the transform from Object (World) to Camera (Robot).
            # We want the Robot's pose in the World frame, so we invert the transform.
            # Camera_Pose_World = inv([R | t])
            
            # Simple inversion for [R | t]:
            # R_world_cam = R^T
            # t_world_cam = -R^T * t
            
            r_world_cam = rot_mat.T
            t_world_cam = -np.dot(r_world_cam, tvec)
            
            # Fill the message
            pose_msg.pose.pose.position.x = float(t_world_cam[0])
            pose_msg.pose.pose.position.y = float(t_world_cam[1])
            pose_msg.pose.pose.position.z = float(t_world_cam[2])
            
            # Quaternion from inverted rotation
            q_inv = Rotation.from_matrix(r_world_cam).as_quat()
            pose_msg.pose.pose.orientation.x = float(q_inv[0])
            pose_msg.pose.pose.orientation.y = float(q_inv[1])
            pose_msg.pose.pose.orientation.z = float(q_inv[2])
            pose_msg.pose.pose.orientation.w = float(q_inv[3])
            
            # Simple covariance (identity for now)
            pose_msg.pose.covariance = [0.01] * 36
            
            self.publisher_.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PnpPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
