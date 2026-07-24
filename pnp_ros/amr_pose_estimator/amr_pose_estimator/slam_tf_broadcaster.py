"""
SLAM TF Broadcaster Node
Broadcasts map -> odom -> base_link transforms and handles initial pose alignment
when net poles are detected by the PnP node.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import numpy as np
from scipy.spatial.transform import Rotation


class SlamTfBroadcaster(Node):
    def __init__(self):
        super().__init__('slam_tf_broadcaster')
        
        # Parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('camera_height', 0.6)
        self.declare_parameter('min_pole_detections', 4)
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('max_lost_frames', 30)
        
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.camera_height = self.get_parameter('camera_height').value
        self.min_pole_detections = self.get_parameter('min_pole_detections').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.max_lost_frames = self.get_parameter('max_lost_frames').value
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # TF listener for getting current transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # State variables
        self.slam_pose = None
        self.pnp_pose = None
        self.last_pnp_pose = None
        self.lost_counter = 0
        self.is_aligned = False
        self.alignment_transform = None
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/vision/slam_status', 10)
        
        # Subscribers
        # SLAM camera pose (from ORB-SLAM3)
        self.slam_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/orb_slam3/camera_pose',
            self.slam_pose_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        
        # PnP robot pose (from net pole detection)
        self.pnp_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/vision/robot_pose',
            self.pnp_pose_callback,
            10
        )
        
        # Timer for broadcasting transforms
        self.timer = self.create_timer(0.033, self.broadcast_transforms)  # ~30 Hz
        
        self.get_logger().info("SLAM TF Broadcaster initialized")
        self.get_logger().info(f"TF tree: {self.map_frame} -> {self.odom_frame} -> {self.base_frame} -> {self.camera_frame}")
    
    def slam_pose_callback(self, msg):
        """Callback for ORB-SLAM3 camera pose."""
        self.slam_pose = msg
        self.lost_counter = 0
        self.publish_status("OK")
    
    def pnp_pose_callback(self, msg):
        """Callback for PnP robot pose from net pole detection."""
        self.pnp_pose = msg
        self.last_pnp_pose = msg
        
        # Check if we need to align the SLAM frame
        if not self.is_aligned:
            self.attempt_initial_alignment(msg)
        else:
            # Periodic correction using PnP pose
            self.apply_pnp_correction(msg)
    
    def attempt_initial_alignment(self, msg):
        """
        Align SLAM frame to pitch coordinate system using PnP pose.
        This establishes the map -> odom transform.
        """
        if self.slam_pose is None:
            self.get_logger().warn("SLAM not initialized yet, waiting for first SLAM pose...")
            return
        
        # Compute the alignment transform: map (pitch) -> odom (SLAM)
        # T_map_odom = T_map_base * T_odom_base^(-1)
        # Where T_map_base is from PnP, T_odom_base is from SLAM
        
        pnp_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        pnp_quat = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        
        slam_pos = np.array([
            self.slam_pose.pose.pose.position.x,
            self.slam_pose.pose.pose.position.y,
            self.slam_pose.pose.pose.position.z
        ])
        
        slam_quat = [
            self.slam_pose.pose.pose.orientation.x,
            self.slam_pose.pose.pose.orientation.y,
            self.slam_pose.pose.pose.orientation.z,
            self.slam_pose.pose.pose.orientation.w
        ]
        
        # Compute rotation difference
        r_pnp = Rotation.from_quat(pnp_quat)
        r_slam = Rotation.from_quat(slam_quat)
        
        # T_map_odom rotation
        r_map_odom = r_pnp * r_slam.inv()
        
        # T_map_odom translation
        # p_map = R_map_odom * p_odom + t_map_odom
        # t_map_odom = p_map - R_map_odom * p_odom
        t_map_odom = pnp_pos - r_map_odom.apply(slam_pos)
        
        # Store alignment transform
        self.alignment_transform = {
            'translation': t_map_odom,
            'rotation': r_map_odom
        }
        
        self.is_aligned = True
        self.get_logger().info(f"SLAM frame aligned to pitch coordinates")
        self.get_logger().info(f"  Translation offset: {t_map_odom}")
        self.get_logger().info(f"  Rotation offset Euler (deg): {r_map_odom.as_euler('xyz', degrees=True)}")
        
        self.publish_status("OK")
    
    def apply_pnp_correction(self, msg):
        """
        Apply periodic correction to reduce SLAM drift using PnP pose.
        """
        if self.alignment_transform is None or self.slam_pose is None:
            return
        
        # Compute current error between PnP and SLAM-transformed pose
        pnp_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        slam_pos = np.array([
            self.slam_pose.pose.pose.position.x,
            self.slam_pose.pose.pose.position.y,
            self.slam_pose.pose.pose.position.z
        ])
        
        # Transform SLAM position to map frame
        slam_in_map = self.alignment_transform['rotation'].apply(slam_pos) + \
                      self.alignment_transform['translation']
        
        # Compute error
        error = np.linalg.norm(pnp_pos - slam_in_map)
        
        # If error is significant, update alignment
        if error > 0.5:  # 0.5 meter threshold
            self.get_logger().warn(f"SLAM drift detected: {error:.2f}m, updating alignment")
            self.attempt_initial_alignment(msg)
    
    def broadcast_transforms(self):
        """Broadcast all transforms in the TF tree."""
        now = self.get_clock().now().to_msg()
        
        # Broadcast base_link -> camera_link (static)
        self.broadcast_static_transform(
            self.base_frame,
            self.camera_frame,
            [0.0, 0.0, self.camera_height],
            [0.0, 0.0, 0.0, 1.0],
            now
        )
        
        # Broadcast odom -> base_link (from SLAM or fallback)
        if self.slam_pose is not None and self.is_aligned:
            # Use SLAM pose with alignment correction
            self.broadcast_odom_to_base(now)
        elif self.last_pnp_pose is not None:
            # Fallback to PnP pose
            self.broadcast_pnp_as_odom(now)
        else:
            # No pose available
            self.lost_counter += 1
            if self.lost_counter > self.max_lost_frames:
                self.publish_status("LOST")
    
    def broadcast_odom_to_base(self, now):
        """Broadcast odom -> base_link using SLAM pose with alignment."""
        slam_pos = np.array([
            self.slam_pose.pose.pose.position.x,
            self.slam_pose.pose.pose.position.y,
            self.slam_pose.pose.pose.position.z
        ])
        
        slam_quat = [
            self.slam_pose.pose.pose.orientation.x,
            self.slam_pose.pose.pose.orientation.y,
            self.slam_pose.pose.pose.orientation.z,
            self.slam_pose.pose.pose.orientation.w
        ]
        
        r_slam = Rotation.from_quat(slam_quat)
        
        # Transform to map frame
        map_pos = self.alignment_transform['rotation'].apply(slam_pos) + \
                  self.alignment_transform['translation']
        
        r_map = self.alignment_transform['rotation'] * r_slam
        map_quat = r_map.as_quat()
        
        # Broadcast map -> odom (alignment transform)
        self.broadcast_static_transform(
            self.map_frame,
            self.odom_frame,
            self.alignment_transform['translation'].tolist(),
            self.alignment_transform['rotation'].as_quat().tolist(),
            now
        )
        
        # Broadcast odom -> base_link (SLAM pose in odom frame = identity after alignment)
        # Actually, we broadcast base_link in map frame directly
        self.broadcast_static_transform(
            self.odom_frame,
            self.base_frame,
            slam_pos.tolist(),
            slam_quat,
            now
        )
    
    def broadcast_pnp_as_odom(self, now):
        """Fallback: use PnP pose directly as map -> base_link."""
        pos = [
            self.last_pnp_pose.pose.pose.position.x,
            self.last_pnp_pose.pose.pose.position.y,
            self.last_pnp_pose.pose.pose.position.z
        ]
        
        quat = [
            self.last_pnp_pose.pose.pose.orientation.x,
            self.last_pnp_pose.pose.pose.orientation.y,
            self.last_pnp_pose.pose.pose.orientation.z,
            self.last_pnp_pose.pose.pose.orientation.w
        ]
        
        # Broadcast map -> base_link (identity odom)
        self.broadcast_static_transform(
            self.map_frame,
            self.odom_frame,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            now
        )
        
        self.broadcast_static_transform(
            self.odom_frame,
            self.base_frame,
            pos,
            quat,
            now
        )
        
        self.publish_status("REINITIALIZING")
    
    def broadcast_static_transform(self, parent_frame, child_frame, translation, rotation, stamp):
        """Broadcast a static transform."""
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        
        t.transform.rotation.x = float(rotation[0])
        t.transform.rotation.y = float(rotation[1])
        t.transform.rotation.z = float(rotation[2])
        t.transform.rotation.w = float(rotation[3])
        
        self.tf_broadcaster.sendTransform(t)
    
    def publish_status(self, status):
        """Publish SLAM tracking status."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SlamTfBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
