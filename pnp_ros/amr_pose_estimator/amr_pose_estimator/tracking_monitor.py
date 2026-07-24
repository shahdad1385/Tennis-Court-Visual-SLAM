"""
Tracking Monitor Node
Monitors ORB-SLAM3 tracking status and publishes recovery events.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
import time


class TrackingMonitor(Node):
    def __init__(self):
        super().__init__('tracking_monitor')
        
        # Parameters
        self.declare_parameter('max_lost_frames', 30)
        self.declare_parameter('reinit_attempts', 3)
        self.declare_parameter('status_publish_rate', 1.0)  # Hz
        
        self.max_lost_frames = self.get_parameter('max_lost_frames').value
        self.reinit_attempts = self.get_parameter('reinit_attempts').value
        
        # State
        self.current_status = "NOT_INITIALIZED"
        self.lost_counter = 0
        self.reinit_counter = 0
        self.last_slam_time = None
        self.slam_available = False
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/vision/slam_status', 10)
        self.recovery_pub = self.create_publisher(String, '/vision/slam_recovery', 10)
        
        # Subscribers
        # ORB-SLAM3 status (if available)
        self.slam_status_sub = self.create_subscription(
            String,
            '/orb_slam3/status',
            self.slam_status_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        
        # ORB-SLAM3 camera pose (to detect tracking)
        self.slam_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/orb_slam3/camera_pose',
            self.slam_pose_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        
        # SLAM TF Broadcaster status
        self.slam_status_sub2 = self.create_subscription(
            String,
            '/vision/slam_status',
            self.slam_tf_status_callback,
            10
        )
        
        # Timer for status monitoring
        rate = self.get_parameter('status_publish_rate').value
        self.timer = self.create_timer(1.0 / rate, self.monitor_tracking)
        
        self.get_logger().info("Tracking Monitor initialized")
    
    def slam_status_callback(self, msg):
        """Callback for ORB-SLAM3 status topic."""
        self.current_status = msg.data
        self.slam_available = True
        self.last_slam_time = self.get_clock().now()
        
        if msg.data == "OK":
            self.lost_counter = 0
            self.reinit_counter = 0
        elif msg.data == "LOST":
            self.lost_counter += 1
            self.get_logger().warn(f"SLAM tracking LOST (count: {self.lost_counter}/{self.max_lost_frames})")
            
            if self.lost_counter >= self.max_lost_frames:
                self.publish_recovery("LOST")
        elif msg.data == "REINITIALIZING":
            self.reinit_counter += 1
            self.get_logger().info(f"SLAM reinitializing (attempt: {self.reinit_counter}/{self.reinit_attempts})")
            
            if self.reinit_counter >= self.reinit_attempts:
                self.publish_recovery("REINIT_FAILED")
    
    def slam_pose_callback(self, msg):
        """Callback for ORB-SLAM3 camera pose (detects tracking activity)."""
        self.last_slam_time = self.get_clock().now()
        self.slam_available = True
        
        if self.current_status == "LOST":
            self.current_status = "OK"
            self.lost_counter = 0
            self.get_logger().info("SLAM tracking recovered")
    
    def slam_tf_status_callback(self, msg):
        """Callback for SLAM TF Broadcaster status."""
        if msg.data == "OK" and self.current_status == "NOT_INITIALIZED":
            self.current_status = "OK"
            self.get_logger().info("SLAM tracking initialized via TF Broadcaster")
    
    def monitor_tracking(self):
        """Periodic monitoring of tracking status."""
        # Check if SLAM is responsive
        if self.slam_available and self.last_slam_time is not None:
            elapsed = (self.get_clock().now() - self.last_slam_time).nanoseconds / 1e9
            
            if elapsed > 5.0 and self.current_status == "OK":
                self.get_logger().warn(f"No SLAM updates for {elapsed:.1f}s")
                self.current_status = "REINITIALIZING"
                self.publish_status()
        
        # Publish current status
        self.publish_status()
    
    def publish_status(self):
        """Publish current tracking status."""
        msg = String()
        msg.data = self.current_status
        self.status_pub.publish(msg)
    
    def publish_recovery(self, event):
        """Publish recovery event."""
        msg = String()
        msg.data = event
        self.recovery_pub.publish(msg)
        self.get_logger().warn(f"Recovery event: {event}")


def main(args=None):
    rclpy.init(args=args)
    node = TrackingMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
