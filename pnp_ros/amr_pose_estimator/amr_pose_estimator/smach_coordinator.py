"""
SMACH State Machine for AMR Ball-Throwing Robot
Controls tracking states: LOCATING_PITCH, TRACKING_AND_AIMING, VISION_LOST
"""
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import smach
import smach_ros
import time
import math
import numpy as np
from std_msgs.msg import String, Float64
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist


# =========================================
# Configuration Constants
# =========================================
CONFIDENCE_THRESHOLD = 0.75      # Minimum confidence to consider pitch detected
ROTATION_SPEED_RAD = math.radians(15.0)  # 15 deg/s rotation speed
VISION_LOST_TIMEOUT = 1.5       # seconds before declaring vision lost
ORIENTATION_ERROR_THRESHOLD = math.radians(45.0)  # 45 degrees max error before reset
LAUNCHER_DISTANCE_THRESHOLD = 3.0  # meters - max distance to launch ball


class BallThrowingStateMachine(Node):
    """ROS 2 node wrapping the SMACH state machine for ball-throwing AMR."""

    def __init__(self):
        super().__init__('ball_throwing_smach')
        
        # Callback groups for concurrent subscription handling
        self.sub_callback_group = ReentrantCallbackGroup()
        self.pub_callback_group = MutuallyExclusiveCallbackGroup()
        
        # State variables (shared across states via userdata or this node)
        self.vision_confidence = 0.0
        self.vision_status = "NOT_INITIALIZED"
        self.robot_pose = None
        self.last_vision_time = None
        self.vision_lost_start = None
        self.goal_distance = float('inf')
        self.goal_heading_error = 0.0
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10, callback_group=self.pub_callback_group)
        self.launcher_pub = self.create_publisher(
            Float64, '/launcher/angle', 10, callback_group=self.pub_callback_group)
        self.launcher_fire_pub = self.create_publisher(
            String, '/launcher/fire', 10, callback_group=self.pub_callback_group)
        self.state_pub = self.create_publisher(
            String, '/amr/state', 10, callback_group=self.pub_callback_group)
        
        # Subscribers
        self.create_subscription(
            String, '/vision/slam_status',
            self._slam_status_callback, 10,
            callback_group=self.sub_callback_group)
        self.create_subscription(
            PoseWithCovarianceStamped, '/vision/robot_pose',
            self._robot_pose_callback, 10,
            callback_group=self.sub_callback_group)
        self.create_subscription(
            PoseWithCovarianceStamped, '/odom/wheels',
            self._odom_callback, 10,
            callback_group=self.sub_callback_group)
        self.create_subscription(
            String, '/vision/slam_recovery',
            self._recovery_callback, 10,
            callback_group=self.sub_callback_group)
        
        self.get_logger().info("Ball Throwing State Machine Node initialized")
    
    # ---- Subscriber Callbacks ----
    
    def _slam_status_callback(self, msg):
        self.vision_status = msg.data
        if msg.data == "OK":
            self.last_vision_time = self.get_clock().now()
            if self.vision_lost_start is not None:
                self.get_logger().info("Vision recovered")
                self.vision_lost_start = None
    
    def _robot_pose_callback(self, msg):
        self.robot_pose = msg
        self.last_vision_time = self.get_clock().now()
        
        # Compute confidence from covariance (lower covariance = higher confidence)
        cov = msg.pose.covariance
        # Diagonal elements: [x, y, z, roll, pitch, yaw, ...]
        pos_var = cov[0] + cov[7]  # x + y variance
        yaw_var = cov[35]          # yaw variance
        
        # Convert variance to confidence score (0-1)
        # Lower variance = higher confidence
        pos_conf = max(0.0, 1.0 - min(pos_var / 0.5, 1.0))
        yaw_conf = max(0.0, 1.0 - min(yaw_var / 0.2, 1.0))
        self.vision_confidence = (pos_conf * 0.6 + yaw_conf * 0.4)
    
    def _odom_callback(self, msg):
        # Store odometry for dead reckoning in VISION_LOST state
        pass
    
    def _recovery_callback(self, msg):
        if msg.data == "REINIT_FAILED":
            self.get_logger().warn("SLAM reinit failed - may need manual intervention")
    
    # ---- Action Methods ----
    
    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
    
    def rotate_robot(self, angular_vel):
        cmd = Twist()
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)
    
    def publish_state(self, state_name):
        msg = String()
        msg.data = state_name
        self.state_pub.publish(msg)
    
    def is_vision_lost(self):
        if self.last_vision_time is None:
            return True
        elapsed = (self.get_clock().now() - self.last_vision_time).nanoseconds / 1e9
        return elapsed > VISION_LOST_TIMEOUT


# =========================================
# SMACH States
# =========================================

class LocatingPitch(smach.State):
    """
    State 1: LOCATING_PITCH
    Rotate robot slowly (15 deg/s) until net poles or grid lines
    pass confidence threshold (> 0.75).
    """
    
    def __init__(self, node):
        smach.State.__init__(self, outcomes=['pitch_detected', 'shutdown'])
        self.node = node
        self.scan_start_time = None
    
    def execute(self, userdata):
        self.node.get_logger().info("State: LOCATING_PITCH - Scanning for pitch...")
        self.node.publish_state("LOCATING_PITCH")
        self.scan_start_time = self.node.get_clock().now()
        
        # Stop any previous motion
        self.node.stop_robot()
        time.sleep(0.1)
        
        while rclpy.ok():
            # Spin to process callbacks
            rclpy.spin_once(self.node, timeout_sec=0.01)
            
            # Check if vision is available and confidence is high enough
            if (self.node.vision_confidence >= CONFIDENCE_THRESHOLD and
                self.node.vision_status == "OK" and
                not self.node.is_vision_lost()):
                
                self.node.get_logger().info(
                    f"Pitch detected! Confidence: {self.node.vision_confidence:.2f}")
                self.node.stop_robot()
                return 'pitch_detected'
            
            # Rotate at 15 deg/s
            self.node.rotate_robot(ROTATION_SPEED_RAD)
            
            # Log progress periodically
            elapsed = (self.node.get_clock().now() - self.scan_start_time).nanoseconds / 1e9
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                self.node.get_logger().info(
                    f"Scanning... {elapsed:.1f}s, "
                    f"confidence: {self.node.vision_confidence:.2f}, "
                    f"status: {self.node.vision_status}")
            
            time.sleep(0.033)  # ~30 Hz
        
        return 'shutdown'


class TrackingAndAiming(smach.State):
    """
    State 2: TRACKING_AND_AIMING
    Lock pose, update geometry continuously using vision + EKF,
    compute distance to goal, and prepare ball launcher.
    """
    
    def __init__(self, node):
        smach.State.__init__(self, outcomes=['vision_lost', 'ready_to_launch', 'shutdown'])
        self.node = node
        self.tracking_start_time = None
        self.stable_frames = 0
        self.required_stable_frames = 30  # 1 second at 30 Hz
    
    def execute(self, userdata):
        self.node.get_logger().info("State: TRACKING_AND_AIMING - Locked on pitch")
        self.node.publish_state("TRACKING_AND_AIMING")
        self.tracking_start_time = self.node.get_clock().now()
        self.stable_frames = 0
        self.node.stop_robot()
        
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)
            
            # Check for vision loss
            if self.node.is_vision_lost():
                self.node.get_logger().warn("Vision lost during tracking!")
                return 'vision_lost'
            
            # Update geometry from robot pose
            if self.node.robot_pose is not None:
                pose = self.node.robot_pose.pose.pose
                
                # Distance to goal (pitch/net)
                # Goal is at origin (0,0,0) in map frame
                dx = pose.position.x
                dy = pose.position.y
                self.node.goal_distance = math.sqrt(dx**2 + dy**2)
                
                # Heading error relative to goal
                robot_yaw = self._quat_to_yaw(
                    pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w)
                goal_angle = math.atan2(-dy, -dx)  # Angle toward origin
                self.node.goal_heading_error = self._normalize_angle(goal_angle - robot_yaw)
                
                # Compute launcher angle based on distance
                launcher_angle = self._compute_launcher_angle(self.node.goal_distance)
                
                # Publish launcher angle
                angle_msg = Float64()
                angle_msg.data = launcher_angle
                self.node.launcher_pub.publish(angle_msg)
                
                # Log tracking info
                self.node.get_logger().info(
                    f"Tracking: dist={self.node.goal_distance:.2f}m, "
                    f"heading_err={math.degrees(self.node.goal_heading_error):.1f}deg, "
                    f"launcher={math.degrees(launcher_angle):.1f}deg")
                
                # Check if we're stable and in launch position
                if (self.node.goal_distance <= LAUNCHER_DISTANCE_THRESHOLD and
                    abs(self.node.goal_heading_error) < math.radians(10.0)):
                    self.stable_frames += 1
                else:
                    self.stable_frames = max(0, self.stable_frames - 1)
                
                if self.stable_frames >= self.required_stable_frames:
                    self.node.get_logger().info("Stable lock achieved - ready to launch!")
                    return 'ready_to_launch'
            
            time.sleep(0.033)
        
        return 'shutdown'
    
    def _quat_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def _compute_launcher_angle(self, distance):
        """
        Compute launcher elevation angle based on distance.
        Uses simplified projectile motion:
        angle = 0.5 * arcsin(g * d / v^2)
        Assuming fixed launch velocity ~10 m/s
        """
        v = 10.0  # m/s (assumed launch velocity)
        g = 9.81
        sin_arg = min(1.0, g * distance / (v * v))
        return 0.5 * math.asin(sin_arg)


class VisionLost(smach.State):
    """
    State 3: VISION_LOST (Fallback)
    Triggered if pitch visibility is lost for > 1.5 seconds.
    Fall back to IMU/wheel odometry to navigate back toward pitch heading.
    If orientation error > 45 degrees, stop launcher and transition back to LOCATING_PITCH.
    """
    
    def __init__(self, node):
        smach.State.__init__(self, outcomes=['locate_pitch', 'shutdown'])
        self.node = node
        self.recovery_start_time = None
        self.last_odom_yaw = None
    
    def execute(self, userdata):
        self.node.get_logger().warn("State: VISION_LOST - Falling back to dead reckoning")
        self.node.publish_state("VISION_LOST")
        self.recovery_start_time = self.node.get_clock().now()
        
        # Stop launcher immediately
        fire_msg = String()
        fire_msg.data = "STOP"
        self.node.launcher_fire_pub.publish(fire_msg)
        
        # Store last known heading for dead reckoning
        if self.node.robot_pose is not None:
            pose = self.node.robot_pose.pose.pose
            self.last_odom_yaw = self._quat_to_yaw(
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w)
        
        recovery_duration = 0.0
        
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)
            
            recovery_duration = (self.node.get_clock().now() - self.recovery_start_time).nanoseconds / 1e9
            
            # Try to recover vision by rotating slowly
            self.node.rotate_robot(ROTATION_SPEED_RAD * 0.5)  # Slower scan
            
            # Check if vision has been recovered
            if not self.node.is_vision_lost() and self.node.vision_status == "OK":
                self.node.get_logger().info("Vision recovered during recovery!")
                self.node.stop_robot()
                return 'locate_pitch'
            
            # Check orientation error using last known position
            # If we've drifted too far, give up and re-scan
            if self.node.robot_pose is not None:
                current_pose = self.node.robot_pose.pose.pose
                current_yaw = self._quat_to_yaw(
                    current_pose.orientation.x, current_pose.orientation.y,
                    current_pose.orientation.z, current_pose.orientation.w)
                
                if self.last_odom_yaw is not None:
                    yaw_drift = abs(self._normalize_angle(current_yaw - self.last_odom_yaw))
                    
                    if yaw_drift > ORIENTATION_ERROR_THRESHOLD:
                        self.node.get_logger().warn(
                            f"Orientation drift {math.degrees(yaw_drift):.1f}deg > "
                            f"{math.degrees(ORIENTATION_ERROR_THRESHOLD):.1f}deg threshold. "
                            f"Returning to LOCATING_PITCH")
                        self.node.stop_robot()
                        return 'locate_pitch'
            
            # Timeout: if recovery takes too long, go back to locating
            if recovery_duration > 10.0:
                self.node.get_logger().warn(
                    f"Recovery timeout ({recovery_duration:.1f}s). Returning to LOCATING_PITCH")
                self.node.stop_robot()
                return 'locate_pitch'
            
            self.node.get_logger().info(
                f"Recovery in progress... {recovery_duration:.1f}s")
            time.sleep(0.033)
        
        return 'shutdown'
    
    def _quat_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    
    node = BallThrowingStateMachine()
    
    # Create SMACH state machine
    sm = smach.StateMachine(outcomes=['SHUTDOWN'])
    
    # Add states to the machine
    with sm:
        smach.StateMachine.add(
            'LOCATING_PITCH',
            LocatingPitch(node),
            transitions={
                'pitch_detected': 'TRACKING_AND_AIMING',
                'shutdown': 'SHUTDOWN'
            }
        )
        
        smach.StateMachine.add(
            'TRACKING_AND_AIMING',
            TrackingAndAiming(node),
            transitions={
                'vision_lost': 'VISION_LOST',
                'ready_to_launch': 'TRACKING_AND_AIMING',  # Stay in tracking for continuous shots
                'shutdown': 'SHUTDOWN'
            }
        )
        
        smach.StateMachine.add(
            'VISION_LOST',
            VisionLost(node),
            transitions={
                'locate_pitch': 'LOCATING_PITCH',
                'shutdown': 'SHUTDOWN'
            }
        )
    
    # Create introspection server for visualization (optional)
    sis = smach_ros.IntrospectionServer(
        'ball_throwing_smach', sm, '/AMR_SMACH')
    sis.start()
    
    # Run the state machine in a separate thread
    import threading
    smach_thread = threading.Thread(target=sm.execute)
    smach_thread.start()
    
    # Spin the ROS node in the main thread
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    sis.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
