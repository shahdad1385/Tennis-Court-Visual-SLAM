#!/usr/bin/env python3
"""
🧪 Dummy Pipeline Test - Test AMR vision pipeline without a real camera.

This script simulates the full pipeline:
1. Publishes dummy 2D landmarks to /vision/landmarks_2d
2. PnP node receives landmarks and computes 6DoF pose
3. Outputs robot pose to /vision/robot_pose

Usage:
    # Terminal 1: Start PnP node
    ros2 run amr_pose_estimator pnp_node
    
    # Terminal 2: Run this test
    python test_dummy_pipeline.py
    
    # Terminal 3: Monitor output
    ros2 topic echo /vision/robot_pose
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import time
import math


class DummyPipelineTest(Node):
    """
    Publishes simulated 2D keypoints to test the PnP pipeline.
    
    Simulates a robot facing the net with 4 pole keypoints visible.
    """
    
    def __init__(self):
        super().__init__('dummy_pipeline_test')
        
        # Publisher for simulated landmarks
        self.landmarks_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/vision/landmarks_2d', 10)
        
        # Subscriber for computed pose
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/vision/robot_pose',
            self.pose_callback, 10)
        
        # Timer to publish landmarks at 10 Hz
        self.timer = self.create_timer(0.1, self.publish_dummy_landmarks)
        
        # State
        self.publish_count = 0
        self.last_pose = None
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("🧪 DUMMY PIPELINE TEST STARTED")
        self.get_logger().info("=" * 50)
        self.get_logger().info("Publishing simulated landmarks to /vision/landmarks_2d")
        self.get_logger().info("Waiting for pose response on /vision/robot_pose")
        self.get_logger().info("Press Ctrl+C to stop")
        self.get_logger().info("=" * 50)
    
    def publish_dummy_landmarks(self):
        """
        Publish simulated pole keypoints.
        
        Simulates a 640x480 camera view with the net visible:
        - Left pole at x=120 (near left edge)
        - Right pole at x=520 (near right edge)
        - Base at y=400 (lower part of image)
        - Top at y=180 (upper part of image)
        """
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        
        # Simulated keypoints for a 640x480 image
        # The robot is facing the net, poles are symmetrically visible
        
        # Pole keypoint coordinates (pixel values)
        left_base_x, left_base_y = 120.0, 400.0
        left_top_x, left_top_y = 120.0, 180.0
        right_base_x, right_base_y = 520.0, 400.0
        right_top_x, right_top_y = 520.0, 180.0
        
        # Encode into PoseWithCovarianceStamped message format
        # (workaround for using standard message type)
        msg.pose.pose.position.x = left_base_x
        msg.pose.pose.position.y = left_base_y
        msg.pose.pose.position.z = left_top_x
        msg.pose.pose.orientation.x = left_top_y
        msg.pose.pose.orientation.y = right_base_x
        msg.pose.pose.orientation.z = right_base_y
        msg.pose.pose.orientation.w = right_top_x
        msg.pose.covariance[0] = right_top_y
        
        # Set low covariance (high confidence)
        for i in range(6):
            msg.pose.covariance[i * 7] = 0.01
        
        self.landmarks_pub.publish(msg)
        self.publish_count += 1
        
        if self.publish_count % 10 == 0:  # Log every second
            self.get_logger().info(
                f"📤 Published {self.publish_count} landmark sets | "
                f"Left: ({left_base_x:.0f},{left_base_y:.0f}) | "
                f"Right: ({right_base_x:.0f},{right_base_y:.0f})")
    
    def pose_callback(self, msg):
        """Receive and display computed pose"""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        
        # Convert quaternion to yaw for display
        siny_cosp = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.get_logger().info(
            f"🎯 POSE RECEIVED | "
            f"Position: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}) m | "
            f"Yaw: {math.degrees(yaw):.1f}°")
        
        self.last_pose = msg


def main(args=None):
    rclpy.init(args=args)
    node = DummyPipelineTest()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\n👋 Test completed!")
        if node.last_pose:
            pos = node.last_pose.pose.pose.position
            node.get_logger().info(
                f"📊 Final pose: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
