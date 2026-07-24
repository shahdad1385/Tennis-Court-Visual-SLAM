"""
ROS 2 Node for YOLOv8-Pose Keypoint Detection.

Subscribes to camera feed, runs YOLOv8-Pose inference, and publishes
detected keypoints to /vision/landmarks_2d for PnP processing.

Pipeline Integration:
  /camera/image_raw -> [yolo_pose_node] -> /vision/landmarks_2d -> [pnp_node] -> /vision/robot_pose
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math


# =========================================
# Keypoint Index Mapping (matches keypoints_schema.py)
# =========================================
# 0: left_pole_base, 1: left_pole_top
# 2: right_pole_base, 3: right_pole_top
# 4: service_line_center, 5: service_line_left
# 6: baseline_center, 7: service_line_right

POLE_INDICES = [0, 1, 2, 3]  # Primary 4 keypoints for PnP
GRID_INDICES = [4, 5, 6, 7]  # Additional grid keypoints


class YoloPoseNode(Node):
    """
    ROS 2 node that runs YOLOv8-Pose inference on camera images
    and publishes detected keypoints for PnP pose estimation.
    """
    
    def __init__(self):
        super().__init__('yolo_pose_node')
        
        # ---- Parameters ----
        self.declare_parameter('model_path', 'weights/best.pt')
        self.declare_parameter('conf_threshold', 0.6)
        self.declare_parameter('kpt_conf_threshold', 0.6)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('use_camera_topic', True)
        self.declare_parameter('camera_id', 0)
        
        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.kpt_conf_threshold = self.get_parameter('kpt_conf_threshold').value
        self.device = self.get_parameter('device').value
        self.imgsz = self.get_parameter('imgsz').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.use_camera_topic = self.get_parameter('use_camera_topic').value
        self.camera_id = self.get_parameter('camera_id').value
        
        # ---- State Variables ----
        self.bridge = CvBridge()
        self.model = None
        self.cap = None
        self.frame_count = 0
        self.fps = 0.0
        self.fps_timer = time.time()
        
        # ---- Publishers ----
        # Primary output: 2D keypoints for PnP processing
        # Format: Float32MultiArray with [x1,y1,x2,y2,x3,y3,x4,y4] for 4 pole keypoints
        self.keypoints_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/vision/landmarks_2d', 10)
        
        # Debug: All keypoints (8x2 = 16 floats)
        self.all_keypoints_pub = self.create_publisher(
            Float32MultiArray, '/vision/all_keypoints', 10)
        
        # Status publisher
        self.status_pub = self.create_publisher(String, '/vision/detection_status', 10)
        
        # ---- Subscribers ----
        if self.use_camera_topic:
            # Subscribe to ROS 2 camera topic
            image_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST
            )
            self.image_sub = self.create_subscription(
                Image, '/camera/image_raw', self.image_callback, image_qos)
            self.get_logger().info(f"Subscribing to /camera/image_raw")
        else:
            # Use OpenCV VideoCapture
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.timer = self.create_timer(0.033, self.timer_callback)  # ~30 Hz
            self.get_logger().info(f"Using OpenCV VideoCapture (camera {self.camera_id})")
        
        # ---- Load Model ----
        self.load_model()
        
        self.get_logger().info("YOLOv8-Pose Node initialized")
        self.get_logger().info(f"  Model: {self.model_path}")
        self.get_logger().info(f"  Confidence threshold: {self.conf_threshold}")
        self.get_logger().info(f"  Keypoint confidence threshold: {self.kpt_conf_threshold}")
    
    def load_model(self):
        """Load the YOLOv8-Pose model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"Model loaded: {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.get_logger().warn("Running in dummy mode - will use simulated keypoints")
            self.model = None
    
    def image_callback(self, msg):
        """Callback for ROS 2 Image message."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_frame(frame)
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
    
    def timer_callback(self):
        """Callback for OpenCV VideoCapture timer."""
        if self.cap is None or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if ret:
            self.process_frame(frame)
    
    def process_frame(self, frame: np.ndarray):
        """
        Process a single frame through YOLOv8-Pose and publish keypoints.
        
        Args:
            frame: BGR image from camera
        """
        self.frame_count += 1
        
        # Update FPS counter
        elapsed = time.time() - self.fps_timer
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_timer = time.time()
        
        # Run inference
        detections = self.detect_keypoints(frame)
        
        if detections:
            # Get best detection (highest confidence)
            best = max(detections, key=lambda d: d['bbox_conf'])
            kpts = best['keypoints']  # (8, 3) - x, y, visibility
            kpt_conf = best['kpt_confs']  # (8,)
            
            # Extract pole keypoints (indices 0-3) with confidence filtering
            pole_points = self.extract_pole_keypoints(kpts, kpt_conf)
            
            if pole_points is not None:
                # Publish to /vision/landmarks_2d
                self.publish_landmarks(pole_points)
                
                # Publish all keypoints for debugging
                self.publish_all_keypoints(kpts)
                
                # Publish status
                self.publish_status("OK", len(detections), self.fps)
            else:
                self.publish_status("INSUFFICIENT_KEYPOINTS", 0, self.fps)
        else:
            self.publish_status("NO_DETECTION", 0, self.fps)
    
    def detect_keypoints(self, frame: np.ndarray) -> list:
        """
        Run YOLOv8-Pose inference on a frame.
        
        Args:
            frame: BGR image
            
        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            # Dummy mode: return simulated detection for testing
            return self.get_dummy_detection(frame.shape)
        
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False
        )
        
        detections = []
        for r in results:
            if r.keypoints is None:
                continue
            
            kpts_xy = r.keypoints.xy[0].cpu().numpy()
            kpt_conf = r.keypoints.conf[0].cpu().numpy()
            bbox = r.boxes.xyxy[0].cpu().numpy()
            bbox_conf = float(r.boxes.conf[0])
            
            kpt_vis = r.keypoints.vis[0].cpu().numpy() if r.keypoints.vis is not None else np.ones(len(kpts_xy))
            keypoints = np.column_stack([kpts_xy, kpt_vis])
            
            detections.append({
                'bbox': bbox.tolist(),
                'bbox_conf': bbox_conf,
                'keypoints': keypoints,
                'kpt_confs': kpt_conf
            })
        
        return detections
    
    def extract_pole_keypoints(self, kpts: np.ndarray, kpt_conf: np.ndarray):
        """
        Extract 4 pole keypoints with confidence filtering.
        
        Args:
            kpts: (N, 3) array of [x, y, visibility]
            kpt_conf: (N,) array of per-keypoint confidence
            
        Returns:
            (4, 2) array of pole keypoint coordinates, or None if insufficient
        """
        # Check if all 4 pole keypoints are above threshold
        pole_mask = [kpt_conf[i] >= self.kpt_conf_threshold for i in POLE_INDICES]
        
        if not all(pole_mask):
            missing = [POLE_INDICES[i] for i, m in enumerate(pole_mask) if not m]
            self.get_logger().debug(f"Missing pole keypoints: {missing}")
            return None
        
        # Extract pole keypoints
        pole_points = kpts[POLE_INDICES, :2].astype(np.float64)
        
        # Enforce spatial ordering: left pole x < right pole x
        if pole_points[0, 0] > pole_points[2, 0]:
            pole_points = pole_points[[2, 3, 0, 1]]
        
        # Enforce vertical ordering: pole top y < pole base y (image coords)
        if pole_points[0, 1] > pole_points[1, 1]:
            pole_points[[0, 1]] = pole_points[[1, 0]]
        if pole_points[2, 1] > pole_points[3, 1]:
            pole_points[[2, 3]] = pole_points[[3, 2]]
        
        return pole_points
    
    def publish_landmarks(self, pole_points: np.ndarray):
        """
        Publish pole keypoints to /vision/landmarks_2d.
        
        The PnP node subscribes to this topic and uses the 4 (x,y) pairs
        to solve for the robot's 6DoF pose relative to the net poles.
        
        Args:
            pole_points: (4, 2) array of pixel coordinates
        """
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        
        # Encode 4 pole keypoints as pose (x,y position + z as secondary point)
        # This is a workaround using PoseWithCovarianceStamped as the message type
        # Format: position = (x1, y1, z1=x2), orientation = (x3, y3, z3=x4, w=y4)
        # The PnP node will parse these accordingly
        
        # Pole 1 (Left Base): position
        msg.pose.pose.position.x = float(pole_points[0, 0])
        msg.pose.pose.position.y = float(pole_points[0, 1])
        msg.pose.pose.position.z = float(pole_points[1, 0])  # Left Top x
        
        # Pole 2 (Left Top + Right Base): orientation
        msg.pose.pose.orientation.x = float(pole_points[1, 1])  # Left Top y
        msg.pose.pose.orientation.y = float(pole_points[2, 0])  # Right Base x
        msg.pose.pose.orientation.z = float(pole_points[2, 1])  # Right Base y
        msg.pose.pose.orientation.w = float(pole_points[3, 0])  # Right Top x
        
        # Store Right Top y in covariance[0] as a workaround
        msg.pose.covariance[0] = float(pole_points[3, 1])
        
        # Set covariance based on keypoint confidence (lower = more confident)
        # Use diagonal covariance matrix
        for i in range(6):
            msg.pose.covariance[i * 7] = 0.01  # Low variance = high confidence
        
        self.keypoints_pub.publish(msg)
        self.get_logger().debug(
            f"Published pole keypoints: {pole_points.tolist()}")
    
    def publish_all_keypoints(self, kpts: np.ndarray):
        """Publish all 8 keypoints for debugging."""
        msg = Float32MultiArray()
        # Flatten to [x1,y1,x2,y2,...,x8,y8]
        msg.data = kpts[:, :2].flatten().tolist()
        self.all_keypoints_pub.publish(msg)
    
    def publish_status(self, status: str, num_detections: int, fps: float):
        """Publish detection status."""
        msg = String()
        msg.data = f"{status}|detections={num_detections}|fps={fps:.1f}"
        self.status_pub.publish(msg)
    
    def get_dummy_detection(self, frame_shape):
        """
        Generate dummy detection for testing without a trained model.
        Simulates 4 pole keypoints in a typical camera view.
        """
        h, w = frame_shape[:2]
        
        # Simulate keypoints based on typical tennis court view
        kpts = np.array([
            [w * 0.2, h * 0.8],   # Left Pole Base
            [w * 0.2, h * 0.3],   # Left Pole Top
            [w * 0.8, h * 0.8],   # Right Pole Base
            [w * 0.8, h * 0.3],   # Right Pole Top
            [w * 0.5, h * 0.5],   # Service Line Center
            [w * 0.2, h * 0.5],   # Service Line Left
            [w * 0.5, h * 0.9],   # Baseline Center
            [w * 0.8, h * 0.5],   # Service Line Right
        ], dtype=np.float32)
        
        kpt_conf = np.array([0.85, 0.82, 0.88, 0.80, 0.75, 0.70, 0.72, 0.71])
        
        bbox = [w * 0.1, h * 0.2, w * 0.9, h * 0.9]
        
        return [{
            'bbox': bbox,
            'bbox_conf': 0.9,
            'keypoints': np.column_stack([kpts, np.ones(8)]),
            'kpt_confs': kpt_conf
        }]
    
    def destroy_node(self):
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloPoseNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
