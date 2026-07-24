# 🎾 Tennis-Court-Visual-SLAM 🤖

> **Autonomous Mobile Robot (AMR) vision system for tennis court navigation and ball-throwing**

A comprehensive vision-based perception stack for an AMR operating on a tennis court. Features include inverse perspective mapping, deep learning keypoint detection, semantic segmentation, Visual SLAM integration, and a state machine for ball-throwing control.

---

## 📁 Project Structure

```
Tennis-Court-Visual-SLAM/
├── 📷 ipm/                    # Inverse Perspective Mapping
├── 🧠 keypoint_detection/     # YOLOv8-Pose keypoint detection
├── 🎯 segmentation/           # Semantic segmentation (pitch lines & poles)
├── 🤖 pnp_ros/                # ROS 2 package (PnP, SLAM, EKF, SMACH)
├── 📊 Report.md               # Technical report
└── 📖 README.md               # This file
```

---

## 🚀 Modules

### 1. 📷 Inverse Perspective Mapping (IPM)

Transforms perspective camera feed into a top-down (bird's-eye) view for ground grid detection.

| Feature | Description |
|---------|-------------|
| 🏗️ Homography | Computed from camera height (0.6m) and tilt (15°) |
| 📏 Line Detection | Canny edge detection + HoughLinesP |
| 📐 Metric Conversion | Pixel coordinates → meters |

```bash
cd ipm && python3 test_ipm.py
```

---

### 2. 🧠 YOLOv8-Pose Keypoint Detection

Deep learning perception front-end for detecting tennis court keypoints.

| Feature | Description |
|---------|-------------|
| 🎯 8 Keypoints | 4 net poles + 4 grid intersections |
| ⚡ Real-time | 30 FPS with confidence filtering (≥0.6) |
| 🔗 PnP Integration | Direct feed to `cv2.solvePnP` |

**Keypoint Map:**

| Index | Name | 3D Coords (m) |
|-------|------|---------------|
| 0 | 📍 left_pole_base | (-5.485, 0.0, 0.0) |
| 1 | 📍 left_pole_top | (-5.485, 0.0, 1.07) |
| 2 | 📍 right_pole_base | (5.485, 0.0, 0.0) |
| 3 | 📍 right_pole_top | (5.485, 0.0, 1.07) |
| 4 | 🔲 service_line_center | (0.0, 6.40, 0.0) |
| 5 | 🔲 service_line_left | (-5.485, 6.40, 0.0) |
| 6 | 🔲 baseline_center | (0.0, 11.885, 0.0) |
| 7 | 🔲 service_line_right | (5.485, 6.40, 0.0) |

```bash
# Generate sample dataset
python keypoint_detection/dataset/create_sample_dataset.py

# Run live inference
python -m keypoint_detection.inference.live_stream --model weights/best.pt --camera 0
```

---

### 3. 🎯 Semantic Segmentation

Pixel-level segmentation of pitch lines and net poles using Fast-SCNN/U-Net.

| Class | Description | Color |
|-------|-------------|-------|
| 1 | 🟢 Pitch Lines | Green |
| 2 | 🔴 Poles | Red |

```bash
python -m segmentation.inference.live_stream --model weights/seg_model.onnx --backend onnx
```

---

### 4. 🤖 ROS 2 Pose Estimation Pipeline

Complete perception pipeline from camera to 6DoF robot pose.

```
📸 /camera/image_raw
       │
       ▼
🧠 [yolo_pose_node]  ← YOLOv8-Pose (conf > 0.6)
       │
       ▼
📊 /vision/landmarks_2d  ← 4 pole keypoints
       │
       ▼
🎯 [pnp_node]  ← cv2.solvePnP
       │
       ▼
🗺️ /vision/robot_pose  ← 6DoF pose in map frame
```

**Launch Options:**

```bash
# 🟢 Basic PnP (with dummy data)
ros2 launch amr_pose_estimator pnp_launch.py

# 🧠 Full YOLOv8-Pose + PnP pipeline
ros2 launch amr_pose_estimator yolo_pnp_launch.py

# 🧠 With GPU acceleration
ros2 launch amr_pose_estimator yolo_pnp_launch.py device:=cuda

# 🗺️ Full Visual SLAM integration
ros2 launch amr_pose_estimator slam_launch.py
```

---

### 5. 🗺️ Visual SLAM Integration (ORB-SLAM3)

Standard ROS 2 TF tree with SLAM-based localization.

```
🌍 map (pitch frame, origin at net center)
 └── 📍 odom (SLAM odometry frame)
      └── 🤖 base_link (robot base)
           └── 📷 camera_link (x=0, y=0, z=0.6m)
```

| Feature | Description |
|---------|-------------|
| 🔄 TF Broadcaster | `map` → `odom` → `base_link` |
| 🎯 Initial Pose | Aligns SLAM to pitch coordinates |
| 👁️ Status Monitor | OK / LOST / REINITIALIZING |
| 🔧 Drift Correction | Periodic PnP alignment |

```bash
# View TF tree
ros2 run tf2_tools view_frames

# Monitor status
ros2 topic echo /vision/slam_status
```

---

### 6. 🎮 Ball-Throwing State Machine (SMACH)

Autonomous state machine for tracking and throwing tennis balls.

```
┌─────────────────┐
│ 🔄 LOCATING_PITCH│ ← Rotate 15°/s until confidence > 0.75
└────────┬────────┘
         │ (pitch detected)
         ▼
┌─────────────────┐
│ 🎯 TRACKING_AIM  │ ← Lock pose, compute launcher angle
└────────┬────────┘
         │ (vision lost > 1.5s)
         ▼
┌─────────────────┐
│ ⚠️ VISION_LOST   │ ← Dead reckoning, check orientation
└────────┬────────┘
         │ (error > 45° or timeout)
         └──────────→ 🔄 LOCATING_PITCH
```

```bash
ros2 launch amr_pose_estimator slam_launch.py  # Includes SMACH node
```

---

### 7. 🧭 ArUco Marker Reorientation

Detects ArUco markers when the robot is not facing the pitch.

| Feature | Description |
|---------|-------------|
| 📌 Dictionary | `DICT_6X6_250` |
| 📏 Marker Size | 0.2m |
| 🔄 Reorientation | Proportional control to rotate toward pitch |

```bash
# Standalone test
python keypoint_detection/utils/aruco_reorientation.py
```

---

### 8. 📡 Sensor Fusion (EKF)

Fuses wheel odometry, IMU, and visual pose for continuous tracking.

| Sensor | Topic | Trust Level |
|--------|-------|-------------|
| 🛞 Wheel Odometry | `/odom/wheels` | HIGH |
| 📐 IMU | `/imu/data` | HIGH |
| 👁️ Visual Pose | `/vision/robot_pose` | MEDIUM |

```bash
# EKF config
cat pnp_ros/amr_pose_estimator/config/ekf.yaml
```

---

## 🧪 Testing Without a Camera (Dummy Mode)

### Quick Test Script

```python
#!/usr/bin/env python3
"""test_dummy_pipeline.py - Test the full pipeline with simulated data"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
import time
import math

class DummyPipelineTest(Node):
    def __init__(self):
        super().__init__('dummy_pipeline_test')
        
        # Publisher for simulated landmarks
        self.landmarks_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/vision/landmarks_2d', 10)
        
        # Subscriber for robot pose
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/vision/robot_pose',
            self.pose_callback, 10)
        
        self.timer = self.create_timer(0.1, self.publish_dummy_landmarks)  # 10 Hz
        self.get_logger().info("🧪 Dummy Pipeline Test Started")
    
    def publish_dummy_landmarks(self):
        """Publish simulated pole keypoints"""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        
        # Simulated keypoints (640x480 image)
        # Format: position = (x1,y1,x2), orientation = (y2,x3,y3,x4), covariance[0] = y4
        msg.pose.pose.position.x = 120.0   # Left Base x
        msg.pose.pose.position.y = 400.0   # Left Base y
        msg.pose.pose.position.z = 120.0   # Left Top x
        msg.pose.pose.orientation.x = 180.0 # Left Top y
        msg.pose.pose.orientation.y = 520.0 # Right Base x
        msg.pose.pose.orientation.z = 400.0 # Right Base y
        msg.pose.pose.orientation.w = 520.0 # Right Top x
        msg.pose.covariance[0] = 180.0      # Right Top y
        
        self.landmarks_pub.publish(msg)
        self.get_logger().info("📤 Published dummy landmarks")
    
    def pose_callback(self, msg):
        """Receive computed pose"""
        pos = msg.pose.pose.position
        self.get_logger().info(
            f"🎯 Received pose: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")

def main():
    rclpy.init()
    node = DummyPipelineTest()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Running the Dummy Test

```bash
# Terminal 1: Start the PnP node
ros2 run amr_pose_estimator pnp_node

# Terminal 2: Run the dummy test
python test_dummy_pipeline.py

# Terminal 3: Monitor output
ros2 topic echo /vision/robot_pose
```

### 🎮 YOLOv8-Pose Dummy Mode

The `yolo_pose_node` includes built-in dummy mode when no model is loaded:

```bash
# Start with non-existent model (triggers dummy mode)
ros2 run amr_pose_estimator yolo_pose_node --ros-args -p model_path:=nonexistent.pt

# Or use OpenCV VideoCapture without a camera (will fail gracefully)
ros2 run amr_pose_estimator yolo_pose_node --ros-args -p use_camera_topic:=false -p camera_id:=99
```

**Dummy Mode Behavior:**
- 🟢 Generates simulated keypoints for a typical tennis court view
- 🟢 Keypoint confidence: 0.70-0.88 (above 0.6 threshold)
- 🟢 Publishes to `/vision/landmarks_2d` at 10 Hz
- 🟢 PnP node computes pose from simulated data

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| 🐍 Python 3.x | Core language |
| 📷 OpenCV (`cv2`) | Computer vision |
| 🔢 NumPy | Numerical computing |
| 📐 SciPy | Rotation math |
| 🧠 Ultralytics | YOLOv8 inference |
| 🤖 ROS 2 (Humble/Iron/Jazzy) | Robot middleware |
| 📨 `geometry_msgs` | Pose messages |
| 📡 `sensor_msgs` | Image messages |
| 🔗 `tf2_ros` | Transform broadcasting |
| 📊 `robot_localization` | EKF sensor fusion |
| 🎮 `smach` | State machine |

---

## 📜 License

This project is licensed under the **Apache License 2.0**.

---

<div align="center">

**🎾 Built for autonomous tennis court robots 🤖**

[![ROS 2](https://img.shields.io/badge/ROS-2-Humble%2FIron%2FJazzy-blue)](https://docs.ros.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-red)](https://ultralytics.com/)

</div>
