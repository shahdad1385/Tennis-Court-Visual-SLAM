# Tennis-Court-Visual-SLAM

This repository contains vision-based modules for an Autonomous Mobile Robot (AMR) operating on a tennis court. It includes Inverse Perspective Mapping (IPM) for ground grid detection and a ROS 2 node for 6DoF pose estimation relative to court landmarks.

## Project Structure

- `ipm/`: Inverse Perspective Mapping module for bird's-eye view generation and grid detection.
- `pnp_ros/`: ROS 2 package for 6DoF pose estimation using OpenCV PnP.
- `keypoint_detection/`: Custom YOLOv8-Pose keypoint detection for net poles and grid intersections.
- `segmentation/`: Semantic segmentation for pitch lines and net poles.
- `pnp_ros/`: ROS 2 package for 6DoF pose estimation and Visual SLAM integration.
- `Report.md`: Detailed technical report of the implementation.

## Modules

### 1. Inverse Perspective Mapping (IPM)

The `ipm_ground_grid.py` script transforms a perspective camera feed into a top-down (bird's-eye) view.

**Features:**
- Computes homography based on camera height (0.6m) and tilt (15°).
- Detects ground grid lines using Canny and HoughLinesP.
- Converts pixel coordinates to metric distances (meters).

**Usage:**
```bash
cd ipm
python3 test_ipm.py
```

### 2. 6DoF Pose Estimation (PnP)

The `amr_pose_estimator` ROS 2 node estimates the robot's pose relative to net poles.

**Features:**
- Uses `cv2.solvePnP` with 4 known 3D landmarks (net poles).
- Publishes `geometry_msgs/PoseWithCovarianceStamped` on `/vision/robot_pose`.

**Usage:**
```bash
# Build and run in a ROS 2 workspace
cd pnp_ros
colcon build
ros2 launch amr_pose_estimator pnp_launch.py
```

### 3. Custom Keypoint Detection (YOLOv8-Pose)

The `keypoint_detection/` module provides a trained YOLOv8-Pose model to detect 8 keypoints on the tennis court: 4 net poles and 4 grid intersections.

**Keypoints (8 total):**

| Index | Name | 3D World Coords (m) |
|-------|------|---------------------|
| 0 | left_pole_base | (-5.485, 0.0, 0.0) |
| 1 | left_pole_top | (-5.485, 0.0, 1.07) |
| 2 | right_pole_base | (5.485, 0.0, 0.0) |
| 3 | right_pole_top | (5.485, 0.0, 1.07) |
| 4 | service_line_center | (0.0, 6.40, 0.0) |
| 5 | service_line_left | (-5.485, 6.40, 0.0) |
| 6 | baseline_center | (0.0, 11.885, 0.0) |
| 7 | service_line_right | (5.485, 6.40, 0.0) |

**Usage:**
```bash
# Generate sample dataset for testing
python keypoint_detection/dataset/create_sample_dataset.py

# Run live inference
python -m keypoint_detection.inference.live_stream --model weights/best.pt --camera 0
```

**Key Features:**
- 30 FPS real-time inference with confidence filtering (≥0.6)
- Temporal smoothing for stable keypoint tracking
- Direct integration with `cv2.solvePnP` via `solvepnp_bridge.py`
- YOLO pose format with 8 keypoints and visibility flags

### 4. Semantic Segmentation (Fast-SCNN / U-Net)

The `segmentation/` module provides a pipeline for pixel-level segmentation of pitch lines and net poles using TensorRT or ONNX models.

**Classes:**
- **Class 1 (Pitch Lines):** Ground lines for grid detection.
- **Class 2 (Poles):** Vertical net poles for localization.

**Usage:**
```bash
# Run live segmentation with an ONNX model
python -m segmentation.inference.live_stream --model weights/seg_model.onnx --backend onnx
```

**Key Features:**
- Lightweight ONNX/TensorRT inference engine.
- Morphological post-processing (dilation/erosion) to clean edge noise.
- Integrated `GridCalculator` to estimate grid distances from segmented masks using homography.
- Real-time bird's-eye view visualization of segmented lines.

### 5. Visual SLAM Integration (ORB-SLAM3)

The `pnp_ros/` package includes ROS 2 nodes for integrating ORB-SLAM3 with the AMR, establishing the standard TF tree and aligning the SLAM frame to the physical pitch coordinate system.

**TF Tree Structure:**
```
map (pitch frame, origin at net center)
 └── odom (SLAM odometry frame)
      └── base_link (robot base)
           └── camera_link (x=0, y=0, z=0.6m)
```

**Features:**
- **SLAM TF Broadcaster**: Broadcasts `map` → `odom` → `base_link` transforms
- **Initial Pose Alignment**: Aligns SLAM frame to pitch coordinates when net poles detected
- **Tracking Monitor**: Monitors ORB-SLAM3 status (OK, LOST, REINITIALIZING)
- **Visual Loss Recovery**: Falls back to PnP pose when SLAM tracking is lost
- **Drift Correction**: Periodic alignment using PnP pole detections

**Launch Files:**
```bash
# Launch PnP only (no SLAM)
ros2 launch amr_pose_estimator pnp_launch.py

# Launch full Visual SLAM integration
ros2 launch amr_pose_estimator slam_launch.py

# With custom camera matrix
ros2 launch amr_pose_estimator slam_launch.py camera_matrix:="[600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]"
```

**Configuration Files:**
- `config/orb_slam3_params.yaml`: ORB-SLAM3 camera and feature parameters
- `config/slam_config.yaml`: TF tree, pitch coordinates, and recovery settings

**Monitoring:**
```bash
# View TF tree
ros2 run tf2_tools view_frames

# Monitor SLAM status
ros2 topic echo /vision/slam_status

# Check transforms
ros2 topic echo /tf
```

## Dependencies

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- SciPy
- Ultralytics (YOLOv8)
- ROS 2 (Humble/Iron/Jazzy)
- `geometry_msgs`, `sensor_msgs`, `std_msgs`
- `tf2_ros`, `tf2_geometry_msgs`
- `nav_msgs`
- `rclpy`

## License

This project is licensed under the Apache License 2.0.
