# Tennis-Court-Visual-SLAM

This repository contains vision-based modules for an Autonomous Mobile Robot (AMR) operating on a tennis court. It includes Inverse Perspective Mapping (IPM) for ground grid detection and a ROS 2 node for 6DoF pose estimation relative to court landmarks.

## Project Structure

- `ipm/`: Inverse Perspective Mapping module for bird's-eye view generation and grid detection.
- `pnp_ros/`: ROS 2 package for 6DoF pose estimation using OpenCV PnP.
- `keypoint_detection/`: Custom YOLOv8-Pose keypoint detection for net poles and grid intersections.
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

## Dependencies

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- SciPy
- Ultralytics (YOLOv8)
- ROS 2 (Humble/Iron/Jazzy)
- `geometry_msgs`, `rclpy`

## License

This project is licensed under the Apache License 2.0.
