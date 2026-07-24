# Tennis-Court-Visual-SLAM

This repository contains vision-based modules for an Autonomous Mobile Robot (AMR) operating on a tennis court. It includes Inverse Perspective Mapping (IPM) for ground grid detection and a ROS 2 node for 6DoF pose estimation relative to court landmarks.

## Project Structure

- `ipm/`: Inverse Perspective Mapping module for bird's-eye view generation and grid detection.
- `pnp_ros/`: ROS 2 package for 6DoF pose estimation using OpenCV PnP.
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

## Dependencies

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- SciPy
- ROS 2 (Humble/Iron/Jazzy)
- `geometry_msgs`, `rclpy`

## License

This project is licensed under the Apache License 2.0.
