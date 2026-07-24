# Report: AMR Vision-Based Pose Estimation and Grid Detection

## 1. Overview
This report details the implementation of two core vision modules for an Autonomous Mobile Robot (AMR) operating on a tennis court:
1.  **Inverse Perspective Mapping (IPM)**: Transforms camera feed into a bird's-eye view for ground grid detection and metric distance estimation.
2.  **6DoF Pose Estimation**: Estimates the robot's position and orientation relative to fixed landmarks (net poles) using OpenCV's PnP algorithm.

## 2. Inverse Perspective Mapping (IPM)
### 2.1 Methodology
- **Coordinate Transformation**: The camera is mounted at a height of 0.6m with a 15-degree downward tilt. The ground plane (Z=0) is mapped to the image plane using a homography matrix computed from camera intrinsics and extrinsic parameters.
- **Homography Calculation**: Four corners of a 4m x 10m region on the pitch are projected to the image plane. A perspective transform is then derived to warp the image to a top-down view.
- **Line Detection**: Canny edge detection and Progressive Probabilistic Hough Transform (HoughLinesP) are used to identify grid lines in the warped image.
- **Metric Conversion**: Detected pixel coordinates are scaled back to metric distances (meters) relative to the camera base using linear interpolation.

### 2.2 Implementation
- **File**: `ipm/ipm_ground_grid.py`
- **Key Feature**: `IPMProcessor` class encapsulates the transformation logic and line detection.

## 3. 6DoF Pose Estimation (PnP)
### 3.1 Methodology
- **Landmarks**: Four 3D points are defined: Left Pole Base/Top and Right Pole Base/Top (World Frame centered at the net).
- **Solver**: `cv2.solvePnP` with `SOLVEPNP_ITERATIVE` computes the transformation from the world (poles) to the camera.
- **Pose Inversion**: The raw PnP result (World-to-Camera) is inverted to obtain the Camera-to-World transformation (Robot Pose).
- **Quaternion Output**: The rotation vector is converted to a quaternion using `scipy.spatial.transform.Rotation` for standard ROS compatibility.

### 3.2 ROS 2 Integration
- **Package**: `pnp_ros/amr_pose_estimator`
- **Topics**:
    - Subscribes to `/vision/landmarks_2d` (2D keypoints).
    - Publishes to `/vision/robot_pose` (`geometry_msgs/PoseWithCovarianceStamped`).

## 4. Conclusion
The implemented modules provide the foundational vision stack for an AMR to navigate and localize itself on a tennis court. The IPM module enables grid-based path planning, while the PnP module offers absolute localization relative to the court's geometry.
