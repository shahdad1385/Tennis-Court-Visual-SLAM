"""
Launch file for YOLOv8-Pose + PnP Pipeline.

Launches:
1. yolo_pose_node: Detects keypoints from camera feed
2. pnp_node: Computes 6DoF pose from keypoints

Pipeline:
  /camera/image_raw -> [yolo_pose_node] -> /vision/landmarks_2d -> [pnp_node] -> /vision/robot_pose
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('amr_pose_estimator')
    
    # Launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='weights/best.pt',
        description='Path to YOLOv8-Pose model weights'
    )
    
    conf_threshold_arg = DeclareLaunchArgument(
        'conf_threshold',
        default_value='0.6',
        description='Minimum detection confidence threshold'
    )
    
    kpt_conf_threshold_arg = DeclareLaunchArgument(
        'kpt_conf_threshold',
        default_value='0.6',
        description='Minimum keypoint confidence threshold'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Inference device (cpu, cuda, mps)'
    )
    
    use_camera_topic_arg = DeclareLaunchArgument(
        'use_camera_topic',
        default_value='true',
        description='Use ROS 2 camera topic (true) or OpenCV VideoCapture (false)'
    )
    
    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='Camera device ID for OpenCV VideoCapture'
    )
    
    camera_matrix_arg = DeclareLaunchArgument(
        'camera_matrix',
        default_value='[600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]',
        description='Camera intrinsic matrix (flat 9-element list)'
    )
    
    # YOLOv8-Pose Detection Node
    yolo_pose_node = Node(
        package='amr_pose_estimator',
        executable='yolo_pose_node',
        name='yolo_pose_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'conf_threshold': LaunchConfiguration('conf_threshold'),
            'kpt_conf_threshold': LaunchConfiguration('kpt_conf_threshold'),
            'device': LaunchConfiguration('device'),
            'imgsz': 640,
            'camera_frame': 'camera_link',
            'use_camera_topic': LaunchConfiguration('use_camera_topic'),
            'camera_id': LaunchConfiguration('camera_id'),
        }]
    )
    
    # PnP Pose Estimation Node
    pnp_node = Node(
        package='amr_pose_estimator',
        executable='pnp_node',
        name='pnp_pose_node',
        output='screen',
        parameters=[{
            'camera_matrix': LaunchConfiguration('camera_matrix'),
        }]
    )
    
    return LaunchDescription([
        # Launch arguments
        model_path_arg,
        conf_threshold_arg,
        kpt_conf_threshold_arg,
        device_arg,
        use_camera_topic_arg,
        camera_id_arg,
        camera_matrix_arg,
        
        # Nodes
        yolo_pose_node,
        pnp_node,
    ])
