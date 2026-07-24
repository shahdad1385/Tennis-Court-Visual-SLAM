"""
Visual SLAM Integration Launch File
Launches ORB-SLAM3, PnP pose estimation, TF broadcaster, and monitoring.
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
    camera_matrix_arg = DeclareLaunchArgument(
        'camera_matrix',
        default_value='[600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]',
        description='Camera intrinsic matrix (flat 9-element list)'
    )
    
    camera_height_arg = DeclareLaunchArgument(
        'camera_height',
        default_value='0.6',
        description='Camera height above base_link in meters'
    )
    
    use_slam_arg = DeclareLaunchArgument(
        'use_slam',
        default_value='true',
        description='Enable ORB-SLAM3 integration'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz for visualization'
    )
    
    # Config file path
    config_dir = os.path.join(pkg_share, 'config')
    orb_slam_config = os.path.join(config_dir, 'orb_slam3_params.yaml')
    slam_config = os.path.join(config_dir, 'slam_config.yaml')
    
    # Static transform: base_link -> camera_link
    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=[
            '0', '0', LaunchConfiguration('camera_height'),  # x, y, z
            '0', '0', '0', '1',  # roll, pitch, yaw, w
            'base_link',
            'camera_link'
        ],
        output='screen'
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
    
    # SLAM TF Broadcaster Node
    slam_tf_node = Node(
        package='amr_pose_estimator',
        executable='slam_tf_broadcaster',
        name='slam_tf_broadcaster',
        output='screen',
        parameters=[{
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'camera_frame': 'camera_link',
            'camera_height': LaunchConfiguration('camera_height'),
            'min_pole_detections': 4,
            'confidence_threshold': 0.6,
            'max_lost_frames': 30,
        }]
    )
    
    # Tracking Monitor Node
    tracking_monitor_node = Node(
        package='amr_pose_estimator',
        executable='tracking_monitor',
        name='tracking_monitor',
        output='screen',
        parameters=[{
            'max_lost_frames': 30,
            'reinit_attempts': 3,
            'status_publish_rate': 1.0,
        }]
    )
    
    # ORB-SLAM3 Node (placeholder - replace with actual ORB-SLAM3 ROS wrapper)
    # Note: ORB-SLAM3 doesn't have an official ROS 2 wrapper.
    # Options:
    # 1. Use orbslam3_ros community package (if available)
    # 2. Use RTAB-Map with visual odometry mode
    # 3. Launch ORB-SLAM3 as subprocess with ROS bridge
    
    # For now, we'll include a placeholder that can be replaced
    slam_node = Node(
        package='amr_pose_estimator',
        executable='pnp_node',  # Placeholder - replace with ORB-SLAM3
        name='orb_slam3_node',
        output='screen',
        condition=LaunchConfiguration('use_slam'),
        # In production, this would be:
        # package='orbslam3_ros',
        # executable='orb_slam3_mono',
    )
    
    # RViz visualization (optional)
    rviz_config = os.path.join(pkg_share, 'config', 'slam_visualization.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=LaunchConfiguration('use_rviz'),
        output='screen'
    )
    
    return LaunchDescription([
        # Launch arguments
        camera_matrix_arg,
        camera_height_arg,
        use_slam_arg,
        use_rviz_arg,
        
        # Nodes
        static_tf_camera,
        pnp_node,
        slam_tf_node,
        tracking_monitor_node,
        slam_node,
        rviz_node,
    ])
