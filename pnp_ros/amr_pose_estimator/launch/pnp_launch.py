from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='amr_pose_estimator',
            executable='pnp_node',
            name='pnp_pose_node',
            output='screen',
            parameters=[{
                'camera_matrix': [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
            }]
        )
    ])
