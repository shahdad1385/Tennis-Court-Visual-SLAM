from setuptools import find_packages, setup

package_name = 'amr_pose_estimator'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pnp_launch.py', 'launch/slam_launch.py']),
        ('share/' + package_name + '/config', ['config/orb_slam3_params.yaml', 'config/slam_config.yaml', 'config/ekf.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='AMR Pose Estimation using PnP',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pnp_node = amr_pose_estimator.pnp_pose_node:main',
            'slam_tf_broadcaster = amr_pose_estimator.slam_tf_broadcaster:main',
            'tracking_monitor = amr_pose_estimator.tracking_monitor:main',
            'ball_throwing_smach = amr_pose_estimator.smach_coordinator:main',
        ],
    },
)
