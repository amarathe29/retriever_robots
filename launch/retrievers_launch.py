from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="retrievers",
                executable="retrieve_node",
                name="retrieve_node",
                namespace="rename_when_launching",
                output="screen",
            ),
            Node(
                package="rosaria2",
                executable="rosaria2_debug",
                name="rosaria2_node",
                namespace="rename_when_launching",
                output="screen",
                remappings=[("pose", "odom")],
                parameters=[
                    {
                        "port": "/dev/ttyUSB0",
                        "frame_id": "base_link",
                        "odom_frame_id": "odom",
                        "tf_prefix": "rename_when_launching",
                    }
                ],
                arguments=['--ros-args', '--log-level', 'warn'],

            ),
            Node(
                package="realsense2_camera",
                executable="realsense2_camera_node",
                name="realsense2_camera_node",
                namespace="rename_when_launching",
                output="screen",
                parameters=[
                    {
                        "enable_pointcloud": False,
                        "enable_sync": True,
                        "align_depth": True,
                        "depth_module.profile": "640x480x30",
                        "rgb_camera.profile": "640x480x30",
                    }
                ],
                arguments=['--ros-args', '--log-level', 'warn'],
            ),
        ]
    )
