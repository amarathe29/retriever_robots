from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

import os

included_package_dir = get_package_share_directory("realsense2_camera")

# 2. Construct the full path to the target launch file
launch_file_path = os.path.join(included_package_dir, "launch", "rs_launch.py")


ns = DeclareLaunchArgument(
    "namespace",
    default_value="rename_when_launching",
    description="Namespace for the node",
)
namespace = LaunchConfiguration("namespace")


def generate_launch_description():
    return LaunchDescription(
        [
            ns,
            Node(
                package="retriever_robots",
                executable="retriever_node",
                name="retriever_node",
                namespace=namespace,
                output="screen",
                arguments=["--ros-args", "--log-level", "info"],
            ),
            Node(
                package="retriever_robots",
                executable="detect_block",
                name="detect_block",
                namespace=namespace,
                output="screen",
                arguments=["--ros-args", "--log-level", "info"],
            ),
            Node(
                package="rosaria2",
                executable="rosaria2_debug",
                name="rosaria2_node",
                namespace=namespace,
                output="screen",
                remappings=[("pose", "odom")],
                parameters=[
                    {
                        "serial_port": "/dev/ttyUSB0",
                        "frame_id_prefix": namespace,
                        "frame_id_odom": "/odom",
                        "frame_id_base_link": "/base_link",
                        "frame_id_bumper": "/bumper",
                        "frame_id_sonar": "/sonar",
                    }
                ],
                arguments=["--ros-args", "--log-level", "warn"],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(launch_file_path),
                launch_arguments={
                    "initial_reset": "true",
                    "camera_namespace": namespace,
                    "depth_module.depth_profile": "640x360x30",
                    "rgb_camera.color_profile": "640x360x30",
                    'publish_tf': 'false',
                }.items(),
            ),
            # ros2 action send_goal /rename_when_launching/gotoblock retriever_msgs/action/GoToBlock   "{goal_pose: {position: {x: 1.3, y: -0.2, z: 0.0}, orientation:{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, block_type: 1}"
        ]
    )
