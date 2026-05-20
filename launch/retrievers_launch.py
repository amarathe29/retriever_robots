from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

import os

included_package_dir = get_package_share_directory('realsense2_camera')
    
# 2. Construct the full path to the target launch file
launch_file_path = os.path.join(included_package_dir, 'launch', 'rs_launch.py')


ns = DeclareLaunchArgument(
                'namespace',
                default_value='rename_when_launching',
                description='Namespace for the node'
            ),
namespace = LaunchConfiguration('namespace')

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
                        "port": "/dev/ttyUSB0",
                        "frame_id": "base_link",
                        "odom_frame_id": "odom",
                        "tf_prefix": "rename_when_launching",
                    }
                ],
                arguments=["--ros-args", "--log-level", "warn"],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(launch_file_path),
                launch_arguments={
                    'initial_reset': 'true', 
                    'camera_namespace': namespace,
                    'depth_module.depth_profile': '640x480x30',
                    'rgb_camera.color_profile': '640x480x30',
                    'enable_sync': 'true',
                    'align_depth': 'true',
                    }.items(),

            ),
            # Node(
            #     package="realsense2_camera",
            #     executable="realsense2_camera_node",
            #     name="realsense2_camera_node",
            #     namespace="rename_when_launching",
            #     output="screen",
            #     parameters=[
            #         {
            #             "enable_pointcloud": False,
            #             "enable_sync": True,
            #             "align_depth": True,
            #             "depth_module.profile": "640x480x30",
            #             "rgb_camera.profile": "640x480x30",
            #         }
            #     ],
            #     arguments=["--ros-args", "--log-level", "warn"],
            # ),

            # ros2 action send_goal /rename_when_launching/gotoblock retriever_msgs/action/GoToBlock   "{goal_pose: {position: {x: 1.3, y: -0.2, z: 0.0}, orientation:{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, block_type: 1}"
        ]
    )
