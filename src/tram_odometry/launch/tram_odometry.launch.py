import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory("tram_odometry"), "config", "params.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params, description="node parameters (YAML)"),
        DeclareLaunchArgument("gnss_init_s", default_value="5.0", description="GNSS alignment window, s (0 = no GNSS)"),
        Node(
            package="tram_odometry",
            executable="odometry_node",
            name="tram_odometry",
            output="screen",
            parameters=[LaunchConfiguration("params_file"), {"gnss_init_s": LaunchConfiguration("gnss_init_s")}],
        ),
    ])
