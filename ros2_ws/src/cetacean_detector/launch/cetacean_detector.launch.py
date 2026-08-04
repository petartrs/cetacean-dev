"""Launch the cetacean detector node with a params file.

Usage:
  ros2 launch cetacean_detector cetacean_detector.launch.py \
      params:=/path/to/config/params_adaptive.yaml

The detector then listens on its image/telemetry topics; publish frames with a
live camera driver or the companion demo_source.launch.py.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_root = os.environ.get("CETACEAN_ROOT", "/home/dell/cetacean-detection-final")
    params = LaunchConfiguration("params")
    cetacean_root = LaunchConfiguration("cetacean_root")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params",
            description="Path to a detector params YAML (params_fixed / "
                        "params_adaptive / params_range)."),
        DeclareLaunchArgument(
            "cetacean_root", default_value=default_root,
            description="Repo root so the shared `cetacean` package is importable."),
        SetEnvironmentVariable("CETACEAN_ROOT", cetacean_root),
        Node(
            package="cetacean_detector",
            executable="detector_node",
            name="cetacean_detector",
            output="screen",
            parameters=[params],
        ),
    ])
