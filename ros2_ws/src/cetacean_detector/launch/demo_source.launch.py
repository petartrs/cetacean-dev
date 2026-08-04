"""End-to-end demo: stream a folder / image / video into the detector.

Usage:
  ros2 launch cetacean_detector demo_source.launch.py \
      params:=/path/to/config/params_adaptive.yaml \
      input:=/path/to/images_or_video \
      rate:=1.0

Runs the detector and the source publisher together, so the whole pipeline can
be exercised without a live camera. The publisher reads per-image altitude from
EXIF/XMP when present, so adaptive modes get real geometry.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_root = os.environ.get("CETACEAN_ROOT", "/home/dell/cetacean-detection-final")
    params = LaunchConfiguration("params")
    src_input = LaunchConfiguration("input")
    rate = LaunchConfiguration("rate")
    cetacean_root = LaunchConfiguration("cetacean_root")

    return LaunchDescription([
        DeclareLaunchArgument("params", description="Detector params YAML."),
        DeclareLaunchArgument(
            "input", description="Folder of images, a single image, or a video."),
        DeclareLaunchArgument("rate", default_value="1.0", description="Publish rate [Hz]."),
        DeclareLaunchArgument("cetacean_root", default_value=default_root),
        SetEnvironmentVariable("CETACEAN_ROOT", cetacean_root),
        Node(
            package="cetacean_detector",
            executable="detector_node",
            name="cetacean_detector",
            output="screen",
            parameters=[params],
        ),
        Node(
            package="cetacean_detector",
            executable="source_publisher",
            name="source_publisher",
            output="screen",
            parameters=[{"input": src_input, "rate": rate, "use_exif": True}],
        ),
    ])
