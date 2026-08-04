from setuptools import find_packages, setup

package_name = "cetacean_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/cetacean_detector.launch.py",
                "launch/demo_source.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/config",
            [
                "config/params_fixed.yaml",
                "config/params_adaptive.yaml",
                "config/params_range.yaml",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SurveyLabs",
    maintainer_email="gerard.dooly@surveylabs.ie",
    description="Aerial cetacean detector (ROS 2 Humble) with fixed / adaptive / "
                "adaptive-range SAHI inference.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detector_node = cetacean_detector.detector_node:main",
            "source_publisher = cetacean_detector.source_publisher_node:main",
        ],
    },
)
