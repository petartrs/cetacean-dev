#!/usr/bin/env python3
"""Demo source publisher for the cetacean detector (ROS 2 Humble).

Streams a folder of stills, a single image, or a video file onto the detector's
``image_topic`` at a fixed rate, and (optionally) publishes matching telemetry on
``altitude_topic`` / ``gsd_topic``. This lets the whole stack be exercised
end-to-end with one ``ros2 launch`` and no live camera or rosbag.

Parameters
----------
input          : folder | image file | video file  (required)
image_topic    : where to publish images (default /camera/image_raw)
altitude_topic : std_msgs/Float32 relative altitude [m]  (default /rel_alt)
gsd_topic      : std_msgs/Float32 ground-sample distance [m/px]  (default /gsd)
frame_id       : image header frame_id (default "camera")
rate           : publish rate [Hz] (default 1.0 -- keep low for huge frames)
loop           : repeat the sequence (default False)
altitude_m     : constant altitude to publish (0 = off)
gsd            : constant GSD to publish (0 = off)
use_exif       : read altitude from each image's EXIF/XMP and publish it (default False)
altitude_csv   : CSV "filename,altitude_m" flight log; overrides const when matched
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# Make the shared inference package importable when it is not pip-installed.
try:
    import cetacean  # noqa: F401
except ModuleNotFoundError:
    _root = os.environ.get("CETACEAN_ROOT")
    if _root and _root not in sys.path:
        sys.path.insert(0, _root)

import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from cetacean_detector.img_bridge import bgr_to_imgmsg

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_VID_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


class SourcePublisher(Node):
    def __init__(self) -> None:
        super().__init__("source_publisher")
        d = self.declare_parameter
        d("input", "")
        d("image_topic", "/camera/image_raw")
        d("altitude_topic", "/rel_alt")
        d("gsd_topic", "/gsd")
        d("frame_id", "camera")
        d("rate", 1.0)
        d("loop", False)
        d("altitude_m", 0.0)
        d("gsd", 0.0)
        d("use_exif", False)
        d("altitude_csv", "")

        g = self.get_parameter
        self.input = g("input").value
        self.frame_id = g("frame_id").value
        self.rate = float(g("rate").value)
        self.loop = bool(g("loop").value)
        self.altitude_m = float(g("altitude_m").value)
        self.gsd = float(g("gsd").value)
        self.use_exif = bool(g("use_exif").value)
        self.altitude_csv = g("altitude_csv").value

        if not self.input:
            self.get_logger().error("Parameter 'input' is required.")
            raise SystemExit(1)

        self.img_pub = self.create_publisher(Image, g("image_topic").value, 10)
        self.alt_pub = self.create_publisher(Float32, g("altitude_topic").value, 10)
        self.gsd_pub = self.create_publisher(Float32, g("gsd_topic").value, 10)

        self._alt_map = self._load_altitude_csv(self.altitude_csv)
        self._items = self._enumerate(self.input)   # (name, path_or_None, frame_or_None)
        if not self._items:
            self.get_logger().error(f"no frames found at '{self.input}'")
            raise SystemExit(1)
        self._i = 0
        self.get_logger().info(
            f"source_publisher: {len(self._items)} frame(s) from '{self.input}' "
            f"@ {self.rate} Hz, loop={self.loop}")
        self.timer = self.create_timer(1.0 / max(self.rate, 0.001), self._tick)

    # ---- input enumeration ---------------------------------------------
    def _enumerate(self, path: str):
        p = Path(path)
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in _IMG_EXTS)
            return [(f.name, f, None) for f in files]
        if p.suffix.lower() in _IMG_EXTS:
            return [(p.name, p, None)]
        if p.suffix.lower() in _VID_EXTS:
            cap = cv2.VideoCapture(str(p))
            frames = []
            idx = 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append((f"{p.stem}_{idx:06d}", None, fr))
                idx += 1
            cap.release()
            return frames
        return []

    def _load_altitude_csv(self, path: str) -> dict:
        if not path or not Path(path).exists():
            return {}
        out = {}
        with open(path) as fh:
            for row in csv.DictReader(fh):
                key = row.get("filename") or row.get("frame") or ""
                try:
                    out[Path(key).name] = float(row.get("altitude_m", ""))
                except (TypeError, ValueError):
                    pass
        return out

    # ---- publishing -----------------------------------------------------
    def _tick(self) -> None:
        if self._i >= len(self._items):
            if self.loop:
                self._i = 0
            else:
                self.get_logger().info("source exhausted; idle")
                self.timer.cancel()
                return
        name, path, frame = self._items[self._i]
        self._i += 1

        if frame is None:
            frame = cv2.imread(str(path))
            if frame is None:
                self.get_logger().warn(f"failed to read {path}")
                return

        stamp = self.get_clock().now().to_msg()

        # telemetry first so the detector has the latest value cached
        alt = self._alt_map.get(name)
        if alt is None and self.use_exif and path is not None:
            from cetacean.inference.adaptive_sahi import read_exif_geometry
            alt = read_exif_geometry(path).get("altitude_m")
        if alt is None and self.altitude_m > 0:
            alt = self.altitude_m
        if alt is not None:
            self.alt_pub.publish(Float32(data=float(alt)))
        if self.gsd > 0:
            self.gsd_pub.publish(Float32(data=float(self.gsd)))

        msg = bgr_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        self.img_pub.publish(msg)
        self.get_logger().info(
            f"published {name} ({frame.shape[1]}x{frame.shape[0]})"
            + (f" alt={alt:.1f}m" if alt is not None else ""),
            throttle_duration_sec=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SourcePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
