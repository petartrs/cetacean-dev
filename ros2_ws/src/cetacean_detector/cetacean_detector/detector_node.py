#!/usr/bin/env python3
"""Aerial cetacean detector node (ROS 2 Humble).

Ingests images or video (from any ``sensor_msgs/Image`` source -- a live camera
or the companion :mod:`source_publisher_node`) and runs the trained detector
under one of three inference strategies, chosen with the ``inference_mode``
parameter:

* ``fixed``          -- fixed-size SAHI tiling (crop = ``fixed_slice`` px);
  imaging geometry is ignored.
* ``adaptive``       -- the crop is sized from ground-sample distance so a single
  target body length lands at the trained object scale; SAHI is skipped when the
  animal is already large enough.
* ``adaptive_range`` -- the crop is sized so the *largest* expected animal
  (``max_len_m``) still fits one tile, keeping large and small animals resolvable
  across a size range.

Geometry (altitude / GSD) comes from telemetry topics when present, otherwise
from parameters, otherwise a one-off blind sweep. The inference core is the
shared :mod:`cetacean.inference` package -- the exact code validated offline, so
there is no separate deployment path to drift out of sync.

Topics
------
sub : <image_topic>    sensor_msgs/Image    (default /camera/image_raw)
sub : <altitude_topic> std_msgs/Float32     relative altitude [m] (default /rel_alt)
sub : <gsd_topic>      std_msgs/Float32     ground-sample distance [m/px] (optional)
pub : ~/detections     vision_msgs/Detection2DArray
pub : ~/count          std_msgs/Int32
pub : ~/annotated      sensor_msgs/Image    (optional)
"""
from __future__ import annotations

import os
import sys
import threading

# Make the shared inference package importable when it is not pip-installed.
try:
    import cetacean  # noqa: F401
except ModuleNotFoundError:
    _root = os.environ.get("CETACEAN_ROOT")
    if _root and _root not in sys.path:
        sys.path.insert(0, _root)

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from cetacean_detector.img_bridge import imgmsg_to_bgr, bgr_to_imgmsg
from cetacean.inference.adaptive_sahi import (
    SliceDecision,
    choose_slice_size,
    run_adaptive,
    blind_sweep_pick,
    _build_sahi_model,
    draw,
)

_MODES = ("fixed", "adaptive", "adaptive_range")


def _count_tiles(img_w: int, img_h: int, slice_size: int, overlap: float) -> int:
    from sahi.slicing import get_slice_bboxes
    try:
        return len(get_slice_bboxes(
            image_height=img_h, image_width=img_w,
            slice_height=slice_size, slice_width=slice_size,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap))
    except TypeError:  # older SAHI positional signature
        return len(get_slice_bboxes(
            img_h, img_w, slice_size, slice_size, False, overlap, overlap))


class CetaceanDetector(Node):
    def __init__(self) -> None:
        super().__init__("cetacean_detector")

        # ---- parameters -------------------------------------------------
        d = self.declare_parameter
        d("inference_mode", "adaptive")           # fixed | adaptive | adaptive_range
        d("model_path", "")
        d("image_topic", "/camera/image_raw")
        d("altitude_topic", "/rel_alt")
        d("gsd_topic", "/gsd")
        d("class_name", "cetacean")
        d("device", "cuda:0")
        # detector
        d("conf", 0.3)
        d("iou", 0.45)
        d("imgsz", 1024)
        d("overlap", 0.2)
        # fixed mode
        d("fixed_slice", 1024)
        # adaptive scale
        d("target_px", 45.0)
        d("skip_px", 45.0)
        d("min_slice", 512)
        d("max_slice", 0)                         # 0 -> None (long side)
        d("target_len_m", 3.0)
        # range mode
        d("max_len_m", 18.0)
        d("max_fill", 0.8)
        # geometry anchors / overrides (0 / "" -> unset)
        d("camera", "")
        d("focal_mm", 0.0)
        d("pixel_pitch_um", 0.0)
        d("gsd", 0.0)                             # m/px override
        d("altitude_m", 0.0)                      # static fallback [m]
        d("blind_slices", [512, 1024, 2048])
        # output
        d("publish_annotated", True)
        d("report_csv", "")

        g = self.get_parameter
        self.inference_mode = g("inference_mode").value
        if self.inference_mode not in _MODES:
            self.get_logger().warn(
                f"unknown inference_mode '{self.inference_mode}', using 'adaptive'")
            self.inference_mode = "adaptive"
        self.model_path = g("model_path").value
        self.image_topic = g("image_topic").value
        self.altitude_topic = g("altitude_topic").value
        self.gsd_topic = g("gsd_topic").value
        self.class_name = g("class_name").value
        self.device = g("device").value
        self.conf = float(g("conf").value)
        self.iou = float(g("iou").value)
        self.imgsz = int(g("imgsz").value)
        self.overlap = float(g("overlap").value)
        self.fixed_slice = int(g("fixed_slice").value)
        self.target_px = float(g("target_px").value)
        self.skip_px = float(g("skip_px").value)
        self.min_slice = int(g("min_slice").value)
        ms = int(g("max_slice").value)
        self.max_slice = ms if ms > 0 else None
        self.target_len_m = float(g("target_len_m").value)
        self.max_len_m = float(g("max_len_m").value)
        self.max_fill = float(g("max_fill").value)
        self.camera = g("camera").value or None
        self.focal_mm = float(g("focal_mm").value) or None
        self.pixel_pitch_um = float(g("pixel_pitch_um").value) or None
        self.p_gsd = float(g("gsd").value) or None
        self.p_altitude_m = float(g("altitude_m").value) or None
        self.blind_slices = [int(x) for x in g("blind_slices").value]
        self.publish_annotated = bool(g("publish_annotated").value)
        self.report_csv = g("report_csv").value

        if not self.model_path:
            self.get_logger().error("Parameter 'model_path' is required.")
            raise SystemExit(1)

        # ---- state ------------------------------------------------------
        self._lock = threading.Lock()
        self._alt_topic_val: float | None = None
        self._gsd_topic_val: float | None = None
        self._blind_slice: int | None = None
        self._frame_i = 0
        self._report_fh = None

        self._load_models()

        if self.report_csv:
            self._report_fh = open(self.report_csv, "w")
            self._report_fh.write("frame,stamp,mode,slice,tiles,n_det\n")

        # ---- pubs / subs ------------------------------------------------
        self.det_pub = self.create_publisher(Detection2DArray, "~/detections", 10)
        self.count_pub = self.create_publisher(Int32, "~/count", 10)
        self.ann_pub = (
            self.create_publisher(Image, "~/annotated", 5)
            if self.publish_annotated else None
        )
        self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.create_subscription(Float32, self.altitude_topic, self.on_altitude, 10)
        self.create_subscription(Float32, self.gsd_topic, self.on_gsd, 10)

        self.get_logger().info(
            f"cetacean_detector[{self.inference_mode}] ready | model={self.model_path} | "
            f"image={self.image_topic} alt={self.altitude_topic} | "
            f"conf={self.conf} imgsz={self.imgsz} overlap={self.overlap} "
            f"fixed_slice={self.fixed_slice} target_len_m={self.target_len_m} "
            f"max_len_m={self.max_len_m}"
        )

    # ---- model loading --------------------------------------------------
    def _load_models(self) -> None:
        from ultralytics import YOLO
        self.yolo_model = YOLO(self.model_path)
        # SAHI model reused across frames (built once; tiles inferred at imgsz).
        self.sahi_model = _build_sahi_model(
            self.model_path, self.conf, self.imgsz, self.device)

    # ---- telemetry caches ----------------------------------------------
    def on_altitude(self, msg: Float32) -> None:
        with self._lock:
            self._alt_topic_val = float(msg.data)

    def on_gsd(self, msg: Float32) -> None:
        with self._lock:
            self._gsd_topic_val = float(msg.data)

    # ---- geometry -> slice decision ------------------------------------
    def _resolve_decision(self, img_w: int, img_h: int) -> SliceDecision:
        # Fixed mode: one crop size for every frame, geometry ignored.
        if self.inference_mode == "fixed":
            hi = self.max_slice or max(img_w, img_h)
            fs = max(self.min_slice, min(self.fixed_slice, hi))
            return SliceDecision(fs, "sahi", None, None, None, "fixed",
                                 f"fixed slice={fs}")

        with self._lock:
            alt_topic = self._alt_topic_val
            gsd_topic = self._gsd_topic_val
        gsd = self.p_gsd if self.p_gsd is not None else gsd_topic
        altitude = alt_topic if alt_topic is not None else self.p_altitude_m
        max_len_m = self.max_len_m if self.inference_mode == "adaptive_range" else None

        dec = choose_slice_size(
            img_w, img_h,
            gsd=gsd, camera=self.camera, altitude_m=altitude,
            focal_mm=self.focal_mm, pixel_pitch_um=self.pixel_pitch_um,
            target_len_m=self.target_len_m, target_px=self.target_px,
            skip_px=self.skip_px, imgsz=self.imgsz,
            min_slice=self.min_slice, max_slice=self.max_slice,
            max_len_m=max_len_m, max_fill=self.max_fill,
        )

        # Reuse a cached blind slice once one has been calibrated for the flight.
        if dec.mode == "blind_sweep" and self._blind_slice is not None:
            dec = SliceDecision(self._blind_slice, "sahi", None, None, None,
                                "blind_cached", f"cached blind slice {self._blind_slice}")
        return dec

    # ---- inference ------------------------------------------------------
    def _infer(self, frame, decision: SliceDecision):
        return run_adaptive(
            frame, self.model_path, decision,
            conf=self.conf, overlap=self.overlap, imgsz=self.imgsz,
            device=self.device, yolo_model=self.yolo_model, sahi_model=self.sahi_model)

    # ---- main callback --------------------------------------------------
    def on_image(self, msg: Image) -> None:
        frame = imgmsg_to_bgr(msg)
        h, w = frame.shape[:2]
        dec = self._resolve_decision(w, h)

        # Lazy blind-sweep calibration (needs the actual frame).
        if dec.mode == "blind_sweep":
            self._blind_slice, counts = blind_sweep_pick(
                frame, self.model_path, self.blind_slices,
                conf=self.conf, overlap=self.overlap, imgsz=self.imgsz, device=self.device)
            self.get_logger().info(
                f"blind sweep picked slice={self._blind_slice} ({counts})")
            dec = SliceDecision(self._blind_slice, "sahi", None, None, None,
                                "blind_cached", f"blind slice {self._blind_slice}")

        dets = self._infer(frame, dec)
        tiles = 1 if dec.mode == "plain" or not dec.slice_size else _count_tiles(
            w, h, dec.slice_size, self.overlap)

        # publish detections
        out = Detection2DArray()
        out.header = msg.header
        for x1, y1, x2, y2, score in dets:
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = float((x1 + x2) / 2.0)
            det.bbox.center.position.y = float((y1 + y2) / 2.0)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self.class_name
            hyp.hypothesis.score = float(score)
            det.results.append(hyp)
            out.detections.append(det)
        self.det_pub.publish(out)
        self.count_pub.publish(Int32(data=len(dets)))

        if self.ann_pub is not None:
            ann = draw(frame.copy(), dets)
            ann_msg = bgr_to_imgmsg(ann, encoding="bgr8")
            ann_msg.header = msg.header
            self.ann_pub.publish(ann_msg)

        self._frame_i += 1
        self.get_logger().info(
            f"[{self._frame_i}] {self.inference_mode}/{dec.mode} slice={dec.slice_size} "
            f"tiles={tiles} dets={len(dets)} | {dec.reason}",
            throttle_duration_sec=2.0,
        )
        if self._report_fh is not None:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._report_fh.write(
                f"{self._frame_i},{stamp:.3f},{dec.mode},{dec.slice_size or 0},{tiles},{len(dets)}\n")
            self._report_fh.flush()

    def destroy_node(self) -> bool:
        if self._report_fh is not None:
            self._report_fh.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CetaceanDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
