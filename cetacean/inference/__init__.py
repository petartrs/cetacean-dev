"""Inference subpackage: geometry-aware adaptive SAHI.

Re-exports the public API of :mod:`cetacean.inference.adaptive_sahi` so callers
(ROS node, CLIs, eval scripts) can simply do::

    from cetacean.inference import choose_slice_size, run_adaptive
"""

from .adaptive_sahi import (
    Camera,
    CAMERA_REGISTRY,
    SliceDecision,
    choose_slice_size,
    gsd_from_geometry,
    run_adaptive,
    blind_sweep_pick,
    read_exif_geometry,
    draw,
    _build_sahi_model,
    DEFAULT_WEIGHTS,
    DEFAULT_TARGET_PX,
    DEFAULT_TARGET_LEN_M,
    DEFAULT_IMGSZ,
)

__all__ = [
    "Camera",
    "CAMERA_REGISTRY",
    "SliceDecision",
    "choose_slice_size",
    "gsd_from_geometry",
    "run_adaptive",
    "blind_sweep_pick",
    "read_exif_geometry",
    "draw",
    "_build_sahi_model",
    "DEFAULT_WEIGHTS",
    "DEFAULT_TARGET_PX",
    "DEFAULT_TARGET_LEN_M",
    "DEFAULT_IMGSZ",
]
