"""Minimal sensor_msgs/Image <-> BGR ndarray conversion (no cv_bridge).

cv_bridge on Humble is compiled against NumPy 1.x and crashes when the ML venv
provides NumPy 2.x. These pure-numpy helpers avoid that ABI clash and keep the
package dependency-light for redistribution. Only the encodings this pipeline
actually uses are supported.
"""
from __future__ import annotations

import cv2
import numpy as np
from sensor_msgs.msg import Image

_CH = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1, "8uc1": 1}


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc not in _CH:
        raise ValueError(f"unsupported image encoding: {msg.encoding!r}")
    ch = _CH[enc]
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    arr = buf.reshape(msg.height, msg.step)[:, : msg.width * ch]
    arr = arr.reshape(msg.height, msg.width, ch)
    if enc in ("bgr8",):
        return arr.copy()
    if enc == "rgb8":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)  # mono8 / 8uc1


def bgr_to_imgmsg(img: np.ndarray, encoding: str = "bgr8") -> Image:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected an HxWx3 BGR image")
    data = img if encoding == "bgr8" else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    msg = Image()
    msg.height = int(h)
    msg.width = int(w)
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(w * 3)
    msg.data = np.ascontiguousarray(data, dtype=np.uint8).tobytes()
    return msg
