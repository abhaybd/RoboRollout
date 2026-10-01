"""
Camera config schemas. Kept separate from the camera implementations so they can be used without
installing camera SDKs. Field names match the corresponding camera constructor arguments.
"""

from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import MISSING


@dataclass
class CameraConfig:
    type: str = MISSING
    id: Any = MISSING
    width: int = MISSING
    height: int = MISSING
    frame_rate: int = 30


@dataclass
class ZedCameraConfig(CameraConfig):
    type: str = "zed"
    id: int = MISSING  # serial number
    width: int = 1280
    height: int = 720
    side: str = "left"
    depth: bool = False


@dataclass
class RealSenseCameraConfig(CameraConfig):
    type: str = "realsense"
    id: str = MISSING  # serial number
    width: int = 1280
    height: int = 720
    buffer_size: int = 10


@dataclass
class GoProCameraConfig(CameraConfig):
    type: str = "gopro"
    id: str = MISSING  # elgato device id
    width: int = 640
    height: int = 480
    buffer_size: int = 7


@dataclass
class V4LCameraConfig(CameraConfig):
    type: str = "v4l"
    id: Any = MISSING  # device number N (opened as /dev/videoN) or full device path such as "/dev/video0"
    width: int = 1280
    height: int = 720
    # if the camera is a concatenated stereo camera, the side to capture ("left" or "right")
    # if not provided, the full frame will be captured
    stereo_side: Optional[str] = None
