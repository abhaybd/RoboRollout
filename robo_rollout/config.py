"""
Config schema for policy rollouts. Robots extend RolloutConfig (and RobotConfig) with their own fields,
and register the result with hydra's ConfigStore. See scripts/droid/config.py for an example.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from omegaconf import MISSING

from robo_rollout.cameras.config import CameraConfig


class ResizeMethod(Enum):
    center_pad = "center_pad"
    center_crop = "center_crop"
    downsample = "downsample"
    none = "none"


@dataclass
class ImageInputConfig:
    """Size and resize method for camera images sent to the policy."""
    width: int = MISSING
    height: int = MISSING
    method: ResizeMethod = ResizeMethod.center_pad


@dataclass
class WandbConfig:
    enabled: bool = False
    entity: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None  # defaults to the task if not set


@dataclass
class RobotConfig:
    cameras: Dict[str, CameraConfig] = field(default_factory=dict)


@dataclass
class RolloutConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)

    policy_host: str = "0.0.0.0"
    policy_port: Optional[int] = 8000
    policy_dt: float = MISSING
    # maps policy observation keys to names of cameras in robot.cameras
    policy_cameras: Dict[str, str] = field(default_factory=dict)
    nominal_timeout: Optional[float] = None  # seconds, null for no timeout
    num_trials: Optional[int] = None  # null to run until ctrl+c
    task: str = MISSING

    image_input: ImageInputConfig = field(default_factory=ImageInputConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
