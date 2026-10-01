from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omegaconf import MISSING

from robo_rollout.cameras.config import CameraConfig, ZedCameraConfig
from robo_rollout.config import ImageInputConfig, ResizeMethod, RobotConfig, RolloutConfig, WandbConfig


@dataclass
class GripperConfig:
    speed: float = 0.085
    force: float = 10.0


@dataclass
class AutohomeConfig:
    enabled: bool = True
    duration: float = 5.0
    home_jointpos: List[float] = field(default_factory=lambda: [0.0, -0.7, 0.0, -2.35619, 0.0, 1.57079, 0.0])


@dataclass
class DroidRobotConfig(RobotConfig):
    robot_host: str = "172.16.0.2"
    gripper: GripperConfig = field(default_factory=GripperConfig)
    cameras: Dict[str, CameraConfig] = field(default_factory=lambda: {
        "wrist_camera": ZedCameraConfig(id=14846828),
        "exo_camera_1": ZedCameraConfig(id=31425515),
        "exo_camera_2": ZedCameraConfig(id=32439448),
    })


@dataclass
class DroidConfig(RolloutConfig):
    robot: DroidRobotConfig = field(default_factory=DroidRobotConfig)

    policy_dt: float = 0.066
    policy_cameras: Dict[str, str] = field(default_factory=lambda: {
        "exo_camera_1": MISSING,
        "wrist_camera": "wrist_camera",
    })
    nominal_timeout: Optional[float] = 45.0

    image_input: ImageInputConfig = field(default_factory=lambda: ImageInputConfig(
        width=640,
        height=360,
        method=ResizeMethod.downsample,
    ))
    wandb: WandbConfig = field(default_factory=lambda: WandbConfig(
        enabled=True,
        entity="adeshpande-princeton-university",
        project="synthetic-wm-evals",
        name="franka_eval",
    ))

    action_scale: float = 1.0
    autohome: AutohomeConfig = field(default_factory=AutohomeConfig)
