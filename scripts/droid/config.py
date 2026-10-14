from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omegaconf import MISSING

from robo_rollout.cameras.config import CameraConfig, ZedCameraConfig
from robo_rollout.config import ImageInputConfig, ResizeMethod, RobotConfig, RolloutConfig, WandbConfig


@dataclass
class GripperConfig:
    speed: float = 0.05
    force: float = 0.1


@dataclass
class AutohomeConfig:
    enabled: bool = True
    duration: float = 5.0
    # DROID default home position
    home_jointpos: List[float] = field(default_factory=lambda: [0.0, -0.628319, 0.0, -2.513274, 0.0, 1.884956, 0.0])


@dataclass
class DroidRobotConfig(RobotConfig):
    robot_host: str = MISSING
    gripper: GripperConfig = field(default_factory=GripperConfig)
    cameras: Dict[str, CameraConfig] = field(default_factory=lambda: {
        "wrist_camera": ZedCameraConfig(id=MISSING),
        "exo_camera_1": ZedCameraConfig(id=MISSING),
        "exo_camera_2": ZedCameraConfig(id=MISSING),
    })


@dataclass
class DroidConfig(RolloutConfig):
    robot: DroidRobotConfig = field(default_factory=DroidRobotConfig)

    policy_dt: float = 0.066
    # maps policy observation keys to names of cameras in robot.cameras
    # here, the policy takes in only the wrist camera and the first exo camera
    policy_cameras: Dict[str, str] = field(default_factory=lambda: {
        "exo_camera_1": "exo_camera_1",
        "wrist_camera": "wrist_camera",
    })
    nominal_timeout: Optional[float] = 45.0

    image_input: ImageInputConfig = field(default_factory=lambda: ImageInputConfig(
        width=640,
        height=360,
        method=ResizeMethod.downsample,
    ))
    wandb: WandbConfig = field(default_factory=lambda: WandbConfig(
        # set wandb.enabled=true, wandb.entity, and wandb.project to log episodes to W&B
        enabled=False,
        name="franka_eval",
    ))

    action_scale: float = 1.0
    max_joint_delta: float = 0.2
    autohome: AutohomeConfig = field(default_factory=AutohomeConfig)
