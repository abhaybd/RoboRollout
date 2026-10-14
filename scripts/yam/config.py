from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omegaconf import MISSING

from robo_rollout.cameras.config import CameraConfig, RealSenseCameraConfig
from robo_rollout.config import ImageInputConfig, ResizeMethod, RobotConfig, RolloutConfig, WandbConfig


@dataclass
class AutohomeConfig:
    duration: float = 4.0
    # arm joint positions in radians. The YAM zero position is its folded rest pose, as used by MolmoAct2.
    left_home_jointpos: List[float] = field(default_factory=lambda: [0.0] * 6)
    right_home_jointpos: List[float] = field(default_factory=lambda: [0.0] * 6)


@dataclass
class YamRobotConfig(RobotConfig):
    # i2rt ArmType and GripperType values, e.g. "yam", "yam_pro", "linear_4310", "crank_4310"
    arm_type: str = "yam"
    gripper_type: str = "linear_4310"
    # CAN interfaces of the follower arms, named as in i2rt's bimanual setup
    left_channel: str = "can_follower_l"
    right_channel: str = "can_follower_r"
    cameras: Dict[str, CameraConfig] = field(default_factory=lambda: {
        # MolmoAct2 records 640x360 at 30 fps
        "top_camera": RealSenseCameraConfig(id=MISSING, width=640, height=360),
        "left_wrist_camera": RealSenseCameraConfig(id=MISSING, width=640, height=360),
        "right_wrist_camera": RealSenseCameraConfig(id=MISSING, width=640, height=360),
    })


@dataclass
class YamConfig(RolloutConfig):
    robot: YamRobotConfig = field(default_factory=YamRobotConfig)

    policy_dt: float = 0.03333  # 30 Hz, as in MolmoAct2
    policy_cameras: Dict[str, str] = field(default_factory=lambda: {
        "top_camera": "top_camera",
        "left_wrist_camera": "left_wrist_camera",
        "right_wrist_camera": "right_wrist_camera",
    })
    nominal_timeout: Optional[float] = 45.0

    image_input: ImageInputConfig = field(default_factory=lambda: ImageInputConfig(
        # the full camera resolution, which MolmoAct2 sends to the policy as-is
        width=640,
        height=360,
        method=ResizeMethod.center_pad,
    ))
    wandb: WandbConfig = field(default_factory=lambda: WandbConfig(
        # set wandb.enabled=true, wandb.entity, and wandb.project to log episodes to W&B
        enabled=False,
        name="yam_eval",
    ))

    action_scale: float = 1.0
    # MolmoAct2 instead splits each action into 0.01 rad sub-steps, one per control tick
    max_joint_delta: float = 0.2
    autohome: AutohomeConfig = field(default_factory=AutohomeConfig)
