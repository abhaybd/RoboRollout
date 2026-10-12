import time
from typing import Dict

import hydra
from hydra.core.config_store import ConfigStore
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType
import numpy as np

from config import YamConfig
from robo_rollout.robot import BaseRobot
from robo_rollout.rollout import TimingBudget, check_config, run
from robo_rollout.utils.misc import multi_closing


SIDES = ("left", "right")


class YamBimanualRobot(BaseRobot):
    """
    Bimanual i2rt YAM, following the DROID conventions for each arm.

    Robot state:
        qpos/left_arm, qpos/right_arm: (6,) joint angles in radians
        qpos/left_gripper, qpos/right_gripper: (1,) in [0, 1], where 0 is open and 1 is closed

    Policy actions:
        left_arm, right_arm: (6,) absolute joint position targets in radians,
            limited to max_joint_delta of motion per step
        left_gripper, right_gripper: scalar or (1,) in [0, 1], where 0 is open and 1 is closed
    """

    def __init__(self, cfg: YamConfig):
        self.cfg = cfg
        channels = {"left": cfg.robot.left_channel, "right": cfg.robot.right_channel}
        self.arms = {}
        try:
            for side in SIDES:
                print(f"Connecting to {side} arm on {channels[side]}...")
                # zero_gravity_mode=False holds the current position instead of going limp
                self.arms[side] = get_yam_robot(
                    channel=channels[side],
                    arm_type=ArmType.from_string_name(cfg.robot.arm_type),
                    gripper_type=GripperType.from_string_name(cfg.robot.gripper_type),
                    zero_gravity_mode=False,
                )
        except:
            self.close()
            raise

    def _move_to(self, targets: Dict[str, np.ndarray], duration: float):
        """Linearly interpolate both arms to the target joint positions (in i2rt's command space) at once."""
        starts = {side: arm.get_joint_pos().copy() for side, arm in self.arms.items()}
        n_steps = max(1, round(duration / 0.02))
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            for side, arm in self.arms.items():
                arm.command_joint_pos((1 - alpha) * starts[side] + alpha * targets[side])
            time.sleep(duration / n_steps)

    def reset(self):
        # always go home, since the arms go limp on exit and must be resting safely when they do
        print("Going home and opening grippers...")
        # i2rt normalizes the gripper so that 1 is open
        homes = {
            "left": np.append(self.cfg.autohome.left_home_jointpos, 1.0),
            "right": np.append(self.cfg.autohome.right_home_jointpos, 1.0),
        }
        self._move_to(homes, self.cfg.autohome.duration)

    def get_state(self):
        qpos = {}
        for side, arm in self.arms.items():
            arm_obs = arm.get_observations()
            qpos[f"{side}_arm"] = np.array(arm_obs["joint_pos"])
            qpos[f"{side}_gripper"] = 1.0 - np.array(arm_obs["gripper_pos"])
        return {"qpos": qpos}

    def apply_action(self, obs, action):
        for side, arm in self.arms.items():
            curr_jp = obs["qpos"][f"{side}_arm"]
            joint_delta = (np.asarray(action[f"{side}_arm"]) - curr_jp) * self.cfg.action_scale
            # limit the joint delta by rescaling it
            max_delta = np.abs(joint_delta).max()
            if max_delta > self.cfg.max_joint_delta:
                joint_delta = joint_delta * (self.cfg.max_joint_delta / max_delta)
            cmd_jp = curr_jp + joint_delta
            cmd_grip_pos = float(np.clip(np.asarray(action[f"{side}_gripper"]).item(), 0.0, 1.0))
            action[f"{side}_arm"] = cmd_jp
            action[f"{side}_gripper"] = np.array([cmd_grip_pos])

            arm.command_joint_pos(np.append(cmd_jp, 1.0 - cmd_grip_pos))
        return action

    def close(self):
        # closing stops commanding the motors, which then go limp
        print("Closing arms...")
        with multi_closing(*self.arms.values()):
            pass


ConfigStore.instance().store(name="config", node=YamConfig)


@hydra.main(version_base=None, config_name="config")
def main(cfg: YamConfig):
    check_config(cfg)
    for side in SIDES:
        home_jointpos = cfg.autohome[f"{side}_home_jointpos"]
        assert len(home_jointpos) == 6, f"autohome.{side}_home_jointpos must have 6 joints, got {len(home_jointpos)}"
    run(cfg, YamBimanualRobot(cfg), TimingBudget(obs=0.01, act=0.01))


if __name__ == "__main__":
    main()
