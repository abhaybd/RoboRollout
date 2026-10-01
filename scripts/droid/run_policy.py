import hydra
from omegaconf import DictConfig
import numpy as np
from polymetis import RobotInterface, GripperInterface
import torch

from robo_rollout.robot import BaseRobot
from robo_rollout.rollout import check_config, run


MAX_GRIP_WIDTH = 0.085
MAX_MLSPACES_GRIP_WIDTH = 0.824033


class DroidRobot(BaseRobot):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.robot = RobotInterface(
            ip_address=cfg.robot.robot_host,
            enforce_version=False,
        )
        self.gripper = GripperInterface(ip_address=cfg.robot.robot_host)
        self._last_gripper_cmd = self.gripper.get_state().width

    def reset(self):
        print("Opening gripper...")
        self.gripper.goto(MAX_GRIP_WIDTH, self.cfg.robot.gripper.speed, self.cfg.robot.gripper.force)

        if self.cfg.autohome.enabled:
            print("Going home...")
            home_jp = torch.tensor(self.cfg.autohome.home_jointpos)
            self.robot.move_to_joint_positions(home_jp, self.cfg.autohome.duration)

    def start_control(self):
        self._last_gripper_cmd = self.gripper.get_state().width
        self.robot.start_joint_impedance()

    def stop_control(self):
        if self.robot.is_running_policy():
            self.robot.terminate_current_policy(return_log=False)

    def get_state(self):
        jp = self.robot.get_joint_positions().numpy()
        ee_pos, ee_quat = [x.numpy() for x in self.robot.get_ee_pose()]
        grip_width = self.gripper.get_state().width
        # transform gripper to be as expected by molmospaces
        grip_normalized = np.clip(grip_width / MAX_GRIP_WIDTH, 0.0, 1.0)
        gripper_input = np.full(2, (1.0 - grip_normalized) * MAX_MLSPACES_GRIP_WIDTH)
        return {
            "qpos": {
                "arm": jp,
                "gripper": gripper_input,
            },
            "ee_pose": {
                "pos": ee_pos,
                "quat": ee_quat,
            }
        }

    def apply_action(self, obs, action):
        curr_jp = obs["qpos"]["arm"]
        cmd_jp = curr_jp + (action["arm"] - curr_jp) * self.cfg.action_scale
        action["arm"] = cmd_jp
        # convert abs joints to joint vel based on droid conversion
        action["arm_vel"] = (cmd_jp - curr_jp) / 0.2

        # transform gripper from mlspaces convention to droid
        cmd_grip_width = (
            1.0 - np.asarray(action["gripper"]).item() / 255.0
        ) * MAX_GRIP_WIDTH

        self.robot.update_desired_joint_positions(torch.tensor(cmd_jp))

        if not np.isclose(cmd_grip_width, self._last_gripper_cmd, atol=0.002):
            self._last_gripper_cmd = cmd_grip_width
            self.gripper.goto(
                cmd_grip_width, self.cfg.robot.gripper.speed, self.cfg.robot.gripper.force, blocking=False
            )
        return action

    def close(self):
        self.stop_control()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    check_config(cfg)
    # the robot must be reset between trials
    assert (cfg.num_trials is not None and cfg.num_trials <= 1) or cfg.autohome.enabled, \
        "autohome.enabled must be true when running more than one trial"
    run(cfg, DroidRobot(cfg))


if __name__ == "__main__":
    main()
