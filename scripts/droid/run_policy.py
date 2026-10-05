import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from polymetis import RobotInterface, GripperInterface
import torch

from config import DroidConfig
from robo_rollout.robot import BaseRobot
from robo_rollout.rollout import TimingBudget, check_config, run


class DroidRobot(BaseRobot):
    """
    Franka robot following the DROID conventions.

    Robot state:
        qpos/arm: (7,) joint angles in radians
        qpos/gripper: (1,) in [0, 1], where 0 is open and 1 is closed
        ee_pose/pos: (3,) end-effector position
        ee_pose/quat: (4,) end-effector orientation as an xyzw quaternion

    Policy actions (DROID's joint_position action space, with gripper position):
        arm: (7,) absolute joint position targets in radians, limited to max_joint_delta of motion per step
        gripper: scalar or (1,) in [0, 1], where 0 is open and 1 is closed
    """

    def __init__(self, cfg: DroidConfig):
        self.cfg = cfg
        self.robot = RobotInterface(
            ip_address=cfg.robot.robot_host,
            enforce_version=False,
        )
        self.gripper = GripperInterface(ip_address=cfg.robot.robot_host)
        self.max_gripper_width = self.gripper.metadata.max_width
        self._last_gripper_cmd = self.gripper.get_state().width

    def reset(self):
        print("Opening gripper...")
        self.gripper.goto(self.max_gripper_width, self.cfg.robot.gripper.speed, self.cfg.robot.gripper.force)

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
        return {
            "qpos": {
                "arm": jp,
                "gripper": np.array([1.0 - grip_width / self.max_gripper_width]),
            },
            "ee_pose": {
                "pos": ee_pos,
                "quat": ee_quat,
            }
        }

    def apply_action(self, obs, action):
        curr_jp = obs["qpos"]["arm"]
        joint_delta = (np.asarray(action["arm"]) - curr_jp) * self.cfg.action_scale
        # limit the joint delta by rescaling it
        max_delta = np.abs(joint_delta).max()
        if max_delta > self.cfg.max_joint_delta:
            joint_delta = joint_delta * (self.cfg.max_joint_delta / max_delta)
        cmd_jp = curr_jp + joint_delta
        action["arm"] = cmd_jp
        # according to DROID, "velocity" is just normalized joint delta
        action["arm_vel"] = joint_delta / self.cfg.max_joint_delta

        cmd_grip_pos = float(np.clip(np.asarray(action["gripper"]).item(), 0.0, 1.0))
        cmd_grip_width = (1.0 - cmd_grip_pos) * self.max_gripper_width

        self.robot.update_desired_joint_positions(torch.tensor(cmd_jp))

        if not np.isclose(cmd_grip_width, self._last_gripper_cmd, atol=0.002):
            self._last_gripper_cmd = cmd_grip_width
            self.gripper.goto(
                cmd_grip_width, self.cfg.robot.gripper.speed, self.cfg.robot.gripper.force, blocking=False
            )
        return action

    def close(self):
        self.stop_control()


ConfigStore.instance().store(name="config", node=DroidConfig)


@hydra.main(version_base=None, config_name="config")
def main(cfg: DroidConfig):
    check_config(cfg)
    # the robot must be reset between trials
    assert (cfg.num_trials is not None and cfg.num_trials <= 1) or cfg.autohome.enabled, \
        "autohome.enabled must be true when running more than one trial"
    run(cfg, DroidRobot(cfg), TimingBudget(obs=0.01, act=0.01))


if __name__ == "__main__":
    main()
