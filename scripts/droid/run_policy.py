import json
from pathlib import Path
import time
from contextlib import contextmanager, suppress
from typing import Dict, List, Optional
import logging
import traceback

import hydra
from ffmpegcv import VideoWriter
from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np
from polymetis import RobotInterface, GripperInterface
import torch
import wandb

from robo_rollout.cameras.abstract import BaseCamera
from robo_rollout.policy.policy_client import PolicyClient
from robo_rollout.cameras.camera import create_camera
from robo_rollout.utils.image import resize_with_crop, resize_with_pad, downsample
from robo_rollout.utils.threading import AsyncMultiVideoWriter, ThreadedPeriodicGetter
from robo_rollout.utils.misc import lod_to_dol, measure_elapsed, multi_closing, suppress_decorator


logging.basicConfig(level=logging.INFO)
np.set_printoptions(linewidth=150)

MAX_GRIP_WIDTH = 0.085
MAX_MLSPACES_GRIP_WIDTH = 0.824033

TIMING_BUDGETS = {
    "obs": 0.01,
    "act": 0.01,
}


def get_timing_budget(config: DictConfig, name: str) -> float:
    """
    Get the timing budget for a given step.
    If the step budget isn't specified in TIMING_BUDGETS,
    use the remaining time after subtracting the other budgets from the total policy_dt.
    """
    if name in TIMING_BUDGETS:
        return TIMING_BUDGETS[name]
    return max(0.0, config.policy_dt - sum(TIMING_BUDGETS.values()))


@contextmanager
def cleanup_cameras(cam_dict: Dict[str, BaseCamera]):
    try:
        yield
    finally:
        try:
            print("Closing cameras...")
            for cam_name, camera in cam_dict.items():
                print(f"\tClosing camera {cam_name}...")
                camera.close()
        except KeyboardInterrupt:
            pass


def prompt_success():
    try:
        while True:
            ret = input("Did the policy succeed? (y/n): ").lower()
            if ret and ret[0] == "y":
                return True
            elif ret and ret[0] == "n":
                return False
            else:
                print("Invalid input.")
    except KeyboardInterrupt:
        print("Received interrupt. Assuming failure.")
        return False


def get_robot_state(robot: RobotInterface, gripper: GripperInterface):
    jp = robot.get_joint_positions().numpy()
    ee_pos, ee_quat = [x.numpy() for x in robot.get_ee_pose()]
    grip_width = gripper.get_state().width
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


def create_writer(cfg, output_dir, cam_name):
    return VideoWriter(output_dir / f"{cam_name}.mp4", fps=round(1.0 / cfg.policy_dt))


class DroidJointImpedance:
    def __init__(self, robot: RobotInterface):
        self.robot = robot
        self.robot.start_joint_impedance()

    @suppress_decorator(KeyboardInterrupt)
    def close(self):
        if self.robot.is_running_policy():
            self.robot.terminate_current_policy(return_log=False)


class DroidRollout:
    def __init__(self, cfg: DictConfig, output_dir: Path, cameras: Dict[str, BaseCamera]):
        self.cfg = cfg
        self.cameras = cameras
        self.observations = []
        self.actions = []

        self.rt_video_writer = AsyncMultiVideoWriter({
            cam_name: create_writer(self.cfg, output_dir, f"{cam_name}_rt") for cam_name in self.cameras
        })
        self.video_writer = AsyncMultiVideoWriter({
            cam_name: create_writer(self.cfg, output_dir, cam_name) for cam_name in self.cameras
        })
        self.rt_capture_task = ThreadedPeriodicGetter(self._capture_rt_frames, self.cfg.policy_dt)

    def _capture_rt_frames(self):
        for cam_name, camera in self.cameras.items():
            frame = camera.get_frame()
            self.rt_video_writer.write(cam_name, frame)

    def start_rt_capture(self):
        self.rt_capture_task.start()

    def close(self):
        self.rt_capture_task.stop()
        self.rt_video_writer.close()
        self.video_writer.close()



class DroidRolloutRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

        self._reset_policy()
        self.robot = RobotInterface(
            ip_address=cfg.robot.robot_host,
            enforce_version=False,
        )
        self.gripper = GripperInterface(ip_address=cfg.robot.robot_host)

        self.resize_func = {
            "center_pad": resize_with_pad,
            "center_crop": resize_with_crop,
            "downsample": downsample,
            "none": lambda x, h, w: x,
        }[cfg.image_input.method]

        print("Opening cameras...")
        self.cameras: dict[str, BaseCamera] = {}
        for cam_name, camera_cfg in cfg.robot.cameras.items():
            try:
                camera = create_camera(camera_cfg)
            except:
                for cam in self.cameras.values():
                    cam.close()
                raise
            self.cameras[cam_name] = camera

    def _reset_policy(self):
        if self.policy_client is not None:
            self.policy_client.close()
        self.policy_client = PolicyClient(host=self.cfg.policy_host, port=self.cfg.policy_port)
        self.metadata = self.policy_client.get_server_metadata()
        print("Server metadata:")
        print(json.dumps(self.metadata, indent=2))

    def reset_robot(self):
        print("Opening gripper...")
        self.gripper.goto(MAX_GRIP_WIDTH, self.cfg.robot.gripper.speed, self.cfg.robot.gripper.force)

        if self.cfg.autohome.enabled:
            print("Going home...")
            home_jp = torch.tensor(self.cfg.autohome.home_jointpos)
            self.robot.move_to_joint_positions(home_jp, self.cfg.autohome.duration)

    def get_obs(self, observations: List, video_writer: AsyncMultiVideoWriter):
        obs = {
            "task": self.cfg.task,
        }

        with measure_elapsed() as elapsed_robot:
            robot_state = get_robot_state(self.robot, self.gripper)
            observations.append(robot_state)
            obs.update(robot_state)

        with measure_elapsed() as elapsed_cam:
            curr_camera_frames = {}
            for cam_name, camera in self.cameras.items():
                frame = camera.get_frame()
                video_writer.write(cam_name, frame)
                frame_resized = self.resize_func(frame, self.cfg.image_input.height, self.cfg.image_input.width)

                frame_std = np.std(np.mean(frame, axis=-1)).item()
                # Sometimes cameras fail silently and return single-color frames
                if frame_std < 1.0:
                    if self.cfg.wandb.enabled and "camera_failure" not in wandb.run.tags:
                        wandb.run.tags = wandb.run.tags + ("camera_failure",)
                    raise ValueError(
                        "Camera failure detected, constant color frames"
                    )

                curr_camera_frames[cam_name] = frame_resized

            for obs_cam_name, cam_name in self.cfg.policy_cameras.items():
                obs[obs_cam_name] = curr_camera_frames[cam_name]

        obs_timing = {
            "elapsed_cam": elapsed_cam.value,
            "elapsed_robot": elapsed_robot.value,
        }
        return obs, obs_timing

    def _capture_frames(self, rt_video_writer: AsyncMultiVideoWriter):
        for cam_name, camera in self.cameras.items():
            frame = camera.get_frame()
            rt_video_writer.write(cam_name, frame)

    @suppress_decorator(KeyboardInterrupt)
    def _rollout_episode(self, rollout: DroidRollout):
        cfg = self.cfg
        rollout.start_rt_capture()

        last_gripper_cmd = MAX_GRIP_WIDTH
        action_scale: float = cfg.action_scale
        sleep_until = time.monotonic()
        nominal_timeout_steps = int(np.ceil(cfg.nominal_timeout / cfg.policy_dt)) if cfg.nominal_timeout else None
        n_steps = 0
        while nominal_timeout_steps is None or n_steps < nominal_timeout_steps:
            with measure_elapsed() as elapsed_obs:
                obs, obs_timing = self.get_obs(rollout.observations, rollout.video_writer)
            n_steps += 1

            with measure_elapsed() as elapsed_infer:
                action_dict = self.policy_client.infer(obs)

            with measure_elapsed() as elapsed_act:
                cmd_jp_np = action_dict["arm"]
                cmd_jp_np = (
                    obs["qpos"]["arm"]
                    + (cmd_jp_np - obs["qpos"]["arm"]) * action_scale
                )
                action_dict["arm"] = cmd_jp_np
                action_dict["arm_vel"] = (cmd_jp_np - obs["qpos"]["arm"]) / 0.2
                action_dict["arm_vel"] /= np.maximum(np.max(np.abs(action_dict["arm_vel"])), 1.0)
                rollout.actions.append(action_dict)

                cmd_jp = torch.tensor(cmd_jp_np)
                cmd_grip_width = (
                    1.0 - action_dict["gripper"].item() / 255.0
                ) * MAX_GRIP_WIDTH

                self.robot.update_desired_joint_positions(cmd_jp)

                if not np.isclose(cmd_grip_width, last_gripper_cmd, atol=0.002):
                    last_gripper_cmd = cmd_grip_width
                    self.gripper.goto(
                        cmd_grip_width, cfg.robot.gripper.speed, cfg.robot.gripper.force, blocking=False
                    )

            sleep_until += cfg.policy_dt
            curr_time = time.monotonic()
            if curr_time < sleep_until:
                time.sleep(sleep_until - curr_time)
            else:
                print(f"WARN: Loop overrun by {curr_time - sleep_until:.3f}s")
                print("Budget violations:")
                if elapsed_obs.value > get_timing_budget(cfg, "obs"):
                    print(f"  Observation: {elapsed_obs.value:.3f}s")
                    for k, v in obs_timing.items():
                        name = k.split("_", 1)[-1].title()
                        print(f"    {name}: {v:.3f}s")
                if elapsed_infer.value > get_timing_budget(cfg, "infer"):
                    print(f"  Inference: {elapsed_infer.value:.3f}s")
                if elapsed_act.value > get_timing_budget(cfg, "act"):
                    print(f"  Action: {elapsed_act.value:.3f}s")

                while sleep_until < curr_time:
                    sleep_until += cfg.policy_dt
                sleep_until -= cfg.policy_dt

        return n_steps

    def rollout_episode(self, task: Optional[str]):
        if task:
            with open_dict(self.cfg):
                self.cfg.task = task
        cfg = self.cfg
        self._reset_policy()

        if cfg.wandb.enabled:
            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb_cfg["policy_metadata"] = self.metadata
            wandb_run = wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.name,
                dir=str(self.output_dir),
                tags=cfg.wandb.tags,
                config=wandb_cfg,
                notes=cfg.wandb.notes or cfg.task,
            )
        else:
            wandb_run = None

        try:
            n_steps = 0
            joint_impedance = DroidJointImpedance(self.robot)
            rollout = DroidRollout(self.cfg, self.output_dir, self.cameras)
            start = time.time()
            with multi_closing(rollout, joint_impedance):
                try:
                    n_steps = self._rollout_episode(rollout)
                except Exception:
                    traceback.print_exc()

            success = prompt_success()

            episode_length = time.time() - start

            info = {
                "episode_length": episode_length,
                "episode_length_nominal": n_steps * cfg.policy_dt,
                "episode_length_steps": n_steps,
                "success": success,
                "success_rate": float(success),
                "task": cfg.task,
            }
            (self.output_dir / "info.json").write_text(json.dumps(info, indent=2))

            print("Saving observations and actions")
            observations_dol = lod_to_dol(rollout.observations)
            actions_dol = lod_to_dol(rollout.actions)
            obs_path = self.output_dir / "observations.npz"
            act_path = self.output_dir / "actions.npz"
            np.savez_compressed(obs_path, **observations_dol)
            np.savez_compressed(act_path, **actions_dol)

            if wandb_run is not None:
                video_dict = {}
                for cam_name in self.cameras:
                    video_dict[f"video/{cam_name}"] = wandb.Video(
                        self.output_dir / f"{cam_name}.mp4",
                        caption=cam_name,
                        format="mp4",
                    )
                    video_dict[f"video/{cam_name}_rt"] = wandb.Video(
                        self.output_dir / f"{cam_name}_rt.mp4",
                        caption=cam_name,
                        format="mp4",
                    )

                wandb_run.summary.update({**info, **video_dict})

                wandb_run.save(obs_path, base_path=obs_path.parent)
                wandb_run.save(act_path, base_path=act_path.parent)
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    def close(self):
        if self.robot.is_running_policy():
            self.robot.terminate_current_policy(return_log=False)
        self.policy_client.close()
        for cam_name, cam in self.cameras.items():
            print(f"Closing camera {cam_name}...")
            cam.close()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    if missing_keys := OmegaConf.missing_keys(cfg):
        raise ValueError(f"Missing keys: {missing_keys}")

    print("Evaluation Configuration:")
    print(OmegaConf.to_yaml(cfg))

    rollout_runner = DroidRolloutRunner(cfg)

    with suppress(KeyboardInterrupt):
        while True:
            rollout_runner.reset_robot()
            task = input(f"Enter task or leave empty to use default \"{cfg.task}\" (ctrl+c to exit): ").strip()
            rollout_runner.rollout_episode(task)

if __name__ == "__main__":
    main()
