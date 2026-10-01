import itertools
import json
from pathlib import Path
import time
from contextlib import closing, suppress
from dataclasses import dataclass, fields, replace
from typing import Dict, List, Optional
import logging
import traceback

import hydra
from ffmpegcv import VideoWriter
from omegaconf import OmegaConf
import numpy as np
import wandb

from robo_rollout.cameras.abstract import BaseCamera
from robo_rollout.cameras.camera import create_camera
from robo_rollout.config import ResizeMethod, RolloutConfig
from robo_rollout.policy.policy_client import PolicyClient
from robo_rollout.robot import BaseRobot
from robo_rollout.utils.image import resize_with_crop, resize_with_pad, downsample
from robo_rollout.utils.threading import AsyncMultiVideoWriter, ThreadedPeriodicGetter
from robo_rollout.utils.misc import lod_to_dol, measure_elapsed, multi_closing, suppress_decorator


def check_config(cfg: RolloutConfig):
    """
    Raise if any config values are missing (i.e. set to ???) or invalid.
    Types are already checked by hydra, so this only checks values.
    """
    if missing_keys := OmegaConf.missing_keys(cfg):
        raise ValueError(f"Missing keys: {missing_keys}")
    assert cfg.num_trials is None or cfg.num_trials >= 0, \
        f"num_trials must be null or non-negative, got {cfg.num_trials}"


@dataclass(frozen=True)
class TimingBudget:
    """
    Time budget in seconds for each part of a control step, used to report which parts caused a loop overrun.
    Budgets left as None evenly split the time remaining in policy_dt after the specified budgets.
    """
    obs: Optional[float] = None
    infer: Optional[float] = None
    act: Optional[float] = None

    def resolve(self, policy_dt: float) -> "TimingBudget":
        """Get a copy with every budget filled in."""
        budgets = {f.name: getattr(self, f.name) for f in fields(self)}
        unspecified = [name for name, budget in budgets.items() if budget is None]
        if not unspecified:
            return self
        specified_total = sum(budget for budget in budgets.values() if budget is not None)
        remaining = max(0.0, policy_dt - specified_total) / len(unspecified)
        return replace(self, **{name: remaining for name in unspecified})


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


def create_writer(cfg: RolloutConfig, output_dir: Path, cam_name: str):
    return VideoWriter(
        str(output_dir / f"{cam_name}.mp4"),
        fps=round(1.0 / cfg.policy_dt),
        pix_fmt="rgb24",
    )


class RobotControl:
    """Keeps the robot under control for the duration of an episode."""

    def __init__(self, robot: BaseRobot):
        self.robot = robot
        self.robot.start_control()

    @suppress_decorator(KeyboardInterrupt)
    def close(self):
        self.robot.stop_control()


class Rollout:
    """Data recorded during a single episode."""

    def __init__(self, cfg: RolloutConfig, output_dir: Path, cameras: Dict[str, BaseCamera]):
        self.cfg = cfg
        self.cameras = cameras
        self.observations = []
        self.actions = []
        self.n_steps = 0

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
        with multi_closing(self.video_writer, self.rt_video_writer):
            self.rt_capture_task.stop()


class RolloutRunner:
    def __init__(self, cfg: RolloutConfig, robot: BaseRobot, timing_budget: Optional[TimingBudget]):
        self.cfg = cfg
        self.timing_budget = timing_budget.resolve(cfg.policy_dt) if timing_budget is not None else None
        self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self._episode_idx = 0
        self.policy_client = None

        self._reset_policy()
        self.robot = robot

        self.resize_func = {
            ResizeMethod.center_pad: resize_with_pad,
            ResizeMethod.center_crop: resize_with_crop,
            ResizeMethod.downsample: downsample,
            ResizeMethod.none: lambda x, h, w: x,
        }[cfg.image_input.method]

        print("Opening cameras...")
        self.cameras: Dict[str, BaseCamera] = {}
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

    def get_obs(self, observations: List, video_writer: AsyncMultiVideoWriter):
        obs = {
            "task": self.cfg.task,
        }

        with measure_elapsed() as elapsed_robot:
            robot_state = self.robot.get_state()
            observations.append(robot_state)
            obs.update(robot_state)

        with measure_elapsed() as elapsed_cam:
            curr_camera_frames = {}
            for cam_name, camera in self.cameras.items():
                frame = camera.get_frame()
                video_writer.write(cam_name, frame)
                frame_resized = self.resize_func(frame, self.cfg.image_input.height, self.cfg.image_input.width)
                curr_camera_frames[cam_name] = frame_resized

            for obs_cam_name, cam_name in self.cfg.policy_cameras.items():
                obs[obs_cam_name] = curr_camera_frames[cam_name]

        obs_timing = {
            "elapsed_cam": elapsed_cam.value,
            "elapsed_robot": elapsed_robot.value,
        }
        return obs, obs_timing

    @suppress_decorator(KeyboardInterrupt)
    def _rollout_episode(self, rollout: Rollout):
        cfg = self.cfg
        rollout.start_rt_capture()

        sleep_until = time.monotonic()
        nominal_timeout_steps = int(np.ceil(cfg.nominal_timeout / cfg.policy_dt)) if cfg.nominal_timeout else None
        while nominal_timeout_steps is None or rollout.n_steps < nominal_timeout_steps:
            with measure_elapsed() as elapsed_obs:
                obs, obs_timing = self.get_obs(rollout.observations, rollout.video_writer)
            rollout.n_steps += 1

            with measure_elapsed() as elapsed_infer:
                action_dict = self.policy_client.infer(obs)

            with measure_elapsed() as elapsed_act:
                rollout.actions.append(self.robot.apply_action(obs, action_dict))

            sleep_until += cfg.policy_dt
            curr_time = time.monotonic()
            if curr_time < sleep_until:
                time.sleep(sleep_until - curr_time)
            else:
                print(f"WARN: Loop overrun by {curr_time - sleep_until:.3f}s")
                if self.timing_budget is not None:
                    print("Budget violations:")
                    if elapsed_obs.value > self.timing_budget.obs:
                        print(f"  Observation: {elapsed_obs.value:.3f}s")
                        for k, v in obs_timing.items():
                            name = k.split("_", 1)[-1].title()
                            print(f"    {name}: {v:.3f}s")
                    if elapsed_infer.value > self.timing_budget.infer:
                        print(f"  Inference: {elapsed_infer.value:.3f}s")
                    if elapsed_act.value > self.timing_budget.act:
                        print(f"  Action: {elapsed_act.value:.3f}s")

                while sleep_until < curr_time:
                    sleep_until += cfg.policy_dt
                sleep_until -= cfg.policy_dt

    def rollout_episode(self, task: Optional[str]):
        if task:
            self.cfg.task = task
        cfg = self.cfg
        self._reset_policy()
        episode_dir = self.output_dir / f"episode_{self._episode_idx:04d}"
        self._episode_idx += 1
        episode_dir.mkdir(parents=True, exist_ok=True)

        if cfg.wandb.enabled:
            wandb_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
            wandb_cfg["policy_metadata"] = self.metadata
            wandb_run = wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.name,
                dir=str(episode_dir),
                tags=cfg.wandb.tags,
                config=wandb_cfg,
                notes=cfg.wandb.notes or cfg.task,
            )
        else:
            wandb_run = None

        try:
            robot_control = RobotControl(self.robot)
            rollout = Rollout(self.cfg, episode_dir, self.cameras)
            try:
                with multi_closing(rollout, robot_control):
                    with measure_elapsed() as elapsed_episode:
                        try:
                            self._rollout_episode(rollout)
                        except Exception:
                            traceback.print_exc()
            except Exception:
                # Report closing failures without crashing
                traceback.print_exc()

            success = prompt_success()

            n_steps = rollout.n_steps

            info = {
                "episode_length": elapsed_episode.value,
                "episode_length_nominal": n_steps * cfg.policy_dt,
                "episode_length_steps": n_steps,
                "success": success,
                "success_rate": float(success),
                "task": cfg.task,
            }
            (episode_dir / "info.json").write_text(json.dumps(info, indent=2))

            print("Saving observations and actions")
            observations_dol = lod_to_dol(rollout.observations)
            actions_dol = lod_to_dol(rollout.actions)
            obs_path = episode_dir / "observations.npz"
            act_path = episode_dir / "actions.npz"
            np.savez_compressed(obs_path, **observations_dol)
            np.savez_compressed(act_path, **actions_dol)

            video_paths = {}
            missing_videos = []
            for cam_name in self.cameras:
                for video_name in (cam_name, f"{cam_name}_rt"):
                    video_path = episode_dir / f"{video_name}.mp4"
                    # writers only create the file on the first frame, so it may not exist
                    if video_path.exists():
                        video_paths[video_name] = video_path
                    else:
                        missing_videos.append(str(video_path))

            if wandb_run is not None:
                video_dict = {
                    f"video/{video_name}": wandb.Video(video_path, caption=video_name, format="mp4")
                    for video_name, video_path in video_paths.items()
                }
                wandb_run.summary.update({**info, **video_dict})

                wandb_run.save(obs_path, base_path=obs_path.parent)
                wandb_run.save(act_path, base_path=act_path.parent)

            # raise only after saving and uploading everything else, so the episode isn't lost
            if missing_videos:
                raise FileNotFoundError(f"Videos missing: {missing_videos}")
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    def close(self):
        self.policy_client.close()
        for cam_name, cam in self.cameras.items():
            print(f"Closing camera {cam_name}...")
            cam.close()


def run(
    cfg: RolloutConfig,
    robot: BaseRobot,
    timing_budget: Optional[TimingBudget] = None,
):
    """
    Run cfg.num_trials policy rollouts, or until the user exits with ctrl+c if it's null.
    Call this from a robot's hydra entry point.

    Args:
        cfg: The full config.
        robot: The robot to control. run() takes ownership of it, and closes it on exit.
        timing_budget: Per-step time budgets used to report which parts of a step caused a loop overrun.
            If None, only the overall overrun is reported.
    """
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=150)

    # close the robot even if the runner fails to initialize
    with closing(robot):
        check_config(cfg)
        num_trials: Optional[int] = cfg.num_trials

        print("Evaluation Configuration:")
        print(OmegaConf.to_yaml(cfg))

        with closing(RolloutRunner(cfg, robot, timing_budget)) as rollout_runner:
            with suppress(KeyboardInterrupt):
                for trial_idx in (itertools.count() if num_trials is None else range(num_trials)):
                    if num_trials is not None:
                        print(f"Trial {trial_idx + 1}/{num_trials}")
                    robot.reset()
                    task = input(f"Enter task or leave empty to use default \"{cfg.task}\" (ctrl+c to exit): ").strip()
                    rollout_runner.rollout_episode(task)
