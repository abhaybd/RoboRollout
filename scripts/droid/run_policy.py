from collections import defaultdict
import json
from pathlib import Path
import time
from contextlib import contextmanager, suppress
from typing import Dict
from collections import deque
import logging
import traceback

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from polymetis import RobotInterface, GripperInterface
import torch
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import wandb

from robo_rollout.cameras.abstract import BaseCamera
from robo_rollout.policy.policy_client import PolicyClient
from robo_rollout.cameras.camera import create_camera
from robo_rollout.utils.image import resize_with_crop, resize_with_pad, downsample
from robo_rollout.utils.threading import ThreadedPeriodicGetter
from robo_rollout.utils.misc import lod_to_dol, measure_elapsed


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


def save_video(cfg, output_dir, cam_name, frame_deque):
    frame_list = list(frame_deque)
    frame_deque.clear()  # empty deque to free memory

    if len(frame_list) > 0:
        video_path = output_dir / f"{cam_name}.mp4"
        video_clip = ImageSequenceClip(frame_list, fps=1.0 / cfg.policy_dt)
        video_clip.write_videofile(str(video_path), codec="libx264", audio=False)
        print(f"Saving video of {cam_name} to {video_path}")
        del frame_list

        if cfg.wandb.enabled:
            return wandb.Video(
                video_path,
                caption=cam_name,
                format="mp4",
            )
    return None


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    if missing_keys := OmegaConf.missing_keys(cfg):
        raise ValueError(f"Missing keys: {missing_keys}")

    print("Evaluation Configuration:")
    print(OmegaConf.to_yaml(cfg))

    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    metadata = policy_client.get_server_metadata()
    print("Server metadata:")
    print(json.dumps(metadata, indent=2))

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    robot = RobotInterface(
        ip_address=cfg.robot.robot_host,
        enforce_version=False,
    )
    gripper = GripperInterface(ip_address=cfg.robot.robot_host)

    if cfg.wandb.enabled:
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_cfg["policy_metadata"] = metadata
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            dir=str(output_dir),
            tags=cfg.wandb.tags,
            config=wandb_cfg,
            notes=cfg.wandb.notes or cfg.task,
        )

    print("Opening cameras...")
    cameras: dict[str, BaseCamera] = {}
    for cam_name, camera_cfg in cfg.robot.cameras.items():
        try:
            camera = create_camera(camera_cfg)
        except:
            for cam in cameras.values():
                cam.close()
            raise
        cameras[cam_name] = camera

    print("Opening gripper...")
    gripper_speed = cfg.robot.gripper.speed
    gripper_force = cfg.robot.gripper.force
    gripper.goto(MAX_GRIP_WIDTH, gripper_speed, gripper_force)

    if cfg.autohome.enabled:
        print("Going home...")
        home_jp = torch.tensor(cfg.autohome.home_jointpos)
        robot.move_to_joint_positions(home_jp, cfg.autohome.duration)

    with cleanup_cameras(cameras):
        input("Press Enter to start...")

        last_gripper_cmd = MAX_GRIP_WIDTH

        observations = []
        actions = []
        cam_frames = defaultdict(deque)  # in sync with saved actions
        cam_frames_rt = defaultdict(deque)  # not in sync with saved actions  (e.g. due to overruns)
        n_steps = 0

        def capture_frames():
            for cam_name, camera in cameras.items():
                frame = camera.get_frame()
                cam_frames_rt[cam_name].append(frame)

        def get_obs():
            obs = {
                "task": cfg.task,
            }

            with measure_elapsed() as elapsed_robot:
                robot_state = get_robot_state(robot, gripper)
                observations.append(robot_state)
                obs.update(robot_state)

            with measure_elapsed() as elapsed_cam:
                curr_camera_frames = {}
                for cam_name, camera in cameras.items():
                    frame = camera.get_frame()
                    cam_frames[cam_name].append(frame)
                    if cfg.image_input.method == "center_pad":
                        frame_resized = resize_with_pad(
                            frame, cfg.image_input.height, cfg.image_input.width
                        )
                    elif cfg.image_input.method == "center_crop":
                        frame_resized = resize_with_crop(
                            frame, cfg.image_input.height, cfg.image_input.width
                        )
                    elif cfg.image_input.method == "downsample":
                        frame_resized = downsample(
                            frame, cfg.image_input.height, cfg.image_input.width
                        )
                    elif cfg.image_input.method == "none":
                        frame_resized = frame
                    else:
                        raise ValueError(
                            f"Unkown image rescaling method {cfg.image_input.method}"
                        )

                    frame_std = np.std(np.mean(frame, axis=-1)).item()
                    # Sometimes cameras fail silently and return single-color frames
                    if frame_std < 1.0:
                        if cfg.wandb.enabled and "camera_failure" not in wandb.run.tags:
                            wandb.run.tags = wandb.run.tags + ("camera_failure",)
                        raise ValueError(
                            "Camera failure detected, constant color frames"
                        )

                    curr_camera_frames[cam_name] = frame_resized

                for obs_cam_name, cam_name in cfg.policy_cameras.items():
                    obs[obs_cam_name] = curr_camera_frames[cam_name]

            obs_timing = {
                "elapsed_cam": elapsed_cam.value,
                "elapsed_robot": elapsed_robot.value,
            }
            return obs, obs_timing

        try:
            frame_capture_task = ThreadedPeriodicGetter(capture_frames, cfg.policy_dt)
            frame_capture_task.start()

            action_scale: float = cfg.action_scale
            robot.start_joint_impedance()
            sleep_until = time.monotonic()
            start = time.time()
            nominal_timeout_steps = int(np.ceil(cfg.nominal_timeout / cfg.policy_dt)) if cfg.nominal_timeout else None
            while nominal_timeout_steps is None or n_steps < nominal_timeout_steps:
                with measure_elapsed() as elapsed_obs:
                    obs, obs_timing = get_obs()
                n_steps += 1

                with measure_elapsed() as elapsed_infer:
                    action_dict = policy_client.infer(obs)

                with measure_elapsed() as elapsed_act:
                    cmd_jp_np = action_dict["arm"]
                    cmd_jp_np = (
                        obs["qpos"]["arm"]
                        + (cmd_jp_np - obs["qpos"]["arm"]) * action_scale
                    )
                    action_dict["arm"] = cmd_jp_np
                    action_dict["arm_vel"] = (cmd_jp_np - obs["qpos"]["arm"]) / 0.2
                    action_dict["arm_vel"] /= np.maximum(np.max(np.abs(action_dict["arm_vel"])), 1.0)
                    actions.append(action_dict)

                    cmd_jp = torch.tensor(cmd_jp_np)
                    cmd_grip_width = (
                        1.0 - action_dict["gripper"].item() / 255.0
                    ) * MAX_GRIP_WIDTH

                    robot.update_desired_joint_positions(cmd_jp)

                    if not np.isclose(cmd_grip_width, last_gripper_cmd, atol=0.002):
                        last_gripper_cmd = cmd_grip_width
                        gripper.goto(
                            cmd_grip_width, gripper_speed, gripper_force, blocking=False
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
        except KeyboardInterrupt:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            with suppress(KeyboardInterrupt):
                print("Terminating policy...")
                if robot.is_running_policy():
                    robot.terminate_current_policy(return_log=False)
            with suppress(KeyboardInterrupt):
                print("Closing policy client...")
                policy_client.close()
            with suppress(KeyboardInterrupt):
                print("Stopping frame capture task...")
                frame_capture_task.stop()

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
    video_dict = {}
    for cam_name, frame_deque in cam_frames.items():
        wandb_vid = save_video(cfg, output_dir, cam_name, frame_deque)
        if wandb_vid is not None:
            video_dict[f"video/{cam_name}"] = wandb_vid

    for cam_name, frame_deque in cam_frames_rt.items():
        wandb_vid = save_video(cfg, output_dir, f"{cam_name}_rt", frame_deque)
        if wandb_vid is not None:
            video_dict[f"video/{cam_name}_rt"] = wandb_vid

    if cfg.wandb.enabled:
        wandb.log({**info, **video_dict})

    print("Saving observations and actions")
    observations_dol = lod_to_dol(observations)
    actions_dol = lod_to_dol(actions)
    obs_path = output_dir / "observations.npz"
    act_path = output_dir / "actions.npz"
    np.savez_compressed(obs_path, **observations_dol)
    np.savez_compressed(act_path, **actions_dol)

    if cfg.wandb.enabled:
        wandb.save(obs_path, base_path=obs_path.parent)
        wandb.save(act_path, base_path=act_path.parent)

    (output_dir / "info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
