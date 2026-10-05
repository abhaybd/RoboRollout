"""
End-to-end tests of run() with a fake robot, cameras, policy client, and video writers.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import robo_rollout.rollout as rollout
from robo_rollout.cameras.abstract import BaseCamera
from robo_rollout.cameras.config import V4LCameraConfig
from robo_rollout.config import ImageInputConfig, ResizeMethod, RobotConfig, RolloutConfig
from robo_rollout.robot import BaseRobot

STEPS_PER_EPISODE = 5


class FakeRobot(BaseRobot):
    def __init__(self):
        self.events = []
        self.qpos = np.zeros(3)

    def reset(self):
        self.events.append("reset")

    def start_control(self):
        self.events.append("start_control")

    def stop_control(self):
        self.events.append("stop_control")

    def get_state(self):
        return {"qpos": {"arm": self.qpos.copy()}}

    def apply_action(self, obs, action):
        self.qpos = action["arm"]
        return {**action, "applied": True}

    def close(self):
        self.events.append("close")


class FakeCamera(BaseCamera):
    def __init__(self, cfg):
        self.shape = (cfg.height, cfg.width, 3)
        self.closed = False

    def get_frame(self):
        return np.full(self.shape, 128, np.uint8)

    def get_frame_shape(self):
        return self.shape

    def close(self):
        self.closed = True


class FakeWriter:
    """Like VideoWriter, only creates its file once a frame is written (unless create_file is False)."""

    def __init__(self, path, create_file):
        self.path = path
        self.create_file = create_file
        self.n_frames = 0

    def write(self, frame):
        if self.n_frames == 0 and self.create_file:
            self.path.touch()
        self.n_frames += 1

    def close(self):
        pass


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Replaces all IO used by run(). Tweak the returned namespace before calling run()."""
    env = SimpleNamespace(
        output_dir=tmp_path,
        answers=[],  # scripted responses to input() prompts; a KeyboardInterrupt instance is raised instead
        connections=[],  # one entry per policy client connection, holding the observations it received
        fail_policy=None,  # (connection index, step, exception) at which the policy raises
        writers={},
        create_video_files=True,
        cameras=[],
        camera_error=None,  # raised when creating the camera after the first one
    )

    class FakePolicy:
        def __init__(self, host, port):
            self.observations = []
            self.index = len(env.connections)
            env.connections.append(self.observations)

        def get_server_metadata(self):
            return {}

        def infer(self, obs):
            self.observations.append(obs)
            if env.fail_policy is not None:
                index, step, exc = env.fail_policy
                if self.index == index and len(self.observations) == step:
                    raise exc
            return {"arm": obs["qpos"]["arm"] + 1}

        def close(self):
            pass

    def create_camera(cfg):
        if env.cameras and env.camera_error is not None:
            raise env.camera_error
        env.cameras.append(FakeCamera(cfg))
        return env.cameras[-1]

    def create_writer(cfg, output_dir, name):
        writer = FakeWriter(output_dir / f"{name}.mp4", env.create_video_files)
        env.writers[(output_dir.name, name)] = writer
        return writer

    def scripted_input(prompt=""):
        assert env.answers, f"unexpected prompt: {prompt}"
        answer = env.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path)))))
    monkeypatch.setattr(rollout, "PolicyClient", FakePolicy)
    monkeypatch.setattr(rollout, "create_camera", create_camera)
    monkeypatch.setattr(rollout, "create_writer", create_writer)
    monkeypatch.setattr("builtins.input", scripted_input)
    return env


def make_cfg(**overrides):
    cfg = RolloutConfig(
        robot=RobotConfig(cameras={
            "wrist": V4LCameraConfig(id=0, width=64, height=48),
            "exo": V4LCameraConfig(id=1, width=64, height=48),
        }),
        policy_dt=0.02,
        policy_cameras={"image": "exo"},
        nominal_timeout=0.02 * STEPS_PER_EPISODE,
        num_trials=2,
        task="pick",
        image_input=ImageInputConfig(width=32, height=24, method=ResizeMethod.downsample),
    )
    return OmegaConf.merge(OmegaConf.structured(cfg), overrides)


def load_episode(env, idx):
    episode_dir = env.output_dir / f"episode_{idx:04d}"
    return SimpleNamespace(
        info=json.loads((episode_dir / "info.json").read_text()),
        obs=dict(np.load(episode_dir / "observations.npz")),
        actions=dict(np.load(episode_dir / "actions.npz")),
    )


def test_runs_num_trials_and_saves_each_episode(env):
    env.answers = ["", "y", "place", "n"]  # (task, success) for each episode
    robot = FakeRobot()
    rollout.run(make_cfg(), robot)

    ep0, ep1 = load_episode(env, 0), load_episode(env, 1)
    assert (ep0.info["task"], ep0.info["success"]) == ("pick", True)
    assert (ep1.info["task"], ep1.info["success"]) == ("place", False)

    for ep in (ep0, ep1):
        assert ep.info["episode_length_steps"] == STEPS_PER_EPISODE
        assert ep.obs["qpos/arm"].shape == (STEPS_PER_EPISODE, 3)
        # saved actions are what apply_action returned, aligned with the observations
        np.testing.assert_array_equal(ep.actions["arm"], ep.obs["qpos/arm"] + 1)
        assert ep.actions["applied"].all()

    # each episode gets a fresh policy connection, so no policy state carries over between episodes
    assert [len(observations) for observations in env.connections] == [0, STEPS_PER_EPISODE, STEPS_PER_EPISODE]
    policy_obs = env.connections[2][0]
    assert policy_obs["task"] == "place"
    assert policy_obs["image"].shape == (24, 32, 3)
    assert set(policy_obs) == {"task", "qpos", "image"}

    # every policy step is recorded for every camera
    for episode in ("episode_0000", "episode_0001"):
        for camera in ("wrist", "exo"):
            assert env.writers[(episode, camera)].n_frames == STEPS_PER_EPISODE
            assert env.writers[(episode, f"{camera}_rt")].n_frames > 0

    assert robot.events == ["reset", "start_control", "stop_control"] * 2 + ["close"]
    assert all(camera.closed for camera in env.cameras)


def test_runs_until_ctrl_c_without_num_trials(env):
    env.answers = ["", "y", "", "n", "", "y", KeyboardInterrupt()]
    robot = FakeRobot()
    rollout.run(make_cfg(num_trials=None), robot)

    assert [load_episode(env, i).info["success"] for i in range(3)] == [True, False, True]
    assert robot.events[-1] == "close"


@pytest.mark.parametrize("exc", [RuntimeError("policy server error"), KeyboardInterrupt()])
def test_policy_error_or_ctrl_c_ends_only_the_current_episode(env, exc):
    env.fail_policy = (1, 3, exc)  # first episode's connection fails on step 3
    env.answers = ["", "n", "", "y"]
    robot = FakeRobot()
    rollout.run(make_cfg(), robot)

    ep0, ep1 = load_episode(env, 0), load_episode(env, 1)
    assert ep0.info["episode_length_steps"] == 3
    # actions are only saved once applied
    assert len(ep0.obs["qpos/arm"]) == 3 and len(ep0.actions["arm"]) == 2
    assert ep1.info["episode_length_steps"] == STEPS_PER_EPISODE
    assert robot.events == ["reset", "start_control", "stop_control"] * 2 + ["close"]


def test_missing_videos_raise_after_episode_is_saved(env):
    env.create_video_files = False
    env.answers = ["", "y"]
    robot = FakeRobot()
    with pytest.raises(FileNotFoundError, match="Videos missing"):
        rollout.run(make_cfg(num_trials=1), robot)

    assert load_episode(env, 0).info["success"] is True
    assert robot.events == ["reset", "start_control", "stop_control", "close"]


def test_robot_and_cameras_closed_if_startup_fails(env):
    env.camera_error = IOError("camera not found")
    robot = FakeRobot()
    with pytest.raises(IOError, match="camera not found"):
        rollout.run(make_cfg(), robot)

    assert robot.events == ["close"]
    assert len(env.cameras) == 1 and env.cameras[0].closed
