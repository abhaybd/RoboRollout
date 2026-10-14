# RoboRollout

RoboRollout is a lightweight framework for real-robot policy evaluation. It seeks to consolidate all robot-agnostic utilities and sensor code, and provides clean and easily extensible entrypoints for some robots.
This includes camera code, image/threading utilities, policy server clients, and more.

**Documentation:** https://abhaybd.github.io/RoboRollout/

The following cameras are supported:
- GoPro
- Intel RealSense
- ZED
- Generic video4linux devices

The following robots are supported:
- Franka / DROID (via Polymetis)
- Bimanual i2rt YAM

To run off-the-shelf policies and VLAs (openpi, MolmoBot, LeRobot, ...) with RoboRollout, see [RoboRollout-VLA](https://github.com/abhaybd/RoboRollout_VLA), which provides ready-made policy servers.

## Features

- **Policy server and client**: Policies run behind a websocket server, so they can run on a separate (e.g. GPU) machine from the robot. To serve your own policy, subclass [`InferencePolicy`](robo_rollout/policy/policy_server.py) and pass it to `WebsocketPolicyServer`. The policy is reset at the start of each episode.
- **Fixed-rate control loop**: The observe-infer-act loop runs at a fixed rate, and reports loop overruns, optionally broken down by which part of the step (observation, inference, or action) went over budget.
- **Cameras**: All cameras share a common interface. Frames are resized to the policy's input size (by center padding, center cropping, or downsampling), and camera names are mapped to the policy's observation keys in the config.
- **Recording**: Each episode saves its observations, actions, and two videos per camera:
  - `<camera>.mp4` is synced with the observations: it has exactly one frame per control step, the frame the policy saw. Use it to see what the policy was given at each step.
  - `<camera>_rt.mp4` is real-time: a background thread captures it at the control rate, regardless of how long each step takes. If steps overrun (e.g. due to slow inference), the observation-synced video plays back faster than real life, while this one still shows how the robot actually moved.
- **Evaluation**: You're prompted for a task before each episode and for success after it, and the results are saved alongside the episode.
- **Weights & Biases integration**: Each episode can be logged as its own W&B run, with its success, length, and task, videos from every camera, saved observations and actions, and the full config and policy server metadata. This makes it easy to compare success rates and watch rollouts across policies and tasks.
- **Configuration**: Settings are [Hydra](https://hydra.cc/) structured configs, so they can be overridden from the command line and are checked before connecting to the robot.
- **Export LeRobot Datasets**: Saved episodes, from disk or W&B, can be filtered by date and metadata (e.g. only successes) and collated into a [LeRobot](https://github.com/huggingface/lerobot) dataset, optionally uploaded to the HuggingFace Hub. See [Collating datasets](#collating-datasets).

## Installation

Please see the [robot-specific documentation](/scripts) for specific installation instructions for usage with the off-the-shelf supported platforms.

For custom use-cases or robots, simply install RoboRollout as a project dependency:

```bash
pip install "robo_rollout @ git+https://github.com/abhaybd/RoboRollout.git"
```

### Optional dependencies

Camera-specific packages/libraries are included as optional dependencies, or extras. To install, use any combination of the following extras:

- `gopro`
- `realsense`

The ZED SDK's python bindings (`pyzed`) aren't on PyPI, so to use ZED cameras, install the [ZED SDK](https://www.stereolabs.com/developers/release) and then its python bindings into your environment by running `python /usr/local/zed/get_python_api.py`.

## Collating datasets

[`scripts/collate_lerobot.py`](scripts/collate_lerobot.py) collates saved episodes into a LeRobot dataset. Sources can be local output directories or W&B projects (as `wandb:<entity>/<project>`), and episodes can be filtered by date (`--since`, `--until`) and by any metadata (`--filter KEY<op>VALUE`). Run it with [uv](https://docs.astral.sh/uv/), which installs its dependencies:

```bash
# list the episodes that would be collated
uv run scripts/collate_lerobot.py outputs/ wandb:my-entity/droid-evals \
    --repo-id my-hf-user/droid-pick-mug --since 2026-10-07 --filter success=true --dry-run

# collate them, and upload the dataset to the Hugging Face Hub
uv run scripts/collate_lerobot.py outputs/ wandb:my-entity/droid-evals \
    --repo-id my-hf-user/droid-pick-mug --since 2026-10-07 --filter success=true --upload
```

See the [documentation](https://abhaybd.github.io/RoboRollout/usage/collating-datasets/) for all options.

## Adding a new robot

All robot-agnostic logic (policy client, cameras, control loop timing, video recording, saving, and W&B logging) lives in [`robo_rollout/rollout.py`](robo_rollout/rollout.py). To add a robot, create a `scripts/<robot>/` folder containing:

1. A subclass of [`BaseRobot`](robo_rollout/robot.py) implementing `get_state()` and `apply_action()`, and optionally `reset()`, `start_control()`, `stop_control()` and `close()`. Its constructor can take whatever arguments you like.
2. A [structured config](https://hydra.cc/docs/tutorials/structured_config/intro/): a dataclass subclassing [`RolloutConfig`](robo_rollout/config.py), which holds the settings shared by all robots. Add robot-specific settings as new fields, and to add robot-specific settings under `robot`, also subclass `RobotConfig`. Override inherited fields to set defaults for your setup. See [`scripts/droid/config.py`](scripts/droid/config.py).
3. An entry point that registers the config, checks it, and passes your robot to `run()`:

```python
ConfigStore.instance().store(name="config", node=MyConfig)

@hydra.main(version_base=None, config_name="config")
def main(cfg: MyConfig):
    check_config(cfg)  # fail fast, before connecting to the robot
    run(cfg, MyRobot(cfg))
```

`run()` also accepts an optional `TimingBudget`, which sets how long each part of a control step should take. If given, loop overruns are broken down by which parts exceeded their budget; otherwise only the overall overrun is reported.

See [`scripts/droid/run_policy.py`](scripts/droid/run_policy.py) for a complete example.
