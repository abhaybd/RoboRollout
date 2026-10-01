# RoboRollout

RoboRollout is a lighweight framework for real-robot policy evaluation. It seeks to consolidate all robot-agnostic utilities and sensor code, and provides clean and easily extensible entrypoints for some robots.
This includes camera code, image/threading utilities, policy server clients, and more.

The following cameras are supported:
- GoPro
- Intel RealSense
- ZED
- Generic video4linux devices

The following robots are supported:
- Franka / DROID (via Polymetis)

## Installation

To use, clone and install this repository:

```bash
git clone https://github.com/abhaybd/RoboRollout.git
cd robo_rollout
pip install -e .
```

See [robot-specific documentation](/scripts) for more specific installation instructions.

### Optional dependencies

Camera-specific packages/libraries are included as optional dependencies, or extras. To install, use any combination of the following extras:

- `gopro`
- `realsense`
- `zed`

## Adding a new robot

All robot-agnostic logic (policy client, cameras, control loop timing, video recording, saving, and W&B logging) lives in [`robo_rollout/rollout.py`](robo_rollout/rollout.py). To add a robot, create a `scripts/<robot>/` folder containing:

1. A subclass of [`BaseRobot`](robo_rollout/robot.py) implementing `get_state()` and `apply_action()`, and optionally `reset()`, `start_control()`, `stop_control()` and `close()`. Its constructor can take whatever arguments you like.
2. A hydra config. Copy [`scripts/droid/config.yaml`](scripts/droid/config.yaml) and edit the `robot` block; the other keys are used by the shared rollout code.
3. An entry point that checks the config and passes your robot to `run()`:

```python
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    check_missing_keys(cfg)  # fail fast, before connecting to the robot
    run(cfg, MyRobot(cfg))
```

`run()` also accepts an optional `TimingBudget`, which sets how long each part of a control step should take; overruns are reported per part.

See [`scripts/droid/run_policy.py`](scripts/droid/run_policy.py) for a complete example.
