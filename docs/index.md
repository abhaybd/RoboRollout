# RoboRollout

RoboRollout is a lightweight framework for real-robot policy evaluation. It consolidates all robot-agnostic utilities and sensor code, and provides clean, easily extensible entry points for some robots.
This includes camera code, image/threading utilities, policy server clients, and more.

The following cameras are supported:

- GoPro
- Intel RealSense
- ZED
- Generic video4linux devices

The following robots are supported:

- Franka / DROID (via Polymetis): see [Getting Started: Franka DROID](getting-started/droid.md)
- Bimanual i2rt YAM: see [Getting Started: Bimanual YAM](getting-started/yam.md)

!!! tip "Off-the-shelf policies"
    To run off-the-shelf policies and VLAs (openpi, MolmoBot, LeRobot, ...) with RoboRollout, see [RoboRollout-VLA](https://github.com/abhaybd/RoboRollout_VLA), which provides ready-made policy servers.

## Features

- **Policy server and client**: Policies run behind a websocket server, so they can run on a separate (e.g. GPU) machine from the robot. To serve your own policy, subclass [`InferencePolicy`][robo_rollout.policy.policy_server.InferencePolicy] and pass it to [`WebsocketPolicyServer`][robo_rollout.policy.policy_server.WebsocketPolicyServer]. The policy is reset at the start of each episode. See [Policy servers](usage/policy-servers.md).
- **Fixed-rate control loop**: The observe-infer-act loop runs at a fixed rate, and reports loop overruns, optionally broken down by which part of the step (observation, inference, or action) went over budget.
- **Cameras**: All cameras share a common interface. Frames are resized to the policy's input size (by center padding, center cropping, or downsampling), and camera names are mapped to the policy's observation keys in the config. See [Cameras](usage/cameras.md).
- **Recording**: Each episode saves its observations, actions, and two videos per camera:
    - `<camera>.mp4` is synced with the observations: it has exactly one frame per control step, the frame the policy saw. Use it to see what the policy was given at each step.
    - `<camera>_rt.mp4` is real-time: a background thread captures it at the control rate, regardless of how long each step takes. If steps overrun (e.g. due to slow inference), the observation-synced video plays back faster than real life, while this one still shows how the robot actually moved.
- **Evaluation**: You're prompted for a task before each episode and for success after it, and the results are saved alongside the episode.
- **Weights & Biases integration**: Each episode can be logged as its own W&B run, with its success, length, and task, videos from every camera, saved observations and actions, and the full config and policy server metadata. This makes it easy to compare success rates and watch rollouts across policies and tasks.
- **Configuration**: Settings are [Hydra](https://hydra.cc/) structured configs, so they can be overridden from the command line and are checked before connecting to the robot. See [Configuration](usage/configuration.md).

## How it fits together

A rollout involves two processes, which can run on different machines:

1. A **policy server**, which loads your policy and serves actions over a websocket. Write your own with [`WebsocketPolicyServer`][robo_rollout.policy.policy_server.WebsocketPolicyServer], or use one from [RoboRollout-VLA](https://github.com/abhaybd/RoboRollout_VLA).
2. A **robot entry point** (e.g. `scripts/droid/run_policy.py`), which connects to the robot, cameras, and policy server, and runs the control loop. Each step, it reads the robot state and camera frames, sends them to the policy server, and applies the returned action to the robot.

All robot-agnostic logic lives in [`robo_rollout.rollout`][robo_rollout.rollout], so a robot entry point only has to implement reading state and applying actions. See [Adding a new robot](usage/adding-a-robot.md).
