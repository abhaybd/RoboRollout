# Running evaluations

This page describes what happens when you run a robot entry point, such as `scripts/droid/run_policy.py`, and what it saves. For robot-specific setup, see [Getting Started: Franka DROID](../getting-started/droid.md) or [Getting Started: Bimanual YAM](../getting-started/yam.md).

## Starting a run

1. Start your policy server (see [Policy servers](policy-servers.md)).
2. Run the robot's entry point, setting the default task and any other [config overrides](configuration.md):

    ```bash
    python run_policy.py task="put the red mug in the black bowl" num_trials=10
    ```

The entry point connects to the robot and prints the full config. It then waits for the policy server if it isn't up yet, printing the server's metadata once connected, and opens the cameras.

## The episode loop

Each episode goes through the following steps:

1. **Reset**: the robot moves to its starting configuration ([`BaseRobot.reset()`][robo_rollout.robot.BaseRobot.reset]).
2. **Task prompt**: you're asked for the task instruction. Leave it empty to reuse the previous task (initially the `task` from the config). Press ctrl+c here to exit.
3. **Policy connection**: the client reconnects to the policy server, which resets the policy.
4. **Control loop**: every `policy_dt` seconds, RoboRollout reads the robot state and camera frames, sends them to the policy, and applies the returned action. The episode ends after `nominal_timeout` seconds of nominal time, or when you press ctrl+c.
5. **Success prompt**: you're asked whether the policy succeeded (`y`/`n`). Pressing ctrl+c here counts as a failure.
6. **Saving**: the episode's data and videos are saved, and uploaded to W&B if enabled.

This repeats for `num_trials` episodes, or until you exit if `num_trials` is null.

### Timing

The control loop runs at a fixed rate of one step per `policy_dt` seconds. If a step takes longer than that, RoboRollout prints a warning with how long the loop overran by, and starts the next step immediately, without trying to make up for the lost time.

If the robot entry point passes a [`TimingBudget`][robo_rollout.rollout.TimingBudget] to [`run()`][robo_rollout.rollout.run], overruns are broken down by which part of the step went over its budget:

```text
WARN: Loop overrun by 0.042s
Budget violations:
  Inference: 0.098s
```

The episode timeout is measured in steps, not wall time: an episode lasts `ceil(nominal_timeout / policy_dt)` steps, which takes longer than `nominal_timeout` if steps overrun.

## Outputs

Outputs are saved to Hydra's output directory, which by default is `outputs/<date>/<time>/` in the directory you ran from. Each run's directory contains Hydra's logs and a copy of the config in `.hydra/`, and one directory per episode:

```text
outputs/2026-10-14/12-34-56/
├── .hydra/
│   ├── config.yaml
│   └── overrides.yaml
├── episode_0000/
│   ├── info.json
│   ├── observations.npz
│   ├── actions.npz
│   ├── <camera>.mp4
│   └── <camera>_rt.mp4
└── episode_0001/
    └── ...
```

### `info.json`

| Key | Description |
| --- | --- |
| `task` | The task instruction |
| `success` | Whether the policy succeeded, as entered at the success prompt |
| `success_rate` | `success` as a float, for averaging in W&B |
| `episode_length` | Wall time of the episode, in seconds |
| `episode_length_steps` | Number of control steps |
| `episode_length_nominal` | `episode_length_steps * policy_dt` |

### `observations.npz` and `actions.npz`

`observations.npz` holds the robot state returned by [`get_state()`][robo_rollout.robot.BaseRobot.get_state] at each step, and `actions.npz` holds the actions returned by [`apply_action()`][robo_rollout.robot.BaseRobot.apply_action]. Camera frames aren't included, since they're saved as videos.

Nested dicts are flattened into keys joined with `/`, and each key holds an array with one entry per step. For example, with the DROID robot:

```python
import numpy as np

obs = np.load("episode_0000/observations.npz")
obs["qpos/arm"].shape  # (n_steps, 7)

actions = np.load("episode_0000/actions.npz")
actions["arm"].shape  # (n_steps, 7)
actions["server_timing/infer_ms"]  # inference time reported by the policy server
```

### Videos

Each camera in `robot.cameras` gets two videos, both written at `1 / policy_dt` frames per second:

- `<camera>.mp4` is synced with the observations: it has exactly one frame per control step, the frame the policy saw. Use it to see what the policy was given at each step. Frames are saved at the camera's full resolution, before resizing.
- `<camera>_rt.mp4` is real-time: a background thread captures it at the control rate, regardless of how long each step takes. If steps overrun (e.g. due to slow inference), the observation-synced video plays back faster than real life, while this one still shows how the robot actually moved.

## Weights & Biases

To log episodes to [W&B](https://wandb.ai), set `wandb.enabled=true` and fill in the rest of the `wandb` config (see [Configuration](configuration.md#weights-biases)). Each episode is logged as its own run, with:

- The full config, plus the policy server's metadata under `policy_metadata`
- Everything in `info.json`
- Every video, under `video/<camera>` and `video/<camera>_rt`
- `observations.npz` and `actions.npz`, as run files

The run's notes default to the task, so you can tell runs apart at a glance. To compare policies, group runs by a config value such as `policy_metadata.model_name`, and average `success_rate`.

## Collating episodes into a dataset

Saved episodes, whether on disk or in W&B, can be collated into a LeRobot dataset. See [Collating datasets](collating-datasets.md).
