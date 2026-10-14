# Configuration

RoboRollout is configured with [Hydra structured configs](https://hydra.cc/docs/tutorials/structured_config/intro/). Each robot defines a config dataclass that extends [`RolloutConfig`][robo_rollout.config.RolloutConfig], which holds the settings shared by all robots, and sets defaults for its setup.

## Overriding settings

Any setting can be overridden on the command line using dotted paths:

```bash
python run_policy.py \
    task="put the red mug in the black bowl" \
    num_trials=10 \
    policy_host=gpu-server.local \
    robot.cameras.wrist_camera.id=12345678 \
    wandb.enabled=false
```

For changes you'll reuse, edit the defaults in the robot's `config.py` instead. Run with `--cfg job` to print the resolved config without running anything.

Settings that have no default (shown as `???`) must be set. Hydra checks types, and [`check_config()`][robo_rollout.rollout.check_config] checks for missing and invalid values before connecting to the robot, so mistakes fail fast.

## Shared settings

These settings are defined in [`RolloutConfig`][robo_rollout.config.RolloutConfig] and available for all robots.

| Setting | Default | Description |
| --- | --- | --- |
| `task` | required | The default task instruction. You can change it before each episode. |
| `policy_dt` | required | Seconds per control step |
| `policy_host` | `0.0.0.0` | Host of the policy server. Can also be a full `ws://` or `wss://` URI. |
| `policy_port` | `8000` | Port of the policy server, or null to use the URI's port |
| `policy_cameras` | `{}` | Which cameras to send to the policy (see below) |
| `nominal_timeout` | null | Episode timeout in seconds of nominal time, or null for no timeout |
| `num_trials` | null | Number of episodes to run, or null to run until you exit |
| `robot.cameras` | `{}` | The cameras to open, by name (see [Cameras](cameras.md)) |
| `image_input` | | How to resize images sent to the policy (see below) |
| `wandb` | | Weights & Biases logging (see below) |

Robots add their own settings: see [Getting Started: Franka DROID](../getting-started/droid.md#configuration) and [Getting Started: Bimanual YAM](../getting-started/yam.md#configuration).

### Cameras and policy observations

`robot.cameras` defines every camera to open, by name. Each one is recorded, but only those listed in `policy_cameras` are sent to the policy. `policy_cameras` maps observation keys to camera names, so the policy can use its own names for cameras:

```yaml
robot:
  cameras:
    wrist_camera: ...
    exo_camera_1: ...
    exo_camera_2: ...
policy_cameras:
  # observation key: camera name
  wrist_image: wrist_camera
  exterior_image: exo_camera_2
```

### Image resizing

Camera frames are resized before they're sent to the policy, as set by `image_input`:

| Setting | Description |
| --- | --- |
| `image_input.width`, `image_input.height` | The size of images sent to the policy |
| `image_input.method` | How to resize them: `center_pad` (the default) resizes to fit and pads the rest, `center_crop` resizes to fill and crops the rest, `downsample` takes every n-th pixel (the frame size must be a multiple of the target size), and `none` sends frames as-is |

Recorded videos always use the camera's full resolution.

### Weights & Biases

| Setting | Description |
| --- | --- |
| `wandb.enabled` | Whether to log episodes to W&B |
| `wandb.entity`, `wandb.project` | Where to log runs |
| `wandb.name` | Name of each run |
| `wandb.tags` | List of tags for each run |
| `wandb.notes` | Notes for each run. Defaults to the task. |

See [Running evaluations](running-evaluations.md#weights-biases) for what's logged.
