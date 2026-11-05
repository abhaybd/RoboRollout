# Getting Started: Bimanual YAM

RoboRollout supports a bimanual pair of [i2rt](https://github.com/i2rt-robotics/i2rt) YAM arms, following the DROID conventions for each arm. The entry point lives in [`scripts/yam`](https://github.com/abhaybd/RoboRollout/tree/main/scripts/yam).

!!! info "Setup instructions"
    Full installation and usage instructions are in the **[YAM README](https://github.com/abhaybd/RoboRollout/blob/main/scripts/yam/README.md)**. It covers:

    - Installing i2rt, RoboRollout, and the RealSense SDK with [uv](https://docs.astral.sh/uv/)
    - Setting up the CAN interfaces
    - Configuring the arms and cameras
    - Running a policy

## Overview

`run_policy.py` runs on the PC connected to the arms, and connects to both arms over CAN, the cameras (RealSense cameras by default), and your policy server. The policy server can run on the same PC or on another machine.

Once everything is installed and the CAN interfaces are up, start your policy server and run:

```bash
cd scripts/yam
uv run run_policy.py task="fold the towel"
```

Before each episode, both arms move to their home positions and open their grippers.

!!! warning
    The arms go limp when the script exits, so make sure the home position is one they can safely rest in.

## Configuring your setup

Before your first run, point the config at your hardware. Edit the defaults in [`scripts/yam/config.py`](https://github.com/abhaybd/RoboRollout/blob/main/scripts/yam/config.py), so you don't have to pass them every time. You can also override any setting on the command line for a single run (see [Configuration](../usage/configuration.md)).

1. **Arms**: set `robot.arm_type` and `robot.gripper_type` to match your arms. If your CAN interfaces aren't named as in MolmoAct2's YAM setup (`can_left` and `can_right`), set `robot.left_channel` and `robot.right_channel`.
2. **Cameras**: set the serial number (`id`) of each camera in `robot.cameras`. The RealSense SDK's `rs-enumerate-devices` tool can list the serial numbers of connected cameras.

    ```python
    cameras: Dict[str, CameraConfig] = field(default_factory=lambda: {
        "top_camera": RealSenseCameraConfig(id="123456789012", width=640, height=360),
        "left_wrist_camera": RealSenseCameraConfig(id="234567890123", width=640, height=360),
        "right_wrist_camera": RealSenseCameraConfig(id="345678901234", width=640, height=360),
    })
    ```

    If your setup has a different set of cameras, add or remove entries to match (see [Cameras](../usage/cameras.md)).
3. **Home pose**: `autohome.left_home_jointpos` and `autohome.right_home_jointpos` default to the folded rest pose. If you change them, make sure the arms can safely go limp there, since they do when the script exits.
4. **Policy cameras**: `policy_cameras` sets which cameras the policy sees, and under which observation keys. By default, the policy sees all three cameras, under their own names. Change it to match what your policy expects.
5. **Policy server**: if your policy server isn't running on the same machine on port 8000, set `policy_host` and `policy_port`.
6. **W&B** (optional): logging is off by default. To log episodes, set `wandb.enabled=true`, `wandb.entity`, and `wandb.project` (see [Running evaluations](../usage/running-evaluations.md#weights-biases)).

To check your config, print it without connecting to anything:

```bash
uv run run_policy.py --cfg job
```

Settings still shown as `???` have to be set before running, apart from `task`, which you pass when you run.

## Observations and actions

Your policy server receives the following observation each step (see [Policy servers](../usage/policy-servers.md)):

| Key | Shape | Description |
| --- | --- | --- |
| `task` | string | The task instruction |
| `qpos/left_arm`, `qpos/right_arm` | `(6,)` | Joint angles in radians |
| `qpos/left_gripper`, `qpos/right_gripper` | `(1,)` | Gripper position in `[0, 1]`, where 0 is open and 1 is closed |
| one key per entry in `policy_cameras` | `(360, 640, 3)` | `uint8` RGB images |

Joint and gripper positions are measured, or the last commanded positions with `commanded_state=true`. Nested keys such as `qpos/left_arm` are nested dicts in the observation, i.e. `obs["qpos"]["left_arm"]`.

The policy must return an action with the following keys:

| Key | Shape | Description |
| --- | --- | --- |
| `left_arm`, `right_arm` | `(6,)` | Absolute joint position targets in radians |
| `left_gripper`, `right_gripper` | scalar or `(1,)` | Gripper position in `[0, 1]`, where 0 is open and 1 is closed |

Each arm's motion is limited to `max_joint_delta` radians per step: if any joint would move further, that arm's whole step is scaled down to fit.

The LeRobot server in [RoboRollout-VLA](https://github.com/abhaybd/RoboRollout_VLA) supports this format with `--robot yam`.

## Running MolmoAct2

[MolmoAct2-BimanualYAM](https://huggingface.co/allenai/MolmoAct2-BimanualYAM) runs with the LeRobot server in [RoboRollout-VLA](https://github.com/abhaybd/RoboRollout_VLA), using the [LeRobot port](https://huggingface.co/lerobot/MolmoAct2-BimanualYAM-LeRobot) of the checkpoint. It executes the first 25 actions of each 30-action chunk and flips the grippers to i2rt's convention, as MolmoAct2's own eval does:

```bash
serve-lerobot --policy lerobot/MolmoAct2-BimanualYAM-LeRobot --robot yam --invert-gripper --n-action-steps 25 \
    --camera top_camera=top --camera left_wrist_camera=left --camera right_wrist_camera=right
```

Then run the rollout with `commanded_state=true`, which sends the last commanded joint positions as the state, as MolmoAct2's YAM driver does (and so as its training data was recorded), instead of the measured ones:

```bash
uv run run_policy.py commanded_state=true task="fold the towel"
```

The cameras must stream 640x360 color, the resolution MolmoAct2 was trained on. D405s connected to USB 2 ports don't offer this mode, so connect the cameras to USB 3 ports.

## Running ABC

[ABC](https://github.com/amazon-far/abc) policies (ABC-DiT and ABC-VLA) run with the ABC server in [RoboRollout-VLA](https://github.com/abhaybd/RoboRollout_VLA). Download a checkpoint with ABC's `prepare.py`, e.g. `uv run prepare.py --dagger-checkpoint folding_paper_box`, which prints the flags it was deployed with, and serve it with those settings:

```bash
serve-abc --checkpoint cache/dagger/folding_paper_box.pt --diffusion-steps 5 --execute-chunk-dim 20 --rtc-prefix-length 4
```

ABC policies are trained on 640x480 images and the measured joint positions (the default, `commanded_state=false`), and each checkpoint starts from its own `init_q`. Set the cameras and image size to 640x480, the home positions to the checkpoint's `init_q` (the first 6 and the 8th to 13th values), and the task to the checkpoint's prompt:

```bash
uv run run_policy.py task=fold_the_paper_box image_input.height=480 \
    robot.cameras.top_camera.height=480 robot.cameras.left_wrist_camera.height=480 robot.cameras.right_wrist_camera.height=480 \
    'autohome.left_home_jointpos=[-0.010221,1.562507,0.396568,0.875347,-0.077486,-0.039846]' \
    'autohome.right_home_jointpos=[0.010027,1.526663,0.338067,1.017695,0.044539,0.045304]'
```

The server predicts each chunk while the robot waits, so steps where a chunk is predicted overrun, unlike ABC's own deploy, which overlaps inference with execution.

## Configuration

The defaults are set in [`scripts/yam/config.py`](https://github.com/abhaybd/RoboRollout/blob/main/scripts/yam/config.py), and match the [MolmoAct2](https://github.com/allenai/molmoact2) bimanual YAM setup. Beyond the [settings shared by all robots](../usage/configuration.md), the YAM config adds:

| Setting | Default | Description |
| --- | --- | --- |
| `robot.arm_type`, `robot.gripper_type` | `yam`, `linear_4310` | i2rt arm and gripper types (see the [i2rt docs](https://github.com/i2rt-robotics/i2rt#arm-variants)) |
| `robot.left_channel`, `robot.right_channel` | `can_left`, `can_right` | CAN interfaces of each arm |
| `robot.cameras` | Three RealSense cameras | `top_camera`, `left_wrist_camera`, and `right_wrist_camera`, at 640x360 |
| `commanded_state` | `false` | Report the last commanded joint positions as the robot state instead of the measured ones, as MolmoAct2 does |
| `action_scale` | `1.0` | Scales each step's joint motion |
| `max_joint_delta` | `0.2` | Maximum joint motion per step, in radians |
| `autohome.left_home_jointpos`, `autohome.right_home_jointpos` | All zeros (the folded rest pose) | Joint positions to start each episode from |
| `autohome.duration` | `4.0` | Seconds to take moving home |

Policies run at 30 Hz (`policy_dt=0.03333`), and episodes time out after 45 seconds of nominal time. All three cameras are sent to the policy at their full resolution.
