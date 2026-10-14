# Getting Started: Franka DROID

RoboRollout supports the Franka Panda following the [DROID](https://droid-dataset.github.io) conventions, controlled through [Polymetis](https://github.com/facebookresearch/fairo/tree/main/polymetis). The entry point lives in [`scripts/droid`](https://github.com/abhaybd/RoboRollout/tree/main/scripts/droid).

!!! info "Setup instructions"
    Full installation and usage instructions are in the **[DROID README](https://github.com/abhaybd/RoboRollout/blob/main/scripts/droid/README.md)**. It covers:

    - Installing Polymetis on the NUC
    - Installing RoboRollout and the ZED SDK on the inference PC, with [pixi](https://pixi.sh) or conda
    - Configuring the robot IP and cameras
    - Starting the robot and gripper servers, and running a policy

## Overview

The setup consists of:

- **The NUC**, which runs the Polymetis robot and gripper servers.
- **The inference PC**, which runs `run_policy.py`. It connects to the NUC, the cameras (ZED cameras by default), and your policy server.
- **The policy server**, which can run on the inference PC or on another machine.

Once everything is installed and the robot and gripper servers are running, start your policy server and run:

```bash
cd scripts/droid
pixi run python run_policy.py task="put the red mug in the black bowl"
```

Before each episode, the robot opens its gripper and moves to its home position.

## Observations and actions

Your policy server receives the following observation each step (see [Policy servers](../usage/policy-servers.md)):

| Key | Shape | Description |
| --- | --- | --- |
| `task` | string | The task instruction |
| `qpos/arm` | `(7,)` | Joint angles in radians |
| `qpos/gripper` | `(1,)` | Gripper position in `[0, 1]`, where 0 is open and 1 is closed |
| `ee_pose/pos` | `(3,)` | End-effector position |
| `ee_pose/quat` | `(4,)` | End-effector orientation as an xyzw quaternion |
| one key per entry in `policy_cameras` | `(360, 640, 3)` | `uint8` RGB images |

Nested keys such as `qpos/arm` are nested dicts in the observation, i.e. `obs["qpos"]["arm"]`.

The policy must return an action with the following keys (DROID's joint position action space, with gripper position):

| Key | Shape | Description |
| --- | --- | --- |
| `arm` | `(7,)` | Absolute joint position targets in radians |
| `gripper` | scalar or `(1,)` | Gripper position in `[0, 1]`, where 0 is open and 1 is closed |

The arm's motion is limited to `max_joint_delta` radians per step: if any joint would move further, the whole step is scaled down to fit.

The openpi and MolmoBot servers in [RoboRollout-VLA](https://github.com/abhaybd/RoboRollout_VLA) use this format, as does its LeRobot server with `--robot droid`.

## Configuration

The defaults are set in [`scripts/droid/config.py`](https://github.com/abhaybd/RoboRollout/blob/main/scripts/droid/config.py). Beyond the [settings shared by all robots](../usage/configuration.md), the DROID config adds:

| Setting | Default | Description |
| --- | --- | --- |
| `robot.robot_host` | `172.16.0.2` | IP of the NUC |
| `robot.gripper.speed`, `robot.gripper.force` | `0.05`, `0.1` | Gripper speed and force |
| `robot.cameras` | Three ZED cameras | `wrist_camera`, `exo_camera_1`, and `exo_camera_2`. Set their serial numbers to match your cameras. |
| `action_scale` | `1.0` | Scales each step's joint motion |
| `max_joint_delta` | `0.2` | Maximum joint motion per step, in radians |
| `autohome.enabled` | `true` | Move to the home position before each episode. Must be enabled to run more than one trial. |
| `autohome.home_jointpos` | DROID home position | Joint positions to start each episode from |
| `autohome.duration` | `5.0` | Seconds to take moving home |

Policies run at about 15 Hz (`policy_dt=0.066`), and episodes time out after 45 seconds of nominal time. Camera images are downsampled to 640x360.

!!! warning "Required settings"
    `policy_cameras.exo_camera_1` has no default, so choose which exterior camera the policy sees, e.g. `policy_cameras.exo_camera_1=exo_camera_2`.

    W&B logging is enabled by default with the maintainers' entity and project. Set `wandb.entity` and `wandb.project` to your own, or disable it with `wandb.enabled=false`.
