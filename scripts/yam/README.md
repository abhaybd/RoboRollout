# Getting Started - Bimanual i2rt YAM

## Software installation

Install the [RealSense SDK](https://github.com/IntelRealSense/librealsense) if needed (`pyrealsense2` is installed below). Then, on the PC connected to the arms:

```bash
# Build dependencies for i2rt, if necessary
sudo apt install build-essential python3-dev linux-headers-$(uname -r)

# Install dependencies
git clone https://github.com/abhaybd/RoboRollout.git
cd RoboRollout/scripts/yam
uv sync
```

Then set up the CAN interfaces following the [i2rt docs](https://github.com/i2rt-robotics/i2rt#can-bus-setup). By default, the follower arms are expected on `can_follower_l` and `can_follower_r`, as in i2rt's bimanual setup.

## Policy Evaluation

### Hardware Setup

Modify the defaults in the [eval config](config.py) to match your hardware setup, or override them on the command line (e.g. `robot.left_channel=can0`). In particular, edit:
 * `robot.left_channel` and `robot.right_channel` to be the CAN interfaces of each arm
 * `robot.arm_type` and `robot.gripper_type` to match your arms (see the [i2rt docs](https://github.com/i2rt-robotics/i2rt#arm-variants) for values)
 * `robot.cameras` to match your camera configuration (including camera ID!). Each camera type has its own config class in [`robo_rollout/cameras/config.py`](/robo_rollout/cameras/config.py).
 * `policy_cameras` to select which cameras to include in policy observations, where the key is the name in the observation and the value is the camera name in `robot.cameras`
 * `autohome.left_home_jointpos` and `autohome.right_home_jointpos` to set the arm joint positions to start each episode from

The defaults (control rate, home position, camera and image resolution) match the [MolmoAct2](https://github.com/allenai/molmoact2) bimanual YAM setup.

If you wish, you can also enable W&B support (upload rollout videos by filling out the `wandb` block of the config and setting `wandb.enabled` to `true`).

### Running your policy

1. Power on the arms and bring up the CAN interfaces.
2. Start up your policy server.
3. Control the robot.
    When running the script, the arms will first go to the home position, and then prompt for a task (leave empty to reuse the previous one) to begin the episode. Press ctrl+c to end an episode early, and again at the task prompt to exit.
    ```bash
    cd scripts/yam
    uv run run_policy.py task="fold the towel"
    ```

    NOTE: the arms go limp when the script exits, so make sure the home position is one they can safely rest in.
