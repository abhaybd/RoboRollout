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

Then set up the CAN interfaces following the [i2rt docs](https://github.com/i2rt-robotics/i2rt#can-bus-setup). By default, the arms are expected on `can_left` and `can_right`, as in MolmoAct2's YAM setup.

## Policy Evaluation

### Hardware Setup

Modify the defaults in the [eval config](config.py) to match your hardware setup, or override them on the command line (e.g. `robot.left_channel=can0`). In particular, edit:
 * `robot.left_channel` and `robot.right_channel` to be the CAN interfaces of each arm
 * `robot.arm_type` and `robot.gripper_type` to match your arms (see the [i2rt docs](https://github.com/i2rt-robotics/i2rt#arm-variants) for values)
 * `robot.cameras` to match your camera configuration (including camera ID!). Each camera type has its own config class in [`robo_rollout/cameras/config.py`](/robo_rollout/cameras/config.py).
 * `policy_cameras` to select which cameras to include in policy observations, where the key is the name in the observation and the value is the camera name in `robot.cameras`
 * `autohome.left_home_jointpos` and `autohome.right_home_jointpos` to set the arm joint positions to start each episode from

The defaults (control rate, home position, camera and image resolution) match the [MolmoAct2](https://github.com/allenai/molmoact2) bimanual YAM setup.

If you wish, you can also enable W&B support (upload rollout videos by setting `wandb.enabled` to `true` and filling out `wandb.entity` and `wandb.project`).

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

### Running MolmoAct2

[MolmoAct2-BimanualYAM](https://huggingface.co/allenai/MolmoAct2-BimanualYAM) runs with the LeRobot server in [RoboRollout-VLA](https://github.com/abhaybd/RoboRollout_VLA), using the [LeRobot port](https://huggingface.co/lerobot/MolmoAct2-BimanualYAM-LeRobot) of the checkpoint. Start the server on a GPU machine:

```bash
serve-lerobot --policy lerobot/MolmoAct2-BimanualYAM-LeRobot --robot yam --invert-gripper --n-action-steps 25 \
    --camera top_camera=top --camera left_wrist_camera=left --camera right_wrist_camera=right
```

As in MolmoAct2's own eval, this executes the first 25 actions of each 30-action chunk, and flips the grippers to i2rt's convention (1 is open), which MolmoAct2 was trained with. Then run the rollout with `commanded_state=true`:

```bash
cd scripts/yam
uv run run_policy.py commanded_state=true task="fold the towel"
```

`commanded_state=true` sends the last commanded joint positions as the robot state instead of the measured ones, as MolmoAct2's YAM driver does (so its training data has the commanded positions too). This matters most for the grippers, which stop short of their commanded position when grasping.

The cameras must stream 640x360 color, the resolution MolmoAct2 was trained on. D405s connected to USB 2 ports don't offer this mode, so connect the cameras to USB 3 ports.

### Running ABC

[ABC](https://github.com/amazon-far/abc) policies (ABC-DiT and ABC-VLA) run with the ABC server in [RoboRollout-VLA](https://github.com/abhaybd/RoboRollout_VLA). Download a checkpoint with ABC's `prepare.py`, e.g. `uv run prepare.py --dagger-checkpoint folding_paper_box`, which prints the flags it was deployed with, and serve it with those settings:

```bash
serve-abc --checkpoint cache/dagger/folding_paper_box.pt --diffusion-steps 5 --execute-chunk-dim 20 --rtc-prefix-length 4
```

ABC policies are trained on 640x480 images and the measured joint positions (the default, `commanded_state=false`), and each checkpoint starts from its own `init_q`. Set the cameras and image size to 640x480, the home positions to the checkpoint's `init_q` (the first 6 and the 8th to 13th values), and the task to the checkpoint's prompt:

```bash
cd scripts/yam
uv run run_policy.py task=fold_the_paper_box image_input.height=480 \
    robot.cameras.top_camera.height=480 robot.cameras.left_wrist_camera.height=480 robot.cameras.right_wrist_camera.height=480 \
    'autohome.left_home_jointpos=[-0.010221,1.562507,0.396568,0.875347,-0.077486,-0.039846]' \
    'autohome.right_home_jointpos=[0.010027,1.526663,0.338067,1.017695,0.044539,0.045304]'
```

The server predicts each chunk while the robot waits, so steps where a chunk is predicted overrun, unlike ABC's own deploy, which overlaps inference with execution.
