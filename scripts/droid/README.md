# Getting Started - Franka DROID

## Software installation

### On your NUC

If you already have the [DROID](https://droid-dataset.github.io) framework installed, skip this step! Otherwise, follow [these instructions](https://droid-dataset.github.io/droid/software-setup/host-installation.html#configuring-python-virtual-environment-conda). If you wish, you can install only [polymetis](https://github.com/facebookresearch/fairo/tree/main/polymetis) instead of DROID.

### On your Inference PC

First, install the [ZED SDK](https://www.stereolabs.com/developers/release), matching your CUDA version. Then:

```bash
# Create fresh conda env for polymetis
conda create -n roborollout_droid python=3.8
conda activate roborollout_droid
conda install -c pytorch -c fair-robotics -c aihabitat -c conda-forge polymetis

# Install ZED SDK python bindings
cd /usr/local/zed
python get_python_api.py

# Install dependencies
pip install git+https://github.com/abhaybd/RoboRollout.git#egg=robo_rollout[zed]
```


## Policy Evaluation

### Hardware Setup

Modify the defaults in the [eval config](config.py) to match your hardware setup, or override them on the command line (e.g. `robot.robot_host=<ip>`). In particular, edit:
 * `robot.robot_host` to be the IP of the NUC
 * `robot.cameras` to match your camera configuration (including camera ID!). Each camera type has its own config class in [`robo_rollout/cameras/config.py`](/robo_rollout/cameras/config.py).
 * `policy_cameras` to select which cameras to include in policy observations, where the key is the name in the observation and the value is the camera name in `robot.cameras`

If you wish, you can also enable W&B support (upload rollout videos by filling out the `wandb` block of the config and setting `wandb.enabled` to `true`).

### Running your policy

1. Log into Franka Desk to unlock joints and enable FCI.
2. Start the robot and gripper servers on the NUC (see [polymetis docs](https://facebookresearch.github.io/fairo/polymetis/usage.html)). For example, if DROID is installed on the NUC:
    ```bash
    # In terminal 1
    ssh <nuc-hostname>
    conda activate polymetis-local
    cd droid/droid/fairo/polymetis
    launch_robot.py robot_client=franka_hardware
    ```

    ```bash
    # In terminal 2
    ssh <nuc-hostname>
    conda activate polymetis-local
    cd droid/droid/fairo/polymetis
    launch_gripper.py gripper=robotiq_2f
    ```
3. Start up your policy server in terminal 3.
4. Activate the environment and control the robot.
    When running the script, the robot will first go to a home position, and then prompt for a task (leave empty to reuse the previous one) to begin the episode. Press ctrl+c to end an episode early, and again at the task prompt to exit.
    ```bash
    # In terminal 4
    conda activate roborollout_droid
    python scripts/droid/run_policy.py task="put the red mug in the black bowl"
    ```

    NOTE: if you get `ffmpeg`/`ImageIO` errors, you may need to run `conda remove --force ffmpeg` and install it system-wide with `sudo apt install ffmpeg`.
