# Adding a new robot

All robot-agnostic logic (policy client, cameras, control loop timing, video recording, saving, and W&B logging) lives in [`robo_rollout.rollout`][robo_rollout.rollout]. To add a robot, create a `scripts/<robot>/` folder containing a robot class, a config, and an entry point. See [`scripts/droid`](https://github.com/abhaybd/RoboRollout/tree/main/scripts/droid) for a complete example.

## 1. The robot class

Subclass [`BaseRobot`][robo_rollout.robot.BaseRobot]. Its constructor can take whatever arguments you like, typically the config. Implement:

| Method | Required | Description |
| --- | --- | --- |
| [`get_state()`][robo_rollout.robot.BaseRobot.get_state] | Yes | Return the robot's proprioceptive state as a (possibly nested) dict of arrays. It's included in the policy observation and saved to `observations.npz`. |
| [`apply_action(obs, action)`][robo_rollout.robot.BaseRobot.apply_action] | Yes | Command the robot with an action from the policy. `obs` is the observation the action was computed from. Return the action to save to `actions.npz`, e.g. with post-processed commands. |
| [`reset()`][robo_rollout.robot.BaseRobot.reset] | No | Move to the starting configuration. Called before each episode. |
| [`start_control()`][robo_rollout.robot.BaseRobot.start_control] | No | Prepare to accept actions, e.g. by starting a controller. Called at the start of each episode. |
| [`stop_control()`][robo_rollout.robot.BaseRobot.stop_control] | No | Undo `start_control()`. Called at the end of each episode, even if it failed. |
| [`close()`][robo_rollout.robot.BaseRobot.close] | No | Release the robot. Called once, on exit. |

`apply_action()` is called once per control step, so it shouldn't block for long. Document the state and action format in the class docstring, so policy server authors know what to expect.

```python
from robo_rollout.robot import BaseRobot


class MyRobot(BaseRobot):
    """
    Robot state:
        qpos: (6,) joint angles in radians

    Policy actions:
        qpos: (6,) absolute joint position targets in radians
    """

    def __init__(self, cfg: MyConfig):
        self.cfg = cfg
        self.arm = connect_to_arm(cfg.robot.host)

    def reset(self):
        self.arm.move_to(self.cfg.home_jointpos)

    def get_state(self):
        return {"qpos": self.arm.get_joint_positions()}

    def apply_action(self, obs, action):
        self.arm.command_joint_positions(action["qpos"])
        return action

    def close(self):
        self.arm.disconnect()
```

## 2. The config

Write a [structured config](https://hydra.cc/docs/tutorials/structured_config/intro/): a dataclass subclassing [`RolloutConfig`][robo_rollout.config.RolloutConfig], which holds the settings shared by all robots (see [Configuration](configuration.md)).

- Add robot-specific settings as new fields.
- To add robot-specific settings under `robot`, also subclass [`RobotConfig`][robo_rollout.config.RobotConfig].
- Override inherited fields to set defaults for your setup, such as `policy_dt`, `robot.cameras`, and `image_input`.

```python
from dataclasses import dataclass, field
from typing import Dict, List

from robo_rollout.cameras.config import CameraConfig, RealSenseCameraConfig
from robo_rollout.config import ImageInputConfig, RobotConfig, RolloutConfig


@dataclass
class MyRobotConfig(RobotConfig):
    host: str = "192.168.1.10"
    cameras: Dict[str, CameraConfig] = field(default_factory=lambda: {
        "wrist_camera": RealSenseCameraConfig(id="123456789012"),
    })


@dataclass
class MyConfig(RolloutConfig):
    robot: MyRobotConfig = field(default_factory=MyRobotConfig)

    policy_dt: float = 0.1
    policy_cameras: Dict[str, str] = field(default_factory=lambda: {"wrist_image": "wrist_camera"})
    image_input: ImageInputConfig = field(default_factory=lambda: ImageInputConfig(width=224, height=224))

    home_jointpos: List[float] = field(default_factory=lambda: [0.0] * 6)
```

See [`scripts/droid/config.py`](https://github.com/abhaybd/RoboRollout/blob/main/scripts/droid/config.py) for a complete example.

## 3. The entry point

Register the config, check it, and pass your robot to [`run()`][robo_rollout.rollout.run]:

```python
import hydra
from hydra.core.config_store import ConfigStore

from robo_rollout.rollout import check_config, run

ConfigStore.instance().store(name="config", node=MyConfig)


@hydra.main(version_base=None, config_name="config")
def main(cfg: MyConfig):
    check_config(cfg)  # fail fast, before connecting to the robot
    run(cfg, MyRobot(cfg))


if __name__ == "__main__":
    main()
```

`run()` takes ownership of the robot, and closes it on exit.

`run()` also accepts an optional [`TimingBudget`][robo_rollout.rollout.TimingBudget], which sets how long each part of a control step should take, in seconds. If given, loop overruns are broken down by which parts exceeded their budget; otherwise only the overall overrun is reported. Parts left unset split the rest of `policy_dt` evenly. For example, to budget 10 ms for observations and actions, and leave the rest for inference:

```python
run(cfg, MyRobot(cfg), TimingBudget(obs=0.01, act=0.01))
```

See [`scripts/droid/run_policy.py`](https://github.com/abhaybd/RoboRollout/blob/main/scripts/droid/run_policy.py) for a complete example.
