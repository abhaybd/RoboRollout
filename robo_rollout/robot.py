from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseRobot(ABC):
    """
    The robot-specific part of a rollout. To support a new robot, subclass this and pass an instance
    to robo_rollout.rollout.run(), which handles the policy, cameras, control loop timing, and logging.
    """

    def reset(self):
        """Move the robot to its starting configuration. Called before each episode."""
        pass

    def start_control(self):
        """Prepare the robot to accept actions, e.g. by starting a controller. Called at the start of each episode."""
        pass

    def stop_control(self):
        """Undo start_control(). Called at the end of each episode, even if it failed."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the proprioceptive state of the robot, as a (possibly nested) dict of arrays.
        This is included in the policy observation and saved to observations.npz.
        """
        pass

    @abstractmethod
    def apply_action(self, obs: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Command the robot with an action returned by the policy.
        obs is the observation the action was computed from.
        Returns the action to save to actions.npz, which may differ from the input (e.g. post-processed commands).
        """
        pass

    def close(self):
        """Release the robot. Called once, when run() exits."""
        pass
