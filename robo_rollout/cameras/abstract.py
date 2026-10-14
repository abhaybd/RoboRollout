from abc import ABC, abstractmethod

import numpy as np

class BaseCamera(ABC):
    """A camera that captures RGB frames. Create one from a config with robo_rollout.cameras.camera.create_camera()."""

    def close(self):
        """Stop capturing and release the camera."""
        pass

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Get latest frame as (H, W, 3) uint8 RGB array."""
        pass

    @abstractmethod
    def get_frame_shape(self) -> tuple:
        """Get the shape of the frames as (H, W, 3)."""
        pass


class CameraException(IOError):
    """Raised when a camera fails to open or capture frames."""
    pass
