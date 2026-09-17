from abc import ABC, abstractmethod

import numpy as np

class BaseCamera(ABC):
    def close(self):
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
    pass
