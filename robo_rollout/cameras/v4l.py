import logging
import time
import threading

import cv2
import numpy as np

from robo_rollout.cameras.abstract import BaseCamera, CameraException
from robo_rollout.cameras.config import V4LCameraConfig


log = logging.getLogger(__name__)

# error after this many consecutive failed reads
MAX_CONSECUTIVE_FAILED_READS = 5


class V4LCamera(BaseCamera):
    """Camera that reads from a Video4Linux device (e.g. /dev/video0)."""

    def __init__(self, cfg: V4LCameraConfig):
        assert cfg.type == "v4l", f"Expected camera type 'v4l', got '{cfg.type}'"
        device, width, height, frame_rate = cfg.id, cfg.width, cfg.height, cfg.frame_rate
        self.device_path = device if isinstance(device, str) else f"/dev/video{device}"
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.stereo_side = cfg.stereo_side

        self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise CameraException(f"Failed to open V4L device {self.device_path}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, frame_rate)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (width, height):
            log.warning(
                f"V4LCamera: requested {width}x{height} but device reports "
                f"{actual_w}x{actual_h}; frames will be resized."
            )
        self._needs_resize = (actual_w, actual_h) != (width, height)

        self._current_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Wait briefly for the first frame
        time.sleep(0.5)

    def get_frame_shape(self) -> tuple:
        return (self.height, self.width, 3)

    def _capture_loop(self):
        n_failed_reads = 0
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                n_failed_reads += 1
                if n_failed_reads >= MAX_CONSECUTIVE_FAILED_READS:
                    raise CameraException(f"{n_failed_reads} consecutive failed reads from {self.device_path}")
                continue
            n_failed_reads = 0

            if self._needs_resize:
                frame = cv2.resize(frame, (self.width, self.height))

            # Convert BGR (OpenCV default) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # If this is a concatenated stereo camera, crop the frame to the desired side
            if self.stereo_side == "left":
                rgb_frame = np.ascontiguousarray(rgb_frame[:, :self.width // 2])
            elif self.stereo_side == "right":
                rgb_frame = np.ascontiguousarray(rgb_frame[:, self.width // 2:])

            with self._frame_lock:
                self._current_frame = rgb_frame

            time.sleep(1 / self.frame_rate)

    def get_frame(self) -> np.ndarray:
        if not self._thread.is_alive():
            raise CameraException(f"Capture thread for {self.device_path} died")
        with self._frame_lock:
            if self._current_frame is not None:
                return self._current_frame.copy()
        raise CameraException(f"No frame available from {self.device_path}")

    def close(self):
        self._stop_event.set()
        self._thread.join()
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        # skip if __init__ failed before there was anything to close
        if hasattr(self, "_thread") and not self._stop_event.is_set():
            self.close()
