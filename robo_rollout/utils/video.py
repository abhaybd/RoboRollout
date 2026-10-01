from subprocess import PIPE, Popen
from typing import Optional, Tuple

import numpy as np


class VideoWriter:
    """
    Writes RGB frames to a video file by piping them to an ffmpeg subprocess.
    ffmpeg runs in its own session, so ctrl+c in the terminal (which is sent to the whole
    foreground process group) doesn't kill it mid-recording.
    """

    def __init__(self, path: str, fps: float, codec: str = "h264"):
        self._path = path
        self._fps = fps
        self._codec = codec
        self._process: Optional[Popen] = None
        self._frame_shape: Optional[Tuple[int, ...]] = None

    def _start(self, height: int, width: int):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(self._fps), "-i", "pipe:",
            "-r", str(self._fps), "-c:v", self._codec, "-pix_fmt", "yuv420p", self._path,
        ]
        # stderr is inherited so ffmpeg errors show up in the terminal
        self._process = Popen(cmd, stdin=PIPE, start_new_session=True)

    def write(self, frame: np.ndarray):
        """Write a (H, W, 3) uint8 RGB frame. All frames must have the same shape."""
        if self._process is None:
            assert frame.ndim == 3 and frame.shape[-1] == 3, f"Expected (H, W, 3) frame, got {frame.shape}"
            self._frame_shape = frame.shape
            self._start(*frame.shape[:2])
        assert frame.shape == self._frame_shape, f"Expected frame of shape {self._frame_shape}, got {frame.shape}"
        self._process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8))

    def close(self):
        """Finish writing the video. The file is only created if at least one frame was written."""
        if self._process is None:
            return
        process, self._process = self._process, None
        try:
            # closing stdin signals EOF, so ffmpeg finalizes the file and exits
            process.stdin.close()
        finally:
            returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {returncode} while writing {self._path}")
