import os
import subprocess

import cv2
import numpy as np
import pytest

from robo_rollout.utils.video import VideoWriter


def ffmpeg_works():
    try:
        return subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not ffmpeg_works(), reason="needs a working ffmpeg on PATH")


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def test_writes_all_frames_in_rgb(tmp_path):
    path = tmp_path / "out.mp4"
    writer = VideoWriter(str(path), fps=15)
    for _ in range(10):
        writer.write(np.full((48, 64, 3), (255, 0, 0), np.uint8))  # pure red
    writer.close()

    frames = read_frames(path)
    assert len(frames) == 10
    mean_rgb = frames[0].reshape(-1, 3).mean(axis=0)
    assert mean_rgb[0] > 200 and mean_rgb[1] < 50 and mean_rgb[2] < 50


def test_no_file_without_frames(tmp_path):
    path = tmp_path / "out.mp4"
    VideoWriter(str(path), fps=15).close()
    assert not path.exists()


def test_rejects_frame_shape_change(tmp_path):
    writer = VideoWriter(str(tmp_path / "out.mp4"), fps=15)
    writer.write(np.zeros((48, 64, 3), np.uint8))
    with pytest.raises(AssertionError):
        writer.write(np.zeros((96, 128, 3), np.uint8))
    writer.close()


def test_ffmpeg_runs_in_own_session(tmp_path):
    # ctrl+c in a terminal is sent to the whole session's foreground process group,
    # so ffmpeg must be in a separate session to survive the ctrl+c that ends an episode
    writer = VideoWriter(str(tmp_path / "out.mp4"), fps=15)
    writer.write(np.zeros((48, 64, 3), np.uint8))
    try:
        assert os.getsid(writer._process.pid) != os.getsid(0)
    finally:
        writer.close()
