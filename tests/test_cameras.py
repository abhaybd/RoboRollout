"""
Camera tests with the device/SDK replaced by fakes, covering frame processing and capture failure handling.
"""

import os
import signal
import sys
import time
import types
from unittest import mock

import numpy as np
import pytest
from omegaconf import OmegaConf

from robo_rollout.cameras.abstract import CameraException
from robo_rollout.cameras.config import GoProCameraConfig, V4LCameraConfig, ZedCameraConfig


def wait_for(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def raises_camera_exception(camera):
    try:
        camera.get_frame()
    except CameraException:
        return True
    return False


class FakeCapture:
    """Stands in for cv2.VideoCapture, returning `frame` (BGR) from each read unless told to fail."""

    def __init__(self, frame):
        self.frame = frame
        self.fail = False  # fail all reads
        self.fail_next = 0  # fail this many reads, then recover

    def isOpened(self):
        return True

    def set(self, prop, value):
        pass

    def get(self, prop):
        import cv2
        return {cv2.CAP_PROP_FRAME_WIDTH: self.frame.shape[1], cv2.CAP_PROP_FRAME_HEIGHT: self.frame.shape[0]}.get(prop, 30)

    def read(self):
        time.sleep(0.002)
        if self.fail_next > 0:
            self.fail_next -= 1
            return False, None
        if self.fail:
            return False, None
        return True, self.frame.copy()

    def release(self):
        pass


# ---------------------------------------------------------------- V4L

@pytest.fixture
def make_v4l(monkeypatch):
    import robo_rollout.cameras.v4l as v4l
    cameras = []

    def make(frame, **cfg):
        capture = FakeCapture(frame)
        monkeypatch.setattr(v4l.cv2, "VideoCapture", lambda *args: capture)
        height, width = frame.shape[:2]
        camera = v4l.V4LCamera(OmegaConf.structured(V4LCameraConfig(id=0, width=width, height=height, **cfg)))
        cameras.append(camera)
        return camera, capture

    yield make
    for camera in cameras:
        camera.close()


def test_v4l_converts_bgr_to_rgb(make_v4l):
    camera, _ = make_v4l(np.full((48, 64, 3), (0, 0, 255), np.uint8))  # red in BGR
    assert (camera.get_frame() == (255, 0, 0)).all()


def test_v4l_crops_stereo_side(make_v4l):
    frame = np.zeros((48, 128, 3), np.uint8)
    frame[:, 64:] = 255  # right image is white
    camera, _ = make_v4l(frame, stereo_side="right")
    out = camera.get_frame()
    assert out.shape == (48, 64, 3) and (out == 255).all()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # the capture thread raising is expected
def test_v4l_raises_after_persistent_read_failures(make_v4l):
    camera, capture = make_v4l(np.zeros((48, 64, 3), np.uint8))
    capture.fail = True
    # rather than returning the last frame forever
    assert wait_for(lambda: raises_camera_exception(camera))


def test_v4l_tolerates_transient_read_failures(make_v4l):
    import robo_rollout.cameras.v4l as v4l
    camera, capture = make_v4l(np.zeros((48, 64, 3), np.uint8))
    capture.fail_next = v4l.MAX_CONSECUTIVE_FAILED_READS - 1
    assert wait_for(lambda: capture.fail_next == 0)
    time.sleep(0.05)
    camera.get_frame()


# ---------------------------------------------------------------- GoPro

@pytest.fixture
def make_gopro(monkeypatch):
    pytest.importorskip("google.protobuf")
    import robo_rollout.cameras.gopro as gopro
    cameras = []

    def make(frame):
        capture = FakeCapture(frame)
        monkeypatch.setattr(gopro.cv2, "VideoCapture", lambda *args: capture)
        monkeypatch.setattr(gopro.os, "listdir", lambda path: ["usb-Elgato_ABC-video-index0"])
        camera = gopro.GoProCamera(OmegaConf.structured(GoProCameraConfig(id="ABC")))
        cameras.append(camera)
        return camera, capture

    yield make
    for camera in cameras:
        camera.close()


def test_gopro_removes_16x9_padding(make_gopro):
    # the elgato stream is a 4:3 image padded with black bars to 16:9
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:, 160:1120] = 255
    camera, _ = make_gopro(frame)
    out = camera.get_frame()
    assert out.shape == (480, 640, 3) and (out == 255).all()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # the capture thread raising is expected
def test_gopro_raises_after_persistent_read_failures(make_gopro):
    camera, capture = make_gopro(np.zeros((720, 1280, 3), np.uint8))
    capture.fail = True
    assert wait_for(lambda: raises_camera_exception(camera))


# ---------------------------------------------------------------- ZED

class FakeMat:
    def get_data(self):
        return self.data


class FakeZed:
    """Each grab yields a BGRA frame with B=10, G=20, R=a counter, so successive frames differ."""

    def __init__(self):
        self.count = 0

    def open(self, params):
        return "success"

    def grab(self, params):
        return "success"

    def retrieve_image(self, mat, view):
        self.count += 1
        mat.data = np.empty((720, 1280, 4), np.uint8)
        mat.data[...] = (10, 20, self.count % 256, 255)

    def close(self):
        pass


@pytest.fixture
def zed_camera(monkeypatch):
    sl = types.SimpleNamespace(
        Camera=FakeZed,
        Mat=FakeMat,
        InitParameters=mock.MagicMock,
        RuntimeParameters=mock.MagicMock,
        ERROR_CODE=types.SimpleNamespace(SUCCESS="success"),
        VIEW=types.SimpleNamespace(LEFT="left", RIGHT="right"),
        RESOLUTION=types.SimpleNamespace(HD720="hd720"),
        DEPTH_MODE=types.SimpleNamespace(NONE="none"),
    )
    monkeypatch.setitem(sys.modules, "pyzed", types.SimpleNamespace(sl=sl))
    monkeypatch.setitem(sys.modules, "pyzed.sl", sl)
    monkeypatch.delitem(sys.modules, "robo_rollout.cameras.zed", raising=False)
    from robo_rollout.cameras.zed import ZedCamera

    camera = ZedCamera(OmegaConf.structured(ZedCameraConfig(id=1)))
    yield camera
    camera.close()


def frames_updating(camera):
    first = camera.get_frame()
    return wait_for(lambda: not np.array_equal(camera.get_frame(), first))


def test_zed_converts_bgra_to_rgb(zed_camera):
    frame = zed_camera.get_frame()
    assert frame.shape == (720, 1280, 3)
    assert (frame[..., 1] == 20).all() and (frame[..., 2] == 10).all()
    assert frames_updating(zed_camera)


def test_zed_capture_survives_sigint(zed_camera):
    # ctrl+c in a terminal sends SIGINT to every process in the foreground group, including capture processes.
    # The capture process must keep running, or every later episode would see a frozen frame.
    os.kill(zed_camera._capture_proc.pid, signal.SIGINT)
    time.sleep(0.2)
    assert zed_camera._capture_proc.is_alive()
    assert frames_updating(zed_camera)


def test_zed_raises_if_capture_process_dies(zed_camera):
    zed_camera._capture_proc.kill()
    zed_camera._capture_proc.join()
    with pytest.raises(CameraException):
        zed_camera.get_frame()
