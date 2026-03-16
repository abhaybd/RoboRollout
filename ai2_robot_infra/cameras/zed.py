import multiprocessing.synchronize
from multiprocessing import Process, Event, Lock
from multiprocessing.shared_memory import SharedMemory
import time

import pyzed.sl as sl
import numpy as np

from ai2_robot_infra.cameras.abstract import BaseCamera, CameraException
from ai2_robot_infra.utils.misc import multi_closing, suppress_decorator


@suppress_decorator(KeyboardInterrupt)
def _capture_task(
    camera_id,
    width: int,
    height: int,
    frame_rate: int,
    side: str,
    depth: bool,
    init_event: multiprocessing.synchronize.Event,
    stop_event: multiprocessing.synchronize.Event,
    frame_lock: multiprocessing.synchronize.Lock,
    shm_name: str,
):
    assert (width, height) == (1280, 720), "Only HD720 resolution is supported for zed!"
    zed = sl.Camera()
    zed_view = sl.VIEW.LEFT if side == "left" else sl.VIEW.RIGHT
    init_params = sl.InitParameters()
    init_params.set_from_serial_number(camera_id)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = frame_rate
    if not depth:
        init_params.depth_mode = sl.DEPTH_MODE.NONE
    if (error_code := zed.open(init_params)) != sl.ERROR_CODE.SUCCESS:
        raise CameraException(f"Unable to open zed camera {camera_id}: {error_code}")

    curr_frame_shm = SharedMemory(name=shm_name)
    curr_frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=curr_frame_shm.buf)

    with multi_closing(zed, curr_frame_shm):
        img = sl.Mat()
        runtime_params = sl.RuntimeParameters()
        sleep_until = time.monotonic()
        while not stop_event.is_set():
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(img, zed_view)
                img_data = img.get_data()
                # Convert BGRA->RGB outside the lock to minimize lock hold time
                processed = np.ascontiguousarray(np.flip(img_data[..., :3], axis=-1))
                with frame_lock:
                    curr_frame[...] = processed

                # signal init after the first frame is captured (this is idempotent)
                init_event.set()

            curr_time = time.monotonic()
            sleep_until = max(sleep_until + 1. / frame_rate, curr_time)
            if curr_time < sleep_until:
                time.sleep(sleep_until - curr_time)


class ZedCamera(BaseCamera):
    def __init__(
        self,
        camera_id,
        width: int,
        height: int,
        frame_rate: int,
        side: str = "left",
        depth: bool = False,
    ):
        init_event = Event()
        self._stop_event = Event()
        self._frame_lock = Lock()
        self._curr_frame_shm = SharedMemory(create=True, size=height * width * 3)
        self._curr_frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=self._curr_frame_shm.buf)

        self._capture_proc = Process(
            target=_capture_task,
            args=(
                camera_id,
                width,
                height,
                frame_rate,
                side,
                depth,
                init_event,
                self._stop_event,
                self._frame_lock,
                self._curr_frame_shm.name,
            ),
            daemon=True,
            name=f"zed_capture_proc_{camera_id}",
        )
        self._capture_proc.start()

        init_start = time.monotonic()
        while time.monotonic() - init_start < 60:
            if init_event.wait(1):
                break
            if not self._capture_proc.is_alive():
                raise CameraException(f"Unable to initialize zed camera {camera_id}, process retcode: {self._capture_proc.exitcode}")
        else:
            self._capture_proc.terminate()
            self._capture_proc.join()
            raise CameraException(f"Unable to initialize zed camera {camera_id}")

    def get_frame_shape(self):
        return self._curr_frame.shape

    def get_frame(self):
        with self._frame_lock:
            ret = self._curr_frame.copy()
        return ret

    def close(self):
        self._stop_event.set()
        self._capture_proc.join()
        self._curr_frame_shm.close()
        self._curr_frame_shm.unlink()

    def __del__(self):
        if self._capture_proc.is_alive():
            self._stop_event.set()
            self._capture_proc.join(timeout=5)
            if self._capture_proc.is_alive():
                self._capture_proc.kill()
                self._capture_proc.join()
        try:
            self._curr_frame_shm.close()
            self._curr_frame_shm.unlink()
        except FileNotFoundError:
            pass
