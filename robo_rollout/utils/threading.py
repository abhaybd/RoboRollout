from queue import Empty, Queue
from typing import Callable, Any, Dict, Optional, Tuple
import threading
import time

import numpy as np

from robo_rollout.utils.video import VideoWriter


class ThreadedPeriodicGetter:
    def __init__(self, func: Callable[[], Any], period: float):
        self._func = func
        self._period = period
        self._thread = threading.Thread(target=self._run)
        self._stop_event = threading.Event()
        self._latest_result: Optional[Any] = None

    def start(self):
        self._thread.start()

    def stop(self, wait: bool = True):
        self._stop_event.set()
        # check ident to avoid join() before start() causing crash
        if wait and self._thread.ident is not None:
            self._thread.join()

    def _run(self):
        sleep_until = time.monotonic()
        while not self._stop_event.is_set():
            # no synchronization necessary since assignment is atomic
            self._latest_result = self._func()
            sleep_until += self._period
            curr_time = time.monotonic()
            if curr_time < sleep_until:
                time.sleep(sleep_until - curr_time)
            else:
                while sleep_until < curr_time:
                    sleep_until += self._period
                sleep_until -= self._period

    def get_latest_result(self) -> Any:
        return self._latest_result


class AsyncMultiVideoWriter:
    def __init__(self, video_writers: Dict[str, VideoWriter]):
        self._video_writers = video_writers
        self._frame_queue: "Queue[Optional[Tuple[str, np.ndarray]]]" = Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._write_error: Optional[Exception] = None
        self._thread = threading.Thread(target=self._run, name="async_video_writer")
        self._thread.start()

    def write(self, cam_name: str, frame: np.ndarray):
        # copy so caller can reuse buffer
        frame = frame.copy()
        with self._lock:
            if self._closed:
                raise RuntimeError(f"{__class__.__name__} is closed")
            if self._write_error is not None:
                return
            self._frame_queue.put((cam_name, frame))

    def _discard_pending(self):
        while True:
            try:
                self._frame_queue.get_nowait()
            except Empty:
                return

    def _run(self):
        while True:
            item = self._frame_queue.get()
            if item is None:
                break
            cam_name, frame = item
            try:
                self._video_writers[cam_name].write(frame)
            except Exception as exc:
                # store exception so we can throw on close, and drop queued
                # frames so a dead writer cannot grow memory for the rest of the episode
                with self._lock:
                    self._write_error = exc
                self._discard_pending()
                break

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._frame_queue.put(None)
        self._thread.join()
        error = self._write_error
        self._write_error = None
        for writer in self._video_writers.values():
            try:
                writer.close()
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def __del__(self):
        self.close()
