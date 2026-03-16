from typing import Callable, Any
import threading
import time


class ThreadedPeriodicGetter:
    def __init__(self, func: Callable[[], Any], period: float):
        self._func = func
        self._period = period
        self._thread = threading.Thread(target=self._run)
        self._stop_event = threading.Event()
        self._latest_result: Any | None = None

    def start(self):
        self._thread.start()

    def stop(self, wait: bool = True):
        self._stop_event.set()
        if wait:
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
