import numpy as np
import pytest

from robo_rollout.utils.threading import AsyncMultiVideoWriter, ThreadedPeriodicGetter


class RecordingWriter:
    def __init__(self, fail_on_write=False):
        self.frames = []
        self.closed = False
        self.fail_on_write = fail_on_write

    def write(self, frame):
        if self.fail_on_write:
            raise IOError("disk full")
        self.frames.append(frame)

    def close(self):
        self.closed = True


def test_async_writer_routes_frames_and_copies_caller_buffer():
    a, b = RecordingWriter(), RecordingWriter()
    writer = AsyncMultiVideoWriter({"a": a, "b": b})
    buffer = np.zeros((2, 2, 3), np.uint8)
    writer.write("a", buffer)
    buffer[:] = 7  # caller reuses its buffer before the frame is written
    writer.write("b", buffer)
    writer.close()

    assert len(a.frames) == 1 and (a.frames[0] == 0).all()
    assert len(b.frames) == 1 and (b.frames[0] == 7).all()
    assert a.closed and b.closed


def test_async_writer_raises_write_error_on_close_and_still_closes_writers():
    bad, good = RecordingWriter(fail_on_write=True), RecordingWriter()
    writer = AsyncMultiVideoWriter({"bad": bad, "good": good})
    writer.write("bad", np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(IOError, match="disk full"):
        writer.close()
    assert bad.closed and good.closed


def test_async_writer_rejects_writes_after_close():
    writer = AsyncMultiVideoWriter({"a": RecordingWriter()})
    writer.close()
    with pytest.raises(RuntimeError):
        writer.write("a", np.zeros((2, 2, 3), np.uint8))


def test_periodic_getter_can_be_stopped_without_starting():
    # a rollout that fails before starting real-time capture still stops it on close
    ThreadedPeriodicGetter(lambda: None, 0.01).stop()
