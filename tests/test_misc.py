import pytest

from robo_rollout.utils.misc import lod_to_dol, multi_closing


class Resource:
    def __init__(self, name, log, error=None):
        self.name, self.log, self.error = name, log, error

    def close(self):
        self.log.append(self.name)
        if self.error is not None:
            raise self.error


def test_multi_closing_closes_in_reverse_order():
    log = []
    with multi_closing(Resource("a", log), Resource("b", log), Resource("c", log)):
        pass
    assert log == ["c", "b", "a"]


def test_multi_closing_closes_everything_and_raises_first_error():
    log = []
    resources = [
        Resource("a", log, ValueError("a failed")),
        Resource("b", log),
        Resource("c", log, KeyError("c failed")),
    ]
    with pytest.raises(KeyError):
        with multi_closing(*resources):
            pass
    assert log == ["c", "b", "a"]


def test_multi_closing_closes_when_body_raises():
    log = []
    with pytest.raises(RuntimeError):
        with multi_closing(Resource("a", log)):
            raise RuntimeError
    assert log == ["a"]


def test_lod_to_dol_flattens_nested_dicts():
    steps = [
        {"qpos": {"arm": 1, "gripper": 2}, "t": 0},
        {"qpos": {"arm": 3, "gripper": 4}, "t": 1},
    ]
    assert lod_to_dol(steps) == {"qpos/arm": [1, 3], "qpos/gripper": [2, 4], "t": [0, 1]}
