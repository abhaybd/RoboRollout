from collections import defaultdict
from contextlib import contextmanager, suppress
from functools import wraps
import time


class MutableNullableFloat:
    def __init__(self):
        self.value: float | None = None


@contextmanager
def measure_elapsed():
    start_time = time.monotonic()
    mnf = MutableNullableFloat()
    yield mnf
    end_time = time.monotonic()
    mnf.value = end_time - start_time


@contextmanager
def multi_closing(*resources):
    """
    Context manager similar to contextlib.closing(), but for multiple resources.
    Closes resources in reverse order.
    If any exceptions occur during closing, the first one is raised after all resources are closed.
    """
    try:
        yield resources
    finally:
        e = None
        for r in reversed(resources):
            try:
                r.close()
            except Exception as ex:
                if e is None:
                    e = ex
        if e is not None:
            raise e


def suppress_decorator(*exceptions):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with suppress(*exceptions):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def flatten_dict(d, pfx=""):
    ret = {}
    for k, v in d.items():
        if isinstance(v, dict):
            ret.update(flatten_dict(v, f"{pfx}{k}/"))
        else:
            ret[f"{pfx}{k}"] = v
    return ret


def lod_to_dol(lod):
    ret = defaultdict(list)
    for d in lod:
        flat_d = flatten_dict(d)
        for k, v in flat_d.items():
            ret[k].append(v)
    return dict(ret)
