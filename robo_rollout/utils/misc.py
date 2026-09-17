from contextlib import contextmanager, suppress
from functools import wraps


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
