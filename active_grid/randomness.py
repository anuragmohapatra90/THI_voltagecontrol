"""Run the reference's NumPy-global generators reproducibly without cross-session RNG leaks."""
from contextlib import contextmanager
from threading import RLock
import numpy as np

_lock = RLock()


@contextmanager
def reference_random(seed):
    with _lock:
        previous = np.random.get_state()
        np.random.seed(int(seed))
        try:
            yield
        finally:
            np.random.set_state(previous)
