import errno
import math
import os
import signal
import time
from functools import partial, wraps
from itertools import product
from typing import Tuple

import numpy as np



class TimeoutError(BaseException):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(repeat_id, signum, frame):
            # logger.warning(f"Catched the signal ({repeat_id}) Setting signal handler {repeat_id + 1}")
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator


if __name__ == "__main__":
        print()
