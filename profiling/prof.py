
from fischer.stopwatch import Stopwatch
from fischer.statistics import moving_average_tracker
from time import perf_counter_ns

from tabulate import tabulate

__all__ = ['profile']



def profile(func, print_interval: float = 500):
    stopwatch = Stopwatch()
    tracker = moving_average_tracker(stopwatch.lap)
    last_print = [0]
    print_interval *= 1e6
    def wrapper(*args, **kwargs):
        stopwatch.lap()
        result = tracker(*args, **kwargs)
        t, moving_average = tracker()
        ns = perf_counter_ns()
        if ns >= last_print[0] + print_interval:
            last_print[0] = ns
            print(f'{func.__name__} | Last call / Moving average: {t} ms / {moving_average} ms')
        return result
    return wrapper





