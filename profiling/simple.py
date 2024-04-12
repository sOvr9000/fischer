
import time

def timer_decorator(n: int = 1):
    def _timer_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter_ns()
            result = func(*args, **kwargs)
            if n > 1:
                for _ in range(n - 1):
                    func(*args, **kwargs)
            end_time = time.perf_counter_ns()
            print(f"Execution time of {func.__name__} (avg of {n} runs): {(end_time - start_time) / (n * 1e6)} ms")
            return result
        return wrapper
    return _timer_decorator

