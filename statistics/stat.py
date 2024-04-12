


__all__ = ['moving_average_tracker']


def moving_average_tracker(func, max_calls: int = 16):
    inv_max_calls = 1. / max_calls
    total_calls = [0]
    calls = []
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if len(calls) < max_calls:
            calls.append(result)
            moving_average = sum(calls) * inv_max_calls
        else:
            calls[total_calls[0] % max_calls] = result
            moving_average = sum(calls) / len(calls)
        total_calls[0] += 1
        return result, moving_average
    return wrapper

