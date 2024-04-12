
import numpy as np
from typing import Union



class Stats:
    def __init__(self):
        self.metrics: dict[str, list[Union[float, np.ndarray[float]]]] = {}
    def add_metric(self, name: str):
        self.metrics[name] = []
    def record(self, metric_name: str, value: Union[float, np.ndarray[float]], max_length: int = 16384):
        '''
        If `max_length = None`, then do not limit the total length of the recorded data.
        '''
        if metric_name not in self.metrics:
            self.add_metric(name=metric_name)
        m = self.metrics[metric_name]
        m.append(value)
        if max_length is not None and len(m) > max_length:
            # self.metrics[metric_name] = m[-max_length:]
            del m[:len(m)-max_length]
    def mean(self, metric_name: str) -> Union[float, np.ndarray[float]]:
        assert metric_name in self.metrics
        if len(self.metrics[metric_name]) == 0:
            return np.nan
        m = self.metrics[metric_name]
        if isinstance(m[0], float):
            return np.mean(m)
        return np.mean(m, axis=0)
    def std(self, metric_name: str) -> Union[float, np.ndarray[float]]:
        assert metric_name in self.metrics
        if len(self.metrics[metric_name]) == 0:
            return np.nan
        m = self.metrics[metric_name]
        if isinstance(m[0], float):
            return np.std(m)
        return np.std(m, axis=0)
    def five_number_summary(self, metric_name: str) -> Union[float, tuple[float, float, float, float, float], np.ndarray[float]]:
        assert metric_name in self.metrics
        if len(self.metrics[metric_name]) == 0:
            return np.nan
        m = self.metrics[metric_name]
        if isinstance(m[0], float):
            return np.min(m), np.percentile(m, 25), np.median(m), np.percentile(m, 75), np.max(m)
        arr = np.empty((len(m), 5), dtype=float)
        for i in range(arr.shape[0]):
            _m = m[i]
            arr[i] = np.min(_m), np.percentile(_m, 25), np.median(_m), np.percentile(_m, 75), np.max(m)
        return arr
    def save(self, fpath: str):
        np.savez(fpath, **self.metrics)
    def load(self, fpath: str):
        d = np.load(fpath)
        for k, v in d.items():
            self.metrics[k] = v
    def extend(self, stats: 'Stats'):
        '''
        Modify in-place.
        '''
        for metric_name, data in stats.metrics.items():
            if metric_name in self.metrics:
                self.metrics[metric_name].extend(data)
            else:
                self.metrics[metric_name] = data[:]
    def __getitem__(self, key: any) -> any:
        if not isinstance(key, str):
            return None
        if key in self.metrics:
            return self.metrics[key]
    def __setitem__(self, key: any, value: any) -> any:
        raise Exception(f'__setitem__() is not supported')
    def __contains__(self, key: any) -> bool:
        return self.metrics.__contains__(key)
