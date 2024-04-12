

import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from keras.callbacks import History



def smooth_data(d: np.ndarray, factor: int) -> np.ndarray:
    '''
    Smooth data by some factor.
    '''
    assert factor >= 1, 'factor is too small'
    assert d.shape[0] - factor + 1 >= 1, f'd is not long enough to smooth with a factor of {factor}'
    if factor == 1:
        return d.copy()
    if len(d.shape) == 1:
        sd = np.zeros(d.shape[0] - factor + 1, dtype=float)
    else:
        sd = np.zeros((d.shape[0] - factor + 1, *d.shape[1:]), dtype=float)
    for k in range(sd.shape[0]):
        sd[k] = np.mean(d[k:k+factor], axis=0)
    return sd

def plot(d: Union[list, np.ndarray, History], title: str = None, x_label: str = None, y_label: str = None, smoothing: Union[int, float] = 1, plot_type: str = 'line'):
    '''
    Plot data with some common and convenient features and functionality.
    '''
    if isinstance(d, History):
        for key, values in d.history.items():
            if title is None:
                t = key
            else:
                t = f'{title} -- {key}'
            plot(values, title=t, x_label=x_label, y_label=y_label, smoothing=smoothing, plot_type=plot_type)
        return
    if isinstance(d, list):
        d = np.array(d)
    if isinstance(smoothing, float):
        smoothing = max(1, int(d.shape[0] * smoothing + .5))
    d = smooth_data(d, smoothing)
    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if plot_type == 'line':
        plt.plot(d)
    elif plot_type == 'scatter':
        plt.scatter(d)
    else:
        raise Exception(f'Unrecognized plot_type: {plot_type}')
    plt.show()


