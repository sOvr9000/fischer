
import numpy as np



__all__ = ['gray_to_rgb', 'rescale_to_255', 'rescale_to_1', 'reverse_channels', 'upscale']

def gray_to_rgb(arr: np.ndarray) -> np.ndarray:
    '''
    Convert an array of shape `(h, w, 1)` or `(h, w)` to an array of shape `(h, w, 3)` by interpreting it as gray values and duplicating the three values to construct an array of RGB values corresponding to the equivalent gray values.  The returned array is always of dtype `float`.
    '''
    if len(arr.shape) == 2:
        return np.repeat(np.expand_dims(arr, axis=2), repeats=3, axis=2).astype(float)
    if len(arr.shape) == 3 and arr.shape[2] == 1:
        return np.repeat(arr, repeats=3, axis=2).astype(float)
    raise TypeError(f'The passed array must be a 2D, or a 3D array with the third dimension having size 1.')

def rescale_to_255(arr: np.ndarray, dtype = int) -> np.ndarray:
    '''
    Given `arr` with values existing in the interval [0, 1], return an array with the values interpolated to the interval [0, 255], casted to the dtype `dtype`.
    '''
    return (arr * 255).astype(dtype)

def rescale_to_1(arr: np.ndarray) -> np.ndarray:
    '''
    Given `arr` with values existing in the interval [0, 255], return an array with the values interpolated to the interval [0, 1].
    '''
    return arr * 0.0039215686274509803921568627451

def reverse_channels(arr: np.ndarray) -> np.ndarray:
    '''
    Reverse the RGB channels of `arr` to BGR, which is the expected order of channels from the module `cv2`.
    '''
    return arr[:, :, ::-1]

def upscale(arr: np.ndarray, scaling: int) -> np.ndarray:
    '''
    Scale up the "pixel size" of the given array.
    '''
    return np.repeat(np.repeat(arr, repeats=scaling, axis=1), repeats=scaling, axis=0)
