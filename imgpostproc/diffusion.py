

import numpy as np
from numba import cuda


@cuda.jit(device=True)
def lerp(a, b, t):
    return a*(1-t) + b*t

@cuda.jit(device=True)
def lerp_delta(a, b, t):
    return (b-a)*t

@cuda.jit
def _cuda_kernel_diffuse_step(arr, new_arr, factor):
    i, j = cuda.grid(2)
    if abs(arr[i, j, 0]) < 1e-3 and abs(arr[i, j, 1]) < 1e-3 and abs(arr[i, j, 2]) < 1e-3: return
    r = arr[i, j, 0]
    g = arr[i, j, 1]
    b = arr[i, j, 2]
    new_arr[i, j, 0] = r
    new_arr[i, j, 1] = g
    new_arr[i, j, 2] = b
    cuda.syncthreads()
    s = 0.
    if i > 0:
        cuda.atomic.add(new_arr, (i-1, j, 0), lerp_delta(arr[i-1, j, 0], r, 1-factor))
        cuda.atomic.add(new_arr, (i-1, j, 1), lerp_delta(arr[i-1, j, 1], g, 1-factor))
        cuda.atomic.add(new_arr, (i-1, j, 2), lerp_delta(arr[i-1, j, 2], b, 1-factor))
        s += 1.
    if j > 0:
        cuda.atomic.add(new_arr, (i, j-1, 0), lerp_delta(arr[i, j-1, 0], r, 1-factor))
        cuda.atomic.add(new_arr, (i, j-1, 1), lerp_delta(arr[i, j-1, 1], g, 1-factor))
        cuda.atomic.add(new_arr, (i, j-1, 2), lerp_delta(arr[i, j-1, 2], b, 1-factor))
        s += 1.
    if i+1 < arr.shape[0]:
        cuda.atomic.add(new_arr, (i+1, j, 0), lerp_delta(arr[i+1, j, 0], r, 1-factor))
        cuda.atomic.add(new_arr, (i+1, j, 1), lerp_delta(arr[i+1, j, 1], g, 1-factor))
        cuda.atomic.add(new_arr, (i+1, j, 2), lerp_delta(arr[i+1, j, 2], b, 1-factor))
        s += 1.
    if j+1 < arr.shape[1]:
        cuda.atomic.add(new_arr, (i, j+1, 0), lerp_delta(arr[i, j+1, 0], r, 1-factor))
        cuda.atomic.add(new_arr, (i, j+1, 1), lerp_delta(arr[i, j+1, 1], g, 1-factor))
        cuda.atomic.add(new_arr, (i, j+1, 2), lerp_delta(arr[i, j+1, 2], b, 1-factor))
        s += 1.
    cuda.syncthreads()
    new_arr[i, j, 0] /= s
    new_arr[i, j, 1] /= s
    new_arr[i, j, 2] /= s

def diffuse_step(img: np.ndarray, diffusion_factor: float = 0.01) -> np.ndarray:
    '''
    Perform a single diffusion step on `img`.  Any pixel that is almost exactly `negative_color` (within `1e-3` on all channels) does not diffuse its color into adjacent pixels.

    This effect can be used to give high-contrast lines a neon-like glow to them.
    '''
    _img = cuda.to_device(img)
    _new_img = cuda.to_device(np.zeros(img.shape))
    _cuda_kernel_diffuse_step[(img.shape[0]//32, img.shape[1]//32), (32, 32)](_img, _new_img, diffusion_factor)
    img = _new_img.copy_to_host()
    return img






