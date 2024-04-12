


import numpy as np
from numba import cuda
from math import ceil, floor
from cmath import log, sin, tan


__all__ = ['detect_cycles']

@cuda.jit(device=True)
def sqr_length(z):
    return z.real * z.real + z.imag * z.imag

@cuda.jit(device=True)
def round_complex(z, x, p):
    lowest_k = complex(floor(z.real), floor(z.imag))
    lowest_dist = sqr_length(x-p*lowest_k)

    k = complex(floor(z.real), ceil(z.imag))
    dist = sqr_length(x-p*k)
    if dist < lowest_dist:
        lowest_dist = dist
        lowest_k = k

    k = complex(ceil(z.real), floor(z.imag))
    dist = sqr_length(x-p*k)
    if dist < lowest_dist:
        lowest_dist = dist
        lowest_k = k

    k = complex(ceil(z.real), ceil(z.imag))
    dist = sqr_length(x-p*k)
    if dist < lowest_dist:
        lowest_dist = dist
        lowest_k = k

    return lowest_k

@cuda.jit(device=True)
def almost_equal_index(l, len_l, x):
    for i in range(len_l-1, -1, -1):
        if sqr_length(l[i] - x) <= 1e-11 * 1e-11:
            return i
    return -1

@cuda.jit(device=True)
def lerp(a, b, t):
    return a*(1-t) + b*t

@cuda.jit(device=True)
def gerp(a, b, t):
    return a**(1-t) * b**t

@cuda.jit
def _cuda_kernel_detect_cycles(xs, bases, cycle_lengths, max_digits):
    i = cuda.grid(1)
    if i >= xs.shape[0]:
        return
    x0 = xs[i]
    x = x0
    xn = x
    base = bases[i]
    if sqr_length(base) <= 1:
        return
    inv_base = 1 / base
    arr = cuda.local.array(MAX_ITERATIONS, np.complex_)
    for t in range(MAX_ITERATIONS):
        exponent = int(floor(abs(log(x) / log(base)) + 0.5))
        if sqr_length(base**(exponent+1)-x) < sqr_length(base**exponent-x):
            exponent += 1
        if sqr_length(base**exponent) > sqr_length(x):
            exponent -= 1
        p = base ** exponent
        s = 0j
        for _ in range(max_digits):
            d = round_complex(x / p, x, p)
            x -= p * d
            if d == 0:
                s += p
                # s = 0
                # break
            else:
                s += p * lerp(tan(d), sin(d), 0.5) / d
            p *= inv_base
            if x == 0:
                break
        xn = s
        x = xn
        _i = almost_equal_index(arr, t, x)
        if _i != -1:
            cycle_lengths[i] = t - _i
            return
        arr[t] = x



def detect_cycles(bases: np.ndarray, xs: np.ndarray) -> np.ndarray:
    shape = bases.shape
    if len(shape) > 1:
        bases = bases.reshape((-1,))
        xs = xs.reshape((-1,))
    cycle_lengths = np.zeros((len(xs),))
    _cycle_lengths = cuda.to_device(cycle_lengths)
    _xs = cuda.to_device(xs)
    _bases = cuda.to_device(bases)
    _cuda_kernel_detect_cycles[xs.shape[0]//32, 32](_xs, _bases, _cycle_lengths, 32)
    cycle_lengths = _cycle_lengths.copy_to_host()
    if len(shape) > 1:
        cycle_lengths = cycle_lengths.reshape(shape)
    return cycle_lengths


MAX_ITERATIONS = 8192

def main():
    # N = 1024
    # x0s = np.exp(np.random.uniform(0, np.pi * 2, (N,)) * 1j)
    # bs = np.random.normal(5, 1, (N,)) * np.exp(np.random.uniform(0, np.pi * 2, (N,)) * 1j)

    resolution = 8
    region = 2+2j, 3+3j
    chunk_size = 8

    N = resolution * resolution
    chunk_rows = resolution // chunk_size
    chunk_cols = resolution // chunk_size
    total_chunks = chunk_rows * chunk_cols
    x0 = 1j
    x0s = np.array([x0] * N)
    bs = np.array([complex(region[0].real + (x_chunk * chunk_size + x) * (region[1].real - region[0].real) / resolution, region[0].imag + (y_chunk * chunk_size + y) * (region[1].imag - region[0].imag) / resolution) for y_chunk in range(chunk_rows) for x_chunk in range(chunk_cols) for y in range(chunk_size) for x in range(chunk_size)], dtype=complex)

    cycle_lengths = detect_cycles(bs, x0s)
    print(cycle_lengths)

if __name__ == '__main__':
    main()






