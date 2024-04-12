

import numpy as np
from numba import cuda
from math import ceil, floor
from cmath import log



__all__ = ['complex_decimal_to_base', 'complex_base_to_decimal']

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

@cuda.jit
def _cuda_kernel_complex_decimal_to_base(xs, bases, digits, exponents):
    i = cuda.grid(1)
    if i >= xs.shape[0]:
        return
    x = xs[i]
    base = bases[i]
    exponent = int(floor(abs(log(x) / log(base)) + 0.5))
    if abs(base**(exponent+1)-x) < abs(base**exponent-x):
        exponent += 1
    p = base ** exponent
    for j in range(digits.shape[1]):
        d = round_complex(x / p, x, p)
        x -= p * d
        p /= base
        digits[i, j] = d
    exponents[i] = exponent

@cuda.jit
def _cuda_kernel_complex_base_to_decimal(bases, digits, exponents, decimal_values):
    i = cuda.grid(1)
    if i >= bases.shape[0]:
        return
    base = bases[i]
    x = 0j
    inv_base = 1. / base
    p = base ** exponents[i]
    for j in range(digits.shape[1]):
        x += digits[i, j] * p
        p *= inv_base
    decimal_values[i] = x

def complex_decimal_to_base(bases: np.ndarray, xs: np.ndarray, max_digits: int = 32) -> np.ndarray:
    digits = np.zeros((len(xs), max_digits), dtype=complex)
    exponents = np.zeros((len(xs),))
    _digits = cuda.to_device(digits)
    _exponents = cuda.to_device(exponents)
    _xs = cuda.to_device(xs)
    _bases = cuda.to_device(bases)
    _cuda_kernel_complex_decimal_to_base[xs.shape[0]//32, 32](_bases, _xs, _digits, _exponents)
    digits = _digits.copy_to_host()
    exponents = _exponents.copy_to_host()
    return digits, exponents

def complex_base_to_decimal(bases: np.ndarray, digits: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    _bases = cuda.to_device(bases)
    _digits = cuda.to_device(digits)
    _exponents = cuda.to_device(exponents)
    _decimal_values = cuda.to_device(np.zeros((bases.shape[0],), dtype=complex))
    _cuda_kernel_complex_base_to_decimal[bases.shape[0]//32, 32](_bases, _digits, _exponents, _decimal_values)
    decimal_values = _decimal_values.copy_to_host()
    return decimal_values



def main():
    N = 65536
    xs = np.exp(np.random.uniform(0, np.pi * 2, (N,)) * 1j)
    bases = np.random.normal(5, 1, (N,)) * np.exp(np.random.uniform(0, np.pi * 2, (N,)) * 1j)
    digits, exponents = complex_decimal_to_base(xs, bases,
        max_digits=32
    )
    print(digits, exponents)
    values = complex_base_to_decimal(bases, digits, exponents)
    print(values)
    print(xs)

if __name__ == '__main__':
    main()




