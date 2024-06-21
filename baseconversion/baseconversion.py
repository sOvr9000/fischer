
import numpy as np
import mpmath as mp
from typing import Union, Sequence, Optional
from itertools import product



__all__ = ['base_to_decimal', 'decimal_to_base', 'complex_base_to_decimal', 'complex_decimal_to_base']



def base_to_decimal(base: Union[int, float, str, mp.mpf], digits: Sequence[int], exponent: int) -> mp.mpf:
    '''
    Return the value of the given `digits` and `exponent` in base `base` converted back to decimal notation.
    '''
    prec = int(len(digits) * mp.log(base) / mp.log(2)) + 4
    with mp.workprec(prec):
        if isinstance(base, (int, float, str)):
            base = mp.mpf(base)
        x = mp.mpf(0)
        inv_base = 1. / base
        p = base ** exponent
        for digit in digits:
            x += digit * p
            p *= inv_base
        return x

def decimal_to_base(base: Union[int, float, str, mp.mpf], dec: Union[int, float, str, mp.mpf], max_digits: int = 64, pad: bool = False) -> tuple[list[int], int]:
    '''
    Return `digits` and `exponent` derived from converting the decimal number `dec` to base `base`.

    If `pad = True`, then expand the length of the returned `digits` list to have a length of `max_digits` if it is shorter than that.
    ```
    base = 2
    dec = 27
    digits, exponent = decimal_to_base(base, dec, max_digits=7, pad=True)
    # digits = [1, 1, 0, 1, 1, 0, 0]
    # exponent = 4
    # digits[0] corresponds to base ** exponent = 16
    num_str = ''.join(map(str, digits[:exponent+1] + ['.'] + digits[exponent+1:]))
    print(num_str)
    # 11011.00
    ```
    '''
    if dec == 0:
        if pad:
            return [0] * max_digits, 0
        return [0], 0
    prec = int(max_digits * mp.log(base) / mp.log(2)) + 4
    with mp.workprec(prec):
        if isinstance(dec, (int, float, str)):
            dec = mp.mpf(dec)
        if isinstance(base, (int, float, str)):
            base = mp.mpf(base)
        digits = []
        exponent = int(mp.floor(mp.log(dec) / mp.log(base)))
        p = base ** exponent
        x = dec
        while x > 0 and len(digits) < max_digits:
            d = int(x / p)
            x -= p * d
            p /= base
            digits.append(d)
        if pad:
            if len(digits) < max_digits:
                digits.extend([0] * (max_digits - len(digits)))
        return digits, exponent

def round_complex(a: Union[float, mp.mpf, complex, mp.mpc], b: Optional[Union[float, mp.mpf]] = None, target_residuals: tuple[mp.mpc, mp.mpc] = None) -> Union[complex, tuple[int, int]]:
    if target_residuals is None:
        if isinstance(a, (complex, mp.mpc)):
            return complex(int(a.real + .5), int(a.imag + .5))
        return int(a + .5), int(b + .5)
    else:
        x, p = target_residuals
        # round a+bi such that x - p * (a+bi) has least absolute value
        if isinstance(a, (complex, mp.mpc)):
            return complex(*round_complex(a.real, a.imag, target_residuals=target_residuals))
        avs = []
        for ra, rb in product((mp.floor, mp.ceil), (mp.floor, mp.ceil)):
            z = complex(int(ra(a)), int(rb(b)))
            avs.append((abs(x-p*z), z.real, z.imag))
        mz = min(avs)
        return mz[1], mz[2]

def floor_complex(a: Union[float, mp.mpf, complex, mp.mpc], b: Optional[Union[float, mp.mpf]] = None) -> Union[complex, tuple[int, int]]:
    '''
    Round the real and imaginary components toward zero.
    '''
    if isinstance(a, (complex, mp.mpc)):
        return complex(int(a.real), int(a.imag))
    return int(a), int(b)

def complex_base_to_decimal(base: Union[int, float, complex, str, mp.mpf, mp.mpc], digits: Sequence[tuple[complex]], exponent: int, prec: Optional[int] = None) -> mp.mpc:
    '''
    Return the value of the given `digits` and `exponent` in base `base` converted back to decimal notation.
    '''
    if prec is None:
        prec = max(24, int(len(digits) * abs(mp.log(base) / mp.log(2))) + 4)
    with mp.workprec(prec):
        if isinstance(base, (int, float, complex, str, mp.mpf)):
            base = mp.mpc(base)
        x = mp.mpc(0, 0)
        inv_base = 1. / base
        p = base ** exponent
        k = exponent
        for digit in digits:
            x += digit * p
            # if base == 2+2j:
            #     print('Digit:', digit.real, digit.imag, '| Exponent of base:', k, '| Current sum:', x.real, x.imag)
            p *= inv_base
            k -= 1
        return x

def complex_decimal_to_base(base: Union[int, float, complex, str, mp.mpf, mp.mpc], dec: Union[int, float, complex, str, mp.mpf, mp.mpc], max_digits: int = 64) -> tuple[list[complex], int]:
    '''
    Return `digits` and `exponent` derived from converting the decimal number `dec` to base `base`.
    '''
    prec = int(max_digits * abs(mp.log(base) / mp.log(2))) + 4
    with mp.workprec(prec):
        if isinstance(dec, (int, float, complex, str, mp.mpf)):
            dec = mp.mpc(dec)
        if isinstance(base, (int, float, complex, str, mp.mpf)):
            base = mp.mpc(base)
        digits = []
        x = dec
        # exponent = int(mp.floor(abs(mp.log(dec) / mp.log(base)) + .5))
        exponent = int(mp.floor(abs(mp.log(dec) / mp.log(base)) + .5))
        if abs(base**(exponent+1)-x) < abs(base**exponent-x):
            exponent += 1
        if abs(base**exponent) > abs(x):
            exponent -= 1
        p = base ** exponent
        while x != 0 and len(digits) < max_digits:
            d = round_complex(x / p, target_residuals=(x, p))
            x -= p * d
            p /= base
            digits.append(d)
        return digits, exponent


if __name__ == '__main__':
    # print(decimal_to_base(144, mp.pi))
    # print(base_to_decimal(144, *decimal_to_base(144, mp.pi)))

    # print(complex_decimal_to_base(mp.mpc(mp.e * -2, 1), mp.mpc(mp.pi * 24, 14)))
    # print(complex_base_to_decimal(mp.mpc(mp.e * -2, 1), *complex_decimal_to_base(mp.mpc(mp.e * -2, 1), mp.mpc(mp.pi * 24, 14))))

    import numpy as np
    N = 65536
    xs = np.exp(np.random.uniform(0, np.pi * 2, (N,)) * 1j)
    bases = np.random.normal(5, 1, (N,)) * np.exp(np.random.uniform(0, np.pi * 2, (N,)) * 1j)
    for n in range(N):
        digits, exponent = complex_decimal_to_base(xs[n], bases[n],
            max_digits=32
        )
    # print(digits, exponents)

