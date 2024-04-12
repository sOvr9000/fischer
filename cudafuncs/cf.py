

from numba.cuda import jit

__all__ = ['lerp', 'gerp', 'cerp', 'clamp']


@jit(device=True)
def lerp(a, b, t):
    '''
    Linear interpolation.
    '''
    return a*(1-t)+b*t

@jit(device=True)
def gerp(a, b, t):
    '''
    Geometric interpolation.

    G(a, b, t) = a^(1-t)b^t
    '''
    return a**(1-t)*b**t

@jit(device=True)
def clamp(v, v_min, v_max):
    '''
    Clamp a value between two values.
    '''
    if v < v_min:
        return v_min
    if v > v_max:
        return v_max
    return v

@jit(device=True)
def cerp(a, b, t):
    '''
    Cubic interpolation.
    '''
    return lerp(a, b, t*t*(3-2*t))



