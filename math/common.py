
import numpy as np
from math import exp, sin, cos, pi, acos, atan2
from typing import Union, Hashable
from colorsys import hsv_to_rgb as hsv2rgb, rgb_to_hsv as rgb2hsv
import random

tau = 2 * pi

def quadratic_roots(a: float, b: float, c: float, allow_complex: bool = True) -> Union[Union[tuple[float, float], tuple[complex, complex]], float, None]:
    '''
    Find the roots of a quadratic polynomial of the form p(x) = ax^2 + bx + c.

    Returns None when no real roots exist and `allow_complex = False`.  Otherwise, return a two-tuple of either both `float` or both `complex` types, depending on the existence of real roots.

    If exactly one real root exists, then return a `float` type.
    '''
    if a == 0:
        raise ValueError(f'Cannot find quadratic roots of polynomial with leading coefficient of zero (a = 0).')
    z = b * b - 4 * a * c
    if z < 0:
        if allow_complex:
            z = complex(z) ** .5
            inv = 1. / (2 * a)
            return (-z - b) * inv, (z - b) * inv
        else:
            return None
    if z == 0:
        return b / (-2 * a)
    z = z ** .5
    inv = 1. / (2 * a)
    return (-z - b) * inv, (z - b) * inv

def greater_quadratic_root(a: float, b: float, c: float) -> Union[float, None]:
    '''
    Similar to `quadratic_roots()`, but only return the real root which is larger than or equal to the other root.  Return None if there are no real roots.
    '''
    roots = quadratic_roots(a, b, c, allow_complex=False)
    if roots is None:
        return None
    if isinstance(roots, float):
        return roots
    return roots[1]

def smaller_quadratic_root(a: float, b: float, c: float) -> Union[float, None]:
    '''
    Similar to `quadratic_roots()`, but only return the real root which is smaller than or equal to the other root.  Return None if there are no real roots.
    '''
    roots = quadratic_roots(a, b, c, allow_complex=False)
    if roots is None:
        return None
    if isinstance(roots, float):
        return roots
    return roots[0]

def complex_quadratic_roots(a: complex, b: complex, c: complex) -> tuple[complex, complex]:
    '''
    Return the complex quadratic roots.  No edge case exists except for receiving `a = 0`.
    '''
    if a == 0:
        raise ValueError(f'Cannot find quadratic roots of polynomial with leading coefficient of zero (a = 0).')
    return ((b * b - 4 * a * c) ** .5 - b) / (2 * a)

# @jit
def poly_area(vertices: np.ndarray[float]) -> float:
    '''
    Calculate the area of a polygon with the vertices `vertices`, an array of shape `(N, 2)` where `N` is the number of vertices in the polygon.

    The shoelace method is used, which means the sides of the polygon must not intersect each other.
    '''
    v = np.sum(vertices * vertices[np.arange(-1, vertices.shape[0]-1), ::-1], axis=0)
    return .5 * abs(v[0] - v[1])

def random_clamped_partition(partition_size: int, total: int, lower_bound: int = 0, upper_bound: int = None) -> np.ndarray[int]:
    '''
    `partition_size` is the length of the returned array.

    `total` is the sum of the integers in the returned array.

    If `upper_bound = None`, then it takes the value of `total`.

    The returned array is randomly populated with integers between `lower_bound` (inclusive) and `upper_bound` (inclusive).
    '''
    if upper_bound is None:
        upper_bound = total
    assert isinstance(total, int)
    assert total >= 0
    assert isinstance(upper_bound, int)
    assert upper_bound >= 0 and upper_bound <= total
    assert isinstance(lower_bound, int)
    assert lower_bound >= 0 and lower_bound <= total
    assert lower_bound <= upper_bound
    assert isinstance(partition_size, int)
    assert partition_size > 0
    assert total <= partition_size * upper_bound
    assert total >= partition_size * lower_bound

    # total = N
    # partition_size = 1
    # lower = 0
    # upper = total
    # possible: 1
    
    # total = 2
    # partition_size = 2
    # lower = 0
    # upper = total
    # possible: 1
    



    x = np.abs(np.random.normal(0, 1, size=partition_size))
    x *= total / x.sum()
    N = (x + .5).astype(int)
    s = N.sum()
    d = N - x
    if s < total:
        for _ in range(total - s):
            j = np.argmin(d)
            N[j] += 1
            d[j] = .5 - d[j]
    elif s > total:
        for _ in range(s - total):
            j = np.argmax(d)
            N[j] -= 1
            d[j] = .5 - d[j]
    d[N < lower_bound] = np.inf
    for _ in range(partition_size * lower_bound):
        for i in range(num_shapes):
            if N[i] < lower_bound:
                j = np.argmin(d)
                N[i] += 1
                N[j] -= 1
                if N[j] <= lower_bound:
                    d[j] = np.inf
                else:
                    d[j] -= 1
                break
        else:
            break
    d[:] = N - x
    d[N > upper_bound] = np.inf
    for _ in range(partition_size * upper_bound):
        for i in range(num_shapes):
            if N[i] > upper_bound:
                j = np.argmin(d)
                N[i] += 1
                N[j] -= 1
                if N[j] <= lower_bound:
                    d[j] = np.inf
                else:
                    d[j] -= 1
                break
        else:
            break
    return N

def glerp(a: float, b: float, t: float) -> float:
    '''
    Geometric interpolation.
    '''
    return a ** (1 - t) * b ** t

def scaled_glerp(a: complex, b: complex, t: complex) -> complex:
    '''
    S(a,b,t) = G(a,b,t) * L(|a|,|b|,t) / |G(a,b,t)|

    Identical to linear interpolation on real numbers.  On complex numbers, the absolute value is scaled linearly as well as argument, unlike `glerp()` where only the argument is scaled linearly.
    '''
    g = glerp(a, b, t)
    if g == 0:
        return 0
    l = lerp(abs(a), abs(b), t)
    return g * l / abs(g)

def lerp(a: float, b: float, t: float) -> float:
    return a*(1-t) + b*t

def inv_lerp(value: float, v_min: float, v_max: float) -> float:
    return (value - v_min) / (v_max - v_min)

def remap(value: float, v_min: float, v_max: float, target_min: float, target_max: float) -> float:
    return lerp(target_min, target_max, inv_lerp(value, v_min, v_max))

def remap_complex(value: complex, v_min: complex, v_max: complex, target_min: complex, target_max: complex) -> complex:
    return complex(remap(value.real, v_min.real, v_max.real, target_min.real, target_max.real), remap(value.imag, v_min.imag, v_max.imag, target_min.imag, target_max.imag))

def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))

def lerp_angle(a: float, b: float, t: float, max_delta: float = 4) -> float:
    '''
    Return a new angle that is interpolated between angles `a` and `b` by the inerpolant `t` and is within the shorter arc between the points at angles `a` and `b`.

    `max_delta` is the maximum angle by which the returned angle is incremented/decremented from `a` toward `b`.
    '''
    a %= tau
    b %= tau
    delta = (b - a + int(abs(a-b) > np.pi) * (2 * int(a > b) - 1) * np.pi) * t
    if abs(delta) > max_delta:
        if delta < 0:
            delta = -max_delta
        else:
            delta = max_delta
    return (a + delta) % tau

def cubic_lerp(a: float, b: float, t: float) -> float:
    '''
    Return the smooth interpolation from `a` to `b` which follows a cubic curve.
    '''
    return lerp(a, b, t*t*(3-2*t))

def sin_lerp(a: float, b: float, t: float) -> float:
    '''
    Return the smooth interpolation from `a` to `b` which follows a sinusoidal curve.
    '''
    return lerp(a, b, 0.5 - 0.5 * cos(t * pi))

def cmp(a:float, b:float) -> int:
    '''
    Return 1 if `a > b`, 0 if `a == b`, or -1 if `a < b`.
    '''
    return int(a > b) - int(a < b)

def sign(x:float) -> int:
    '''
    Return 1 if `x` is positive, 0 if `x` is zero, or -1 if `x` is negative.
    '''
    return cmp(x, 0)

def sq(x: float) -> float:
    '''
    Return the square of `x`.
    '''
    return x * x


def hsv_to_rgb(h:float, s:float, v:float) -> tuple[int, int, int]:
    '''
    h, s, and v must be floats 0-1

    Return r, g, and b as integers 0-255
    '''
    r, g, b = hsv2rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def rgb_to_hsv(r:int, g:int, b:int) -> tuple[float, float, float]:
    '''
    r, g, and b can be integers 0-255, but can also be floats 0-255.

    Return h, s, and v as floats 0-1
    '''
    return rgb2hsv(r*0.0039215686274509803921568627451, g*0.0039215686274509803921568627451, b*0.0039215686274509803921568627451)


def color_modify(c:tuple[int, int, int], hue:float=0., saturation:float=0., value:float=0.) -> tuple[int, int, int]:
    '''
    hue is added to the hue of c.

    If saturation < 0, then scale the saturation of c by the percentage (-saturation) toward zero.
    If saturation > 0, then scale the saturation of c by the percentage (saturation) toward one.
    If saturation = 0, then the saturation of c is left unchanged.

    If value < 0, then scale the value of c by the percentage (-value) toward zero.
    If value > 0, then scale the value of c by the percentage (value) toward one.
    If value = 0, then the value of c is left unchanged.
    '''
    h, s, v = rgb2hsv(*c)
    target_saturation = int(saturation >= 0)
    target_value = int(value >= 0)
    return hsv2rgb(
        h + hue,
        lerp(s, target_saturation, saturation * (2*target_saturation-1)),
        lerp(v, target_value, value * (2*target_value-1))
    )


def color_distort(c:tuple[int, int, int], hue:float=0., saturation:float=0., value:float=.1) -> tuple[int, int, int]:
    '''
    Call color_modify() with randomized parameters based on hue, saturation, and value.
    '''
    return color_modify(
        c,
        np.random.uniform(-hue, hue),
        np.random.random(-saturation, saturation),
        np.random.random(-value, value)
    )


def color_inverse(c: tuple[int, int, int]) -> tuple[int, int, int]:
    '''
    Return the additive inverse of the given color `c`, which is assumed to be in RGB format with integers between 0 and 255.
    '''
    r, g, b = c
    return 255-r, 255-g, 255-b


def point_cluster(n:int, density:int=8, seed:Hashable=None) -> list[tuple[float, float]]:
    '''
    Generate `n` 2D points where each point is at least 1 unit away from other points by Euclidean distance.

    `density` is the number of bisection iterations to determine how closely the points will be packed together while retaining the minimum distance constraint.
    '''
    if seed is not None:
        rng = random.Random()
        rng.seed(hash(seed))
        rand = rng.random
    else:
        rand = random
    r = rand()
    t = rand() * tau
    p = [(r * cos(t), r * sin(t))]
    for _ in range(n):
        mx, my = np.mean(p, axis=0)
        r, t = n, atan2(-my, -mx) + rand()*.125-.0625
        cs, sn = cos(t), sin(t)
        a, b = 0, r
        for _ in range(density):
            npx, npy = r * cs, r * sn
            if any((npx - px) * (npx - px) + (npy - py) * (npy - py) < 1 for px, py in p):
                a = r
                r = (r + b) * .5
            else:
                b = r
                r = (r + a) * .5
        p.append((npx, npy))
    return p


def polar_point(r:float, theta:float) -> tuple[float, float]:
    '''
    Calculate the point (x, y) which lies on a circle (centered at the origin) of radius `r` at the angle `theta`.
    '''
    return r * cos(theta), r * sin(theta)


def rotated_point(x:float, y:float, theta:float, pivot_x:float = 0, pivot_y:float = 0) -> tuple[float, float]:
    '''
    Return the point `(x, y)` rotated about the point `(pivot_x, pivot_y)` by `theta` (which is measured in radians).
    '''
    sn = sin(theta)
    cs = cos(theta)
    rx, ry = np.matmul([[cs, -sn], [sn, cs]], [x - pivot_x, y - pivot_y])
    return pivot_x + rx, pivot_y + ry


def normalized_vector(x:float, y:float, length:float=1, default:tuple[float, float]=(1, 0)) -> tuple[float, float]:
    '''
    Return the vector `(x, y)` normalized to have a length of `length`.

    If `(x, y) == (0, 0)`, then return `default`.
    '''
    if x == 0 and y == 0:
        return default
    inv_cur_length = 1 / (x * x + y * y) ** .5
    if length == 1:
        return x * inv_cur_length, y * inv_cur_length
    else:
        return x * length * inv_cur_length, y * length * inv_cur_length


def vector_add(x0:float, y0:float, x1:float, y1:float) -> tuple[float, float]:
    '''
    Return a new vector which adds `(x0, y0)` to `(x1, y1)`.
    '''
    return x0 + x1, y0 + y1


def vector_subtract(x0:float, y0:float, x1:float, y1:float) -> tuple[float, float]:
    '''
    Return a new vector which subtracts `(x1, y1)` from `(x0, y0)`.
    '''
    return x0 - x1, y0 - y1


def vector_multiply(x:float, y:float, v:float) -> tuple[float, float]:
    '''
    Return a new vector which multiplies `(x, y)` by `v`.
    '''
    return x * v, y * v


def vector_magnitude(x:float, y:float) -> float:
    '''
    Return the length of the vector `(x, y)`.
    '''
    return (x*x+y*y)**.5


def vector_sq_magnitude(x:float, y:float) -> float:
    '''
    Return the squared length of the vector `(x, y)`.
    '''
    return x*x+y*y

def l2_distance(x0:float, y0:float, x1:float, y1:float) -> float:
    '''
    Return the Euclidean distance between points `(x0, y0)` and `(x1, y1)`.
    '''
    return (sq(x1 - x0) + sq(y1 - y0)) ** .5


def sq_l2_distance(x0:float, y0:float, x1:float, y1:float) -> float:
    '''
    Return the squared Euclidean distance between points `(x0, y0)` and `(x1, y1)`.

    This is useful in particular when comparing distances so that square roots can be avoided in calculations.
    '''
    return sq(x1 - x0) + sq(y1 - y0)

def l1_distance(x0: int, y0: int, x1: int, y1: int) -> int:
    '''
    Return the Manhattan distance between points `(x0, y0)` and `(x1, y1)`.
    '''
    return abs(x0 - x1) + abs(y0 - y1)

def project_vector(v0x:float, v0y:float, v1x:float, v1y:float) -> tuple[float, float]:
    '''
    Project vector `(v0x, v0y)` onto vector `(v1x, v1y)`.
    '''
    return scale_vector(v1x, v1y, dot_product(v0x, v0y, v1x, v1y) / dot_product(v1x, v1y, v1x, v1y))


def dot_product(v0x:float, v0y:float, v1x:float, v1y:float) -> float:
    '''
    Return the dot product of vectors `(v0x, v0y)` and `(v1x, v1y)`.
    '''
    return v0x * v1x + v0y * v1y


def scale_vector(x:float, y:float, scalar:float) -> tuple[float, float]:
    '''
    Scale the vector `(x, y)` by `scalar`.
    '''
    return x * scalar, y * scalar


def point_orientation(x0:float, y0:float, x1:float, y1:float, x2:float, y2:float) -> int:
    return cmp((x2 - x1) * (y1 - y0), (x1 - x0) * (y2 - y1))


def segments_intersect(
    x0:float, y0:float, x1:float, y1:float,
    x2:float, y2:float, x3:float, y3:float,
    # check_endpoints=True,
) -> bool:
    '''
    Given endpoints of two line segments, return whether the segments intersect each other.
    '''
    return point_orientation(x0, y0, x1, y1, x2, y2) != point_orientation(x0, y0, x1, y1, x3, y3) and point_orientation(x2, y2, x3, y3, x0, y0) != point_orientation(x2, y2, x3, y3, x1, y1)


def line_intersection_point(
    x0:float, y0:float, x1:float, y1:float,
    x2:float, y2:float, x3:float, y3:float,
) -> tuple[float, float]:
    '''
    Given two pairs of points `{(x0, y0), (x1, y1)}` and `{(x2, y2), (x3, y3)}` which define two lines, return the point of intersection between the two lines, or return None if there is no intersection.
    '''
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
    D = (x0-x1)*(y2-y3)+(y1-y0)*(x2-x3)
    if D == 0:
        return None
    D = 1./D
    A = (x0*y1-y0*x1)*D
    B = (x2*y3-y2*x3)*D
    return A*(x2-x3)-B*(x0-x1), A*(y2-y3)-B*(y0-y1)


def reflect_vector(vx:float, vy:float, rx:float, ry:float) -> tuple[float, float]:
    '''
    Reflect vector `(vx, vy)` across vector `(rx, ry)`.
    '''
    s = (rx*rx + ry*ry) ** -.5
    nx = ry * s
    ny = -rx * s
    d = 2 * (vx * nx + vy * ny)
    return vx - d * nx, vy - d * ny


def reflect_segment(
    x0:float, y0:float, x1:float, y1:float,
    x2:float, y2:float, x3:float, y3:float,
) -> tuple[float, float, float, float]:
    '''
    The segment defined by `{(x0, y0), (x1, y1)}` is split into two smaller segments at the point of intersection `P` between `{(x0, y0), (x1, y1)}` and `{(x2, y2), (x3, y3)}`.
    Then, reflect the segment `{P, (x1, y1)}` across `{(x2, y2), (x3, y3)}` and denote it `{P, Q}`.
    Return the points `P` and `Q` as a flattened tuple (four floats in a single-dimensional tuple).
    '''
    Px, Py = line_intersection_point(x0, y0, x1, y1, x2, y2, x3, y3)
    rx, ry = reflect_vector(x1 - Px, y1 - Py, x3 - x2, y3 - y2)
    return Px, Py, Px + rx, Py + ry


def rectangle_contains_point(rect_x0:float, rect_y0:float, rect_x1:float, rect_y1:float, point_x:float, point_y:float) -> bool:
    return point_x >= rect_x0 and point_x <= rect_x1 and point_y >= rect_y0 and point_y <= rect_y1


def vector_angle(v0x:float, v0y:float, v1x:float, v1y:float) -> float:
    '''
    Return the angle between the two vectors.  The returned angle is always between 0 (inclusive) and π (exclusive).  Neither of the vectors may be zero.
    '''
    return acos((v0x * v1x + v0y * v1y) / ((v1x * v1x + v1y * v1y) * (v0x * v0x + v0y * v0y)) ** .5)


def lerp_angle(a:float, b:float, t:float, max_delta:float=7) -> float:
    '''
    Return a new angle that is interpolated between angles `a` and `b` by the inerpolant `t` and is within the shorter arc between the points at angles `a` and `b`.

    `max_delta` is the maximum angle by which the returned angle is incremented/decremented from `a` toward `b`.
    '''
    a %= tau
    b %= tau
    delta = (b - a + int(abs(a-b) > pi) * (2 * int(a > b) - 1) * tau) * t
    if abs(delta) > max_delta:
        if delta < 0:
            delta = -max_delta
        else:
            delta = max_delta
    return (a + delta) % tau


def angle_from_to(x0:float, y0:float, x1:float, y1:float) -> float:
    '''
    Return the angle from `(x0, y0)` to `(x1, y1)` in radians.
    '''
    return atan2(y1 - y0, x1 - x0)


def angle_delta(a:float, b:float) -> float:
    '''
    Return the difference between angles `a` and `b`.  The returned value is always less than or equal to π.
    '''
    t = abs(a - b) % tau
    if t >= pi:
        return tau - t
    else:
        return t


def lerp_angle_direction(a:float, b:float) -> int:
    return sign(b - a + int(abs(a-b) > pi) * (2 * int(a > b) - 1) * tau)


def lerp_vector(x0:float, y0:float, x1:float, y1:float, t:float, max_delta: float = None) -> tuple[float, float]:
    '''
    Linearly interpolate from vector `(x0, y0)` to `(x1, y1)` by the interpolant `t`.
    '''
    if max_delta is None:
        return lerp(x0, x1, t), lerp(y0, y1, t)
    lx = lerp(x0, x1, t)
    ly = lerp(y0, y1, t)
    if sq_l2_distance(x0, y0, lx, ly) > max_delta * max_delta:
        dx, dy = normalized_vector(lx - x0, ly - y0, length=max_delta)
        return x0 + dx, y0 + dy
    return lx, ly

def inv_lerp_vector(x0: float, y0: float, x1: float, y1: float, xt: float, yt: float) -> float:
    '''
    Given a point `(xt, yt)` on a line with endpoints `(x0, y0)` and `(x1, y1)`, calculate `t` such that `lerp_vector(x0, y0, x1, y1, t) = (xt, yt)`.
    '''
    if x0 == x1:
        return inv_lerp(yt, y0, y1)
    return inv_lerp(xt, x0, x1)

def u_interp(x: float) -> float:
    '''
    Return a parabola translated and scaled such that a "U" shape connects from (0, 1) to (1, 1), causing this function to output `u_interp(0) = 1`, `u_interp(0.5) = 0`, and `u_interp(1) = 1` with a symmetric weight toward zero.
    '''
    return (x + x - 1) * (x + x - 1)
