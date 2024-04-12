

import numpy as np
from numba import cuda
from numba import float64
from matplotlib import pyplot as plt
from math import sin, cos, acos


PHI = 2.3999632297286533222315555066336

def fib_points(n: int) -> np.ndarray:
    points = np.zeros((n, 2), dtype=np.float64)
    x = 0
    for i in range(n):
        x = (x + PHI) % 6.283185307179586476925286766559
        points[i] = np.pi * .5 - np.arccos(-1 + i * 2 / (n - 1)), x - np.pi
    return points




def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
    '''
    Assume `radius = 1`.
    '''
    cos = np.cos(spherical[:, 0])
    return np.column_stack((np.cos(spherical[:, 1]) * cos, np.sin(spherical[:, 0]), np.sin(spherical[:, 1]) * cos))

def plot(points: np.array) -> None:
    cart = spherical_to_cartesian(points)
    plt.figure().add_subplot(111, projection='3d').scatter(*cart.T);
    plt.show()

@cuda.jit(device=True)
def compute_greatcircle(a, b, x, y):
    return acos(cos(a) * cos(b) * cos(x - y) + sin(a) * sin(b))

@cuda.jit(device=True)
def compute_greatcircle_inv_sqr(a, b, x, y):
    d = compute_greatcircle(a, b, x, y)
    return 1. / (d * d)

@cuda.jit(device=True)
def compute_gradient(a, b, x, y):
    cosa = cos(a)
    cosb = cos(b)
    sina = sin(a)
    sinb = sin(b)
    cosxy = cos(x - y)
    c0 = sina * sinb + cosa * cosb * cosxy
    if abs(c0) >= 1:
        return 0., 0., 0., 0.
    c1 = acos(c0)
    c0 = 1. / ((1. - c0 * c0) ** .5 * c1 * c1 * c1)
    c2 = sin(x - y) * cosa * cosb * c0
    return (
        c0 * (sinb * cosa - sina * cosb * cosxy),
        c0 * (sina * cosb - sinb * cosa * cosxy),
        -c2,
        c2
    )

@cuda.jit
def _cuda_kernel_simulate(points, dt, iterations):
    deltas = cuda.shared.array((512, 2), float64)
    i = cuda.grid(1)
    if i >= points.shape[0]: return
    for _ in range(iterations):
        deltas[i, 0] = 0
        deltas[i, 1] = 0
        cuda.syncthreads()
        for j in range(points.shape[0]):
            if j == i: continue
            da, db, dx, dy = compute_gradient(points[i, 0], points[j, 0], points[i, 1], points[j, 1])
            cuda.atomic.sub(deltas, (i, 0), da)
            cuda.atomic.sub(deltas, (j, 0), db)
            cuda.atomic.sub(deltas, (i, 1), dx)
            cuda.atomic.sub(deltas, (j, 1), dy)
            # cuda.syncthreads()
        cuda.syncthreads()
        cuda.atomic.add(points, (i, 0), deltas[i, 0] * dt)
        cuda.atomic.add(points, (i, 1), deltas[i, 1] * dt)


def simulate(N: int, dt: float, iterations: int) -> np.ndarray:
    points = fib_points(N)
    rect = spherical_to_cartesian(points)
    print(minimum_distance(rect))
    _points = cuda.to_device(points)
    _cuda_kernel_simulate[N // 16, 16](_points, dt, iterations)
    points[:] = _points.copy_to_host()
    return points

def minimum_distance(rectangular_points: np.ndarray) -> float:
    return min(((rectangular_points[i, 0] - rectangular_points[j, 0]) * (rectangular_points[i, 0] - rectangular_points[j, 0]) + (rectangular_points[i, 1] - rectangular_points[j, 1]) * (rectangular_points[i, 1] - rectangular_points[j, 1])) for i in range(rectangular_points.shape[0] - 1) for j in range(i + 1, rectangular_points.shape[0])) ** .5




def main():
    points = simulate(N=512, dt=0.000001, iterations=1024)
    rect = spherical_to_cartesian(points)
    print(minimum_distance(rect))
    plot(points)

if __name__ == '__main__':
    main()








