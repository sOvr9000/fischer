
import numpy as np
import matplotlib.pyplot as plt



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

def compute_greatcircle(point_pairs: np.ndarray) -> np.ndarray:
    '''
    `point_pairs` must have shape `(N, 4)` for some `N`.
    '''
    A, B, X, Y = point_pairs.T
    return np.arccos(np.cos(A) * np.cos(B) * np.cos(X - Y) + np.sin(A) * np.sin(B))

def compute_greatcircle_inv_sqr(point_pairs: np.ndarray) -> np.ndarray:
    d = compute_greatcircle(point_pairs)
    return 1. / (d * d)

def compute_gradient(point_pairs: np.ndarray) -> np.ndarray:
    xy = point_pairs[:, 2] - point_pairs[:, 3]
    cosa, cosb = np.cos(point_pairs[:, :2]).T
    sina, sinb = np.sin(point_pairs[:, :2]).T
    cosxy = np.cos(xy)
    c0 = sina * sinb + cosa * cosb * cosxy
    c1 = np.arccos(c0)
    c0 = 1. / ((1. - c0 * c0) ** .5 * c1 * c1 * c1)
    c2 = np.sin(xy) * cosa * cosb * c0
    return np.where(np.repeat(c0[:, np.newaxis], 4, axis=1) >= 1, 0, np.column_stack((
        c0 * (-sina * cosb * cosxy + sinb * cosa),
        c0 * (sina * cosb - sinb * cosa * cosxy),
        -c2,
        c2
    )))

def generate_point_pairs(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Take an array of shape `(N, 2)` and return an array of shape `(N*(N-1)/2, 4)` along with the indices (of `points`) for the items of each pair.
    '''
    N = points.shape[0]
    S = N * (N - 1) // 2
    I = np.zeros((S, 2), dtype=int)
    point_pairs = np.zeros((S, 4), dtype=points.dtype)
    k = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            I[k] = i, j
            point_pairs[k] = *points[i], *points[j]
            k += 1
    return point_pairs, I

def simulate_step(points: np.ndarray, dt: float) -> None:
    '''
    Adjust the points along the gradient of the inverted squared great circle distance between all pairs of points.
    '''
    point_pairs, I = generate_point_pairs(points)
    g = compute_gradient(point_pairs) * dt
    # points[I[:, 0]] -= g[:, [0, 2]]
    # points[I[:, 1]] -= g[:, [1, 3]]
    for (a, b, x, y), (i, j) in zip(g, I):
        points[i] -= a, x
        points[j] -= b, y

def distance_array(points: np.ndarray) -> np.ndarray:
    N = points.shape[0]
    point_pairs, I = generate_point_pairs(points)
    unsorted_dists = compute_greatcircle(point_pairs)
    dists = np.zeros((N, N - 1), dtype=unsorted_dists.dtype)
    for d, (i, j) in zip(unsorted_dists, I):
        ji = j > i
        dists[i, j - int(ji)] = d
        dists[j - int(ji) + 1, i] = d
    return dists

def minimal_distances(points: np.ndarray) -> np.ndarray:
    '''
    For each point, compute the distance to the closest point.  In a well-packed distribution of points on a sphere, the minimum of the returned array should be at a maximum.
    '''
    return np.min(distance_array(points), axis=1)

def minimal_distance(points: np.ndarray) -> np.ndarray:
    '''
    Find the smallest distance between all pairs of points.
    '''
    return np.min(minimal_distances(points))

def minimal_distances_error(points: np.ndarray) -> np.ndarray:
    '''
    Find the percentage error in the range of the minimal distances to the mean of the minimal distances.  In a perfectly packed distribution of points on a sphere, this is zero, which is only possible with 4, 6, 8, 12, or 20 points.
    '''
    dists = minimal_distances(points)
    M = np.max(dists)
    m = np.min(dists)
    u = np.mean(dists)
    return (M - m) / u



points = fib_points(128)
plot(points)

for _ in range(512):
    simulate_step(points, 0.001)
    # print(minimal_distance(points))
    print(points[1])

plot(points)


