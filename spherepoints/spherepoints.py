


from sympy import acos, cos, sin, symbols, diff, init_printing
import numpy as np


init_printing()

a, b, x, y = symbols('a b x y')

greatcircle = acos(cos(a) * cos(b) * cos(x - y) + sin(a) * sin(b))

inv_sqr_gs = greatcircle ** -2

da = diff(inv_sqr_gs, a)
db = diff(inv_sqr_gs, b)
dx = diff(inv_sqr_gs, x)
dy = diff(inv_sqr_gs, y)

gradient = [da, db, dx, dy]

print(inv_sqr_gs)

for g in gradient:
	print(g)

def compute_greatcircle(v: np.array) -> float:
	subs = {a: v[0], b: v[1], x: v[2], y: v[3]}
	return greatcircle.evalf(subs=subs)

def compute_greatcircle_inv_sqr(v: np.array) -> float:
	subs = {a: v[0], b: v[1], x: v[2], y: v[3]}
	return inv_sqr_gs.evalf(subs=subs)

def compute_gradient(v: np.array) -> np.array:
	subs = {a: v[0], b: v[1], x: v[2], y: v[3]}
	return np.array([float(g.evalf(subs=subs)) for g in gradient])


dt = 0.001
N = 10
p0 = np.column_stack((np.random.normal(0, np.pi/4, size=N), np.random.uniform(-np.pi, np.pi, size=N)))

print(p0)
input()

p = p0.copy()

while True:
	increment = np.zeros_like(p)
	total_d, total_ds = 0, 0
	for i in range(N):
		for j in range(N):
			v = np.array([p[i,0], p[j,0], p[i,1], p[j,1]])
			g = compute_gradient(v) * dt
			increment[i] -= g[0], g[2]
			increment[j] -= g[1], g[3]
			d, ds = compute_greatcircle(v), compute_greatcircle_inv_sqr(v)
			total_d += d
			total_ds += ds
	p += increment
	print(total_d, total_ds)


