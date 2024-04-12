
from .common import greater_quadratic_root
from typing import Tuple



def ulam_coords_inv(x: int, y: int) -> int:
	'''
	Return `n` where `ulam_coords(n) = (x, y)`.
	'''
	if -x <= y >= x:
		return 2 * y * (2 * y - 1) + y - x
	if -x > y > x:
		return 2 * x * (2 * x - 1) - y + x
	if -x >= y <= x:
		return 2 * y * (2 * y - 1) - y + x
	return 4 * x * x - 3 * x + y

def ulam_coords(n: int) -> Tuple[int, int]:
	'''
	Return the integer coordinates on the Cartesian plane that follow the Ulam spiral ("square spiral").
	'''
	k = int(n ** .5) // 2 * 2
	k2 = k * k
	nk2 = n - k2
	if 0 <= nk2 <= k:
		return (
			int(k * -.5),
			int(k * .5 - nk2)
		)
	elif k < nk2 <= k * 2 + 1:
		return (
			int(k * -.5 + nk2 - k),
			int(k * -.5)
		)
	elif k * 2 + 1 < nk2 < k * 3 + 2:
		return (
			int(k * .5 + 1),
			int(k * -2.5 + nk2 - 1)
		)
	else:
		return (
			int(k * 3.5 - nk2 + 3),
			int(k * .5 + 1)
		)



if __name__ == '__main__':
	print(ulam_coords_inv(1, 1))
	# print([ulam_coords(n) for n in range(10)])



