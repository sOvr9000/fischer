

import numpy as np
from cuda import mandelbrot_gpu_many
from cpu import test_region_mp
from fischer.stopwatch import Stopwatch
import cv2



def main():
	cx, cy = -0.77568377, 0.13646737
	zoom = 1

	SCALE = 2
	sw = Stopwatch()
	# A = test_region_mp(cx - 1.75 * zoom, cy - zoom, cx + 1.75 * zoom, cy + zoom, 1750*SCALE, 1000*SCALE, 1000, 15)
	# print(f'{sw.lap()} ms')
	# cv2.imwrite('cpu_mandel.png', A[:, :, 0])

	THREADS = 64
	PARALLEL_SCALE = 4
	B = mandelbrot_gpu_many(np.array([[cx - 1.75 * v * zoom, cx + 1.75 * v * zoom, cy - v * zoom, cy + v * zoom] for v in np.logspace(0, 1, PARALLEL_SCALE, base=0.1)]), (1792*SCALE, 1024*SCALE), THREADS)
	print(f'{sw.lap()} ms')
	for i, A in enumerate(B):
		cv2.imwrite(f'gpu_mandel_{i}.png', A)

	# THREADS = 32
	# SCALE = 1
	# mandelbrot_gpu(-2.5, 1, -1, 1, 1750*SCALE, 1000*SCALE, THREADS)
	# print(f'{sw.lap()} ms')

if __name__ == '__main__':
	main()






