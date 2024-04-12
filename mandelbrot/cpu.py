


from math import atan2
from typing import List, Tuple, Union
import numpy as np
from fischer.stopwatch import Stopwatch
from numba import jit
from multiprocessing import Pool
import cv2
from fischer.colorpalettes import get_palette, get_color_from_palette, convert_palette, scale_value, shift_palette_hue, cycle_palette, shift_hue
import colorcet
import mpmath as mpm



@jit
def test_point(re:float, im:float, max_iterations:int=1000) -> Tuple[int, float]:
	x:float = re
	y:float = im
	for iteration in range(max_iterations):
		if x*x+y*y > 4: return iteration, atan2(y,x)
		x, y = x*x-y*y+re, 2*x*y+im
	return -1, atan2(y,x)

def test_point_ap(re:mpm.mpf, im:mpm.mpf, max_iterations:int=1000, prec:int=16) -> Tuple[int, float]:
	# TODO Use arbitrary precision
	# with mpm.workdps(prec):
	x:mpm.mpf = re
	y:mpm.mpf = im
	x2:mpm.mpf = x*x
	y2:mpm.mpf = y*y
	for iteration in range(max_iterations):
		if x2+y2 > 4: return iteration, atan2(float(y),float(x))
		x2, y2 = x*x, y*y
		x, y = x2-y2+re, 2*x*y+im
	return -1, atan2(float(y),float(x))

# def test_region(re_min:float, im_min:float, re_max:float, im_max:float, width:int, height:int, max_iterations:int=1000) -> np.ndarray:
# 	result:np.ndarray = np.zeros((height, width), dtype=int)
# 	range_re:float = re_max - re_min
# 	range_im:float = im_max - im_min
# 	inv_width:float = 1. / (width - 1)
# 	inv_height:float = 1. / (height - 1)
# 	for i in range(height):
# 		im = im_min + range_im * i * inv_height
# 		for j in range(width):
# 			re = re_min + range_re * j * inv_width
# 			result[i, j] = test_point(re, im, max_iterations=max_iterations)
# 	return result

@jit
def _test_region(re_min:float, im_min:float, re_max:float, im_max:float, width:int, height:int, max_iterations:int=1000, endpoints=True) -> List[List[Tuple[int, float]]]:
	result:List[List[Tuple[int, float]]] = [[(0, 0.) for _ in range(width)] for _ in range(height)]
	range_re:float = re_max - re_min
	range_im:float = im_max - im_min
	if endpoints:
		inv_width:float = 1. / (width - 1)
		inv_height:float = 1. / (height - 1)
	else:
		inv_width:float = 1. / width
		inv_height:float = 1. / height
	for i in range(height):
		im:float = im_min + range_im * i * inv_height
		for j in range(width):
			re:float = re_min + range_re * j * inv_width
			result[i][j] = test_point(re, im, max_iterations=max_iterations)
	return result

def _test_region_ap(re_min:mpm.mpf, im_min:mpm.mpf, re_max:mpm.mpf, im_max:mpm.mpf, width:int, height:int, max_iterations:int=1000, endpoints=True, prec:int=16) -> List[List[Tuple[int, float]]]:
	result:List[List[Tuple[int, float]]] = [[(0, 0.) for _ in range(width)] for _ in range(height)]
	# with mpm.workdps(prec):
	range_re:mpm.mpf = re_max - re_min
	range_im:mpm.mpf = im_max - im_min
	if endpoints:
		inv_width:mpm.mpf = 1. / (width - 1)
		inv_height:mpm.mpf = 1. / (height - 1)
	else:
		inv_width:mpm.mpf = 1. / width
		inv_height:mpm.mpf = 1. / height
	for i in range(height):
		im:mpm.mpf = im_min + range_im * i * inv_height
		for j in range(width):
			re:mpm.mpf = re_min + range_re * j * inv_width
			result[i][j] = test_point_ap(re, im, max_iterations=max_iterations, prec=prec)
	return result

def test_region(re_min:float, im_min:float, re_max:float, im_max:float, width:int, height:int, max_iterations:int=1000, endpoints:bool=True, prec:int=16) -> np.ndarray:
	if prec == 16:
		return np.array(_test_region(re_min, im_min, re_max, im_max, width, height, max_iterations=max_iterations, endpoints=endpoints), dtype=float)
	else:
		return np.array(_test_region_ap(re_min, im_min, re_max, im_max, width, height, max_iterations=max_iterations, endpoints=endpoints, prec=prec), dtype=float)

def test_region_indexed(i:int, j:int, re_min:float, im_min:float, re_max:float, im_max:float, width:int, height:int, max_iterations:int=1000, endpoints:bool=True, prec:int=16):
	return i, j, test_region(re_min, im_min, re_max, im_max, width, height, max_iterations=max_iterations, endpoints=endpoints, prec=prec)

def lerp(a:float, b:float, t:float) -> float:
	return a*(1-t)+b*t

def inv_lerp(a:float, b:float, x:float) -> float:
	return (x-a)/(b-a)

def worker_init(dps):
	# print(f'Current worker dps: {mpm.mp.dps}')
	mpm.mp.dps = dps
	# print(f'New worker dps: {mpm.mp.dps}')

def test_region_mp(re_min:Union[float, mpm.mpf], im_min:Union[float, mpm.mpf], re_max:Union[float, mpm.mpf], im_max:Union[float, mpm.mpf], width:int, height:int, max_iterations:int=1000, workers:int=8, prec:int=16) -> np.ndarray:
	'''
	Points at `im_max` and `re_max` are included.
	'''
	CHUNK_SIZE = 24
	if prec != 16:
		# with mpm.workdps(prec):
		res = [lerp(re_min, re_max, k / CHUNK_SIZE) for k in range(CHUNK_SIZE+1)]
		ims = [lerp(im_min, im_max, k / CHUNK_SIZE) for k in range(CHUNK_SIZE+1)]
	else:
		res = np.linspace(re_min, re_max, CHUNK_SIZE+1, endpoint=True)
		ims = np.linspace(im_min, im_max, CHUNK_SIZE+1, endpoint=True)
	ws = [width//CHUNK_SIZE] * CHUNK_SIZE
	hs = [height//CHUNK_SIZE] * CHUNK_SIZE
	s = sum(ws)
	if s < width:
		s = width - s
		for t in range(s):
			ws[t] += 1
	s = sum(hs)
	if s < height:
		s = height - s
		for t in range(s):
			hs[t] += 1
	regions = [
		(
			i, j,
			res[j], ims[i],
			res[j+1], ims[i+1],
			ws[j], hs[i],
			max_iterations,
			False, # endpoints
			prec
		)
		for i in range(CHUNK_SIZE)
		for j in range(CHUNK_SIZE)
	]
	cz = int(16 - mpm.log10(re_max - re_min))
	with Pool(processes=workers, initializer=worker_init, initargs=(cz,)) as pool:
		data = pool.starmap(test_region_indexed, regions)
	data.sort()
	results = np.zeros((height, width, 2), dtype=float)
	i_ = 0
	for i in range(CHUNK_SIZE):
		h = data[i*CHUNK_SIZE][2].shape[0]
		results[i_:i_+h] = np.concatenate(tuple(a for _, _, a in data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]), axis=1)
		i_ += h
	return results
