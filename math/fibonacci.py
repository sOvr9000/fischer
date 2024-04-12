
from typing import Union
from math import atan2 as _atan2, pi as _pi
from mpmath import *
from fischer.math.constants import *
from fischer.colorpalettes import get_palette, get_color_from_palette, convert_palette, set_saturation, scale_value
import colorcet as cc



def fib(n:Union[int, float, complex]) -> Union[float, complex]:
	return PHI_FIB1 * PHI ** n + PHI_FIB2 * (-PHI_INV) ** n

def fib_ap(n:Union[mpf, mpc]) -> Union[mpf, mpc]:
	return

fib_rec_memo = []

def fib_rec(n:int):
	global fib_rec_memo
	if n < 2:
		return 1
	if n < len(fib_rec_memo):
		return fib_rec_memo[n]
	if n >= len(fib_rec_memo):
		fib_rec_memo += [-1] * (n - len(fib_rec_memo) + 1)
	f = fib_rec(n-1) + fib_rec(n-2)
	fib_rec_memo[n] = f
	return f



if __name__ == '__main__':
	import cv2
	import numpy as np
	WIDTH = 512
	HEIGHT = 512
	im = np.zeros((HEIGHT, WIDTH, 3))
	palette = convert_palette(cc.CET_C6)
	print(len(palette))
	for i in range(HEIGHT):
		for j in range(WIDTH):
			a = _atan2(HEIGHT//2-i, WIDTH//2-j) / (2 * _pi)
			c = get_color_from_palette(palette, a * len(palette))
			d = ((i-HEIGHT//2)**2 + (j-WIDTH//2)**2) ** .5
			im[i, j] = set_saturation(c, d / 100)
	cv2.imshow('test', im[:, :, ::-1] / 255)
	cv2.waitKey(0)


