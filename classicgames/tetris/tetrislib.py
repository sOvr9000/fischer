
import numpy as np
from math import floor


def vec(x,y):
	return np.array([x,y])

def vec_center(vecs):
	return sum(vecs)/len(vecs)

def vec_rounded(vector):
	return vec(floor(vector[0]+0.5), floor(vector[1]+0.5))

def vec_rotated90(vector):
	'''Return a new vector which is the given vector rotated 90 degrees counter-clockwise.'''
	return vec(-vector[1], vector[0])

def vec_rotated270(vector):
	'''Return a new vector which is the given vector rotated 90 degrees clockwise.'''
	return vec(vector[1], -vector[0])

# 180 degree rotation is negation, so do -vector instead of vec_rotated180(vector)

def contains_vec(lst, vec):
	for v in lst:
		if np.array_equal(v,vec):
			return True
	return False

VEC_RIGHT = vec(1,0)
VEC_UP = vec(0,1)
VEC_LEFT = vec(-1,0)
VEC_DOWN = vec(0,-1)
