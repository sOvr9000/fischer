

import numpy as np
import cv2
from numba.cuda import jit, grid
from fischer.mathexpressions import *
from math import sin
from fischer.cudafuncs import lerp



@jit
def _cuda_kernel_eval_func(inputs_min, inputs_max, out_arr, num_inputs: int):
    t = grid(num_inputs)
    i, = t
    x = lerp(inputs_min[0], inputs_max[0], i / out_arr.shape[0])
    out_arr[i] = x*x-1+sin(x)


def eval_func(root_operator: Operator, inputs_min: list[float], inputs_max: list[float], step_size: list[float]) -> list[float]:
    '''
    Evaluate `root_operator` with certain inputs.
    '''




