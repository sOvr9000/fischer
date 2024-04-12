
from math import atan2, log2, log1p
import numpy as np
from numba import cuda
import cv2
from fischer.stopwatch import Stopwatch
# from colorsys import hsv_to_rgb


@cuda.jit
def _mandelbrot_cuda_kernel(re_min, re_max, im_min, im_max, out_arr):
    i, j = cuda.grid(2)
    i_ = i / out_arr.shape[0]
    j_ = j / out_arr.shape[1]
    re = re_min * (1 - j_) + re_max * j_
    im = im_min * (1 - i_) + im_max * i_
    x = re
    y = im
    for n in range(10000):
        t = x
        x = x * x - y * y + re
        y = 2 * t * y + im
        if x * x + y * y >= 4:
            out_arr[i, j] = n
            return
    out_arr[i, j] = -1


def mandelbrot_gpu(re_min: float, re_max: float, im_min: float, im_max: float, w: int, h: int, threads: int) -> np.ndarray[float]:
    _out_arr = cuda.device_array((h, w))
    _mandelbrot_cuda_kernel[((h//threads) + 1, w), (threads, 1)](re_min, re_max, im_min, im_max, _out_arr)
    return _out_arr.copy_to_host()



@cuda.jit
def _mandelbrot_cuda_kernel_many(regions, out_arr):
    k, i, j = cuda.grid(3)
    i_ = i / out_arr.shape[1]
    j_ = j / out_arr.shape[2]
    re_min, re_max, im_min, im_max = regions[k]
    re = re_min * (1 - j_) + re_max * j_
    im = im_min * (1 - i_) + im_max * i_
    x = re
    y = im
    for n in range(10000):
        t = x
        x = x * x - y * y + re
        y = 2 * t * y + im
        if x * x + y * y >= 4:
            out_arr[k, i, j] = n
            return
    out_arr[k, i, j] = -1


def mandelbrot_gpu_many(regions: np.ndarray[float], size: tuple[int, int], threads: int) -> np.ndarray[float]:
    w, h = size
    _regions = cuda.to_device(regions)
    _out_arr = cuda.device_array((regions.shape[0], h, w))
    _mandelbrot_cuda_kernel_many[(regions.shape[0], (h//threads), w), (1, threads, 1)](_regions, _out_arr)
    return _out_arr.copy_to_host()



@cuda.jit
def _mandelbrot_cuda_kernel_escape_many(regions, out_arr):
    # Also computes escape angles
    k, i, j = cuda.grid(3)
    i_ = i / out_arr.shape[1]
    j_ = j / out_arr.shape[2]
    re_min, re_max, im_min, im_max = regions[k]
    re = re_min * (1 - j_) + re_max * j_
    im = im_min * (1 - i_) + im_max * i_
    x = re
    y = im
    for n in range(10000):
        j_ = x * x - y * y + re # XXX Reusing i_ and j_ as a horrible attempt to save memory
        i_ = 2 * x * y + im
        if j_ * j_ + i_ * i_ >= 4:
            out_arr[k, i, j, 0] = n
            out_arr[k, i, j, 1] = atan2(i_ - y, j_ - x)
            return
        x = j_
        y = i_
    out_arr[k, i, j, 0] = -1


def mandelbrot_gpu_escape_many(regions: np.ndarray[float], size: tuple[int, int], threads: int) -> np.ndarray[float]:
    '''
    Also return escape angles, along with the iterations to divergence.
    '''
    w, h = size
    _regions = cuda.to_device(regions)
    _out_arr = cuda.device_array((regions.shape[0], h, w, 2))
    _mandelbrot_cuda_kernel_escape_many[(regions.shape[0], (h//threads) + 1, w), (1, threads, 1)](_regions, _out_arr)
    return _out_arr.copy_to_host()


@cuda.jit(device=True)
def hsv_to_rgb(h, s, v):
    h %= 1
    h *= 6
    if s == 0.0:
        return v, v, v
    i = int(h) # XXX assume int() truncates!
    h -= i
    i %= 6
    if i == 0:
        return v, v*(1.0 - s*(1.0-h)), v*(1.0 - s)
    if i == 1:
        return v*(1.0 - s*h), v, v*(1.0 - s)
    if i == 2:
        return v*(1.0 - s), v, v*(1.0 - s*(1.0-h))
    if i == 3:
        return v*(1.0 - s), v*(1.0 - s*h), v
    if i == 4:
        return v*(1.0 - s*(1.0-h)), v*(1.0 - s), v
    if i == 5:
        return v, v*(1.0 - s), v*(1.0 - s*h)

@cuda.jit(device=True)
def lerp(a, b, t):
    return a*(1-t) + b*t

@cuda.jit
def _mandelbrot_cuda_kernel_render_many(regions, out_arr, iterations, color_palette, color_palette_rate, log_mapped):
    # Directly computes pixel colors from iterations and escape angles.
    k, i, j = cuda.grid(3)
    if k >= out_arr.shape[0] or i >= out_arr.shape[1] or j >= out_arr.shape[2] or k < 0 or i < 0 or j < 0:
        return
    i_ = i / out_arr.shape[1]
    j_ = j / out_arr.shape[2]
    re = regions[k, 0] * (1 - j_) + regions[k, 1] * j_
    im = regions[k, 2] * (1 - i_) + regions[k, 3] * i_
    x = re
    y = im
    x2 = re * re
    y2 = im * im
    for n in range(iterations):
        px = x
        py = y
        y = (x + x) * y + im
        x = x2 - y2 + re
        x2 = x * x
        y2 = y * y
        if x2 + y2 >= 256:
            i_ = atan2(y - py, x - px)
            # At this point, i_ represents escape angle, and n represents number of iterations to divergence.

            # https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set#Continuous_(smooth)_coloring
            # n += 1 - log2(log2(x2 + y2) * 0.5)
            n += 1 - log2(log2(x2 + y2) * 0.5)

            # https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set#Exponentially_mapped_and_Cyclic_Iterations
            # n = ((n * color_palette_rate / 10) ** 1.2 * color_palette.shape[0]) ** 1.5

            # r, g, b = hsv_to_rgb(n * color_palette_rate, 1, 1)
            # r, g, b = color_palette[int(n * color_palette_rate) % color_palette.shape[0]]

            if log_mapped:
                n = log1p(n)

            # NOTE: UNCOMMENT THIS LINE IF NOT USING EXPONENTIALLY MAPPED ITERATIONS
            n *= color_palette_rate

            n = max(0, n)
            nm = n % 1
            index = int(n)
            r1, g1, b1 = color_palette[index % color_palette.shape[0]]
            r2, g2, b2 = color_palette[(index + 1) % color_palette.shape[0]]
            # v = (i_ / np.pi + 1) / 2
            v = 1
            r = lerp(r1, r2, nm) * v
            g = lerp(g1, g2, nm) * v
            b = lerp(b1, b2, nm) * v

            out_arr[k, i, j, 0] = r
            out_arr[k, i, j, 1] = g
            out_arr[k, i, j, 2] = b
            return


@cuda.jit
def _cuda_kernel_generate_regions_by_path(max_path, start_path, generator_path, out_regions, power):
    k = cuda.grid(1)
    cur_path = (start_path + generator_path * k) % max_path
    w = 3.5
    h = 2
    re_min = -2.5
    im_min = -1
    K = 3
    for t in range(0, 2*power, 2):
        w *= .5
        h *= .5
        quad = (cur_path & K) >> t
        if quad == 0:
            re_min += w
            im_min += h
        elif quad == 1:
            im_min += h
        elif quad == 3:
            re_min += w
        K <<= 2
    out_regions[k, 0] = re_min
    out_regions[k, 1] = re_min + w
    out_regions[k, 2] = im_min
    out_regions[k, 3] = im_min + h

# @cuda.jit
# def _mandelbrot_cuda_kernel_render_many_by_paths(max_path, start_path, generator_path, out_arr, color_palette, color_palette_rate):
# 	# Directly computes pixel colors from iterations and escape angles.
# 	# Instead of directly providing regions, integers (a single int for each path) can be generated on the fly to come up with previously unseen regions.
# 	# This takes less memory overall at the cost of slightly more computation while also having the property of never revisiting regions until all have been visited once.
# 	# However, each region is not likely to be an "interesting" part of the complex plane, where the Mandelbrot set intersects the region.
# 	# Thus, GPU acceleration is preferred and worth giving a shot, and a good generator_path should be used.
# 	k, i, j = cuda.grid(3)
# 	i_ = i / out_arr.shape[1]
# 	j_ = j / out_arr.shape[2]

# 	cur_path = (start_path + generator_path * k) % max_path
# 	w = 3.5
# 	h = 2
# 	re_min = -2.5
# 	im_min = -1
# 	K = 3
# 	for t in range(0, 30, 2):
# 		w *= .5
# 		h *= .5
# 		quad = (cur_path & K) >> t
# 		if quad == 0:
# 			re_min += w
# 			im_min += h
# 		elif quad == 1:
# 			im_min += h
# 		elif quad == 3:
# 			re_min += w
# 		K <<= 2
# 	re = re_min + w * j_
# 	im = im_min + h * i_

# 	x = re
# 	y = im
# 	x2 = re * re
# 	y2 = im * im
# 	for n in range(10000):
# 		px = x
# 		py = y
# 		y = (x + x) * y + im
# 		x = x2 - y2 + re
# 		x2 = x * x
# 		y2 = y * y
# 		# j_ = x * x - y * y + re # XXX Reusing i_ and j_ as a horrible attempt to save memory
# 		# i_ = 2 * x * y + im
# 		if x2 + y2 >= 256:
# 			i_ = atan2(y - py, x - px)
# 			# At this point, i_ represents escape angle, and n represents number of iterations to divergence.

# 			# https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set#Continuous_(smooth)_coloring
# 			# n += 1 - log2(log2(x2 + y2) * 0.5)
# 			n += 1 - log2(log2(x2 + y2) * 0.5)

# 			# https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set#Exponentially_mapped_and_Cyclic_Iterations
# 			# n = ((n * color_palette_rate / 10) ** 1.2 * color_palette.shape[0]) ** 1.5

# 			# r, g, b = hsv_to_rgb(n * color_palette_rate, 1, 1)
# 			# r, g, b = color_palette[int(n * color_palette_rate) % color_palette.shape[0]]

# 			# NOTE: UNCOMMENT THIS LINE IF NOT USING EXPONENTIALLY MAPPED ITERATIONS
# 			# n *= color_palette_rate

# 			n = max(0, n)
# 			nm = n % 1
# 			r1, g1, b1 = color_palette[int(n) % color_palette.shape[0]]
# 			r2, g2, b2 = color_palette[(int(n) + 1) % color_palette.shape[0]]
# 			# v = (i_ / np.pi + 1) / 2
# 			v = 1
# 			r = lerp(r1, r2, nm) * v
# 			g = lerp(g1, g2, nm) * v
# 			b = lerp(b1, b2, nm) * v

# 			out_arr[k, i, j, 0] = r
# 			out_arr[k, i, j, 1] = g
# 			out_arr[k, i, j, 2] = b
# 			return
# 		# x = j_
# 		# y = i_

def get_regions_by_paths(max_path: int, start_path: int, generator_path: int, num_paths: int, power: int):
    if num_paths == 1:
        th = 1
    else:
        th = 4
    _regions = cuda.device_array((num_paths, 4))
    _cuda_kernel_generate_regions_by_path[num_paths // th, th](max_path, start_path, generator_path, _regions, power)
    return _regions.copy_to_host()

def render_many(regions: np.ndarray[float], size: tuple[int, int], threads: int = 32, iterations: int = 1000, color_palette: np.ndarray[float] = None, color_palette_rate: float = 0.02, log_mapped: bool = False) -> np.ndarray[float]:
    '''
    Return pixel values in RGB format, each float ranging 0-1.
    '''
    w, h = size
    _regions = cuda.to_device(regions)
    _out_arr = cuda.device_array((regions.shape[0], h, w, 3))
    _palette = cuda.to_device(color_palette)
    _mandelbrot_cuda_kernel_render_many[(regions.shape[0], h // threads, w // threads), (1, threads, threads)](_regions, _out_arr, iterations, _palette, color_palette_rate, log_mapped)
    return _out_arr.copy_to_host()

def render_many_by_path(max_path: int, start_path: int, generator_path: int, num_paths: int, power: int, size: tuple[int, int], threads: int = 32, iterations: int = 1000, color_palette: np.ndarray[float] = None, color_palette_rate: float = 0.02, log_mapped: bool = False) -> np.ndarray[float]:
    '''
    Return pixel values in RGB format, each float ranging 0-1.

    If `num_paths > max_path`, then some images will be of the exact same regions of the complex plane.
    If `num_paths <= max_path`, then all images will be of unique regions of the complex plane, although most will be uninteresting like a solid color.
    '''
    w, h = size
    if num_paths == 1:
        th = 1
    else:
        th = 4
    _regions = cuda.device_array((num_paths, 4))
    _cuda_kernel_generate_regions_by_path[num_paths // th, th](max_path, start_path, generator_path, _regions, power)
    _out_arr = cuda.device_array((_regions.shape[0], h, w, 3))
    _palette = cuda.to_device(color_palette)
    _mandelbrot_cuda_kernel_render_many[(_regions.shape[0], h // threads, w // threads), (1, threads, threads)](_regions, _out_arr, iterations, _palette, color_palette_rate, log_mapped)
    return _out_arr.copy_to_host()



def is_roi_interesting(roi: np.ndarray) -> bool:
    mean = roi.mean()
    std = roi.flatten().std()
    # print(mean, std)
    if mean < 0.05 or std < 0.05:
        return False
    if mean > 0.07 and mean < 0.27:
        return 1.48717 / (1 + np.exp(-2.51624 * mean)) - .746756 < std
    else:
        return abs(0.895484 * mean - 0.000661805 - std) > 0.001


def main():
    W, H = 101*1, 101*1
    print(W, H)
    re_min = -2.5
    re_max = 1
    im_min = -1
    im_max = 1
    threads = 8

    from time import sleep
    from fischer.colorpalettes import convert_palette

    import colorcet as cc
    cc.koyw = ['#000000', '#050400', '#090700', '#0f0b00', '#130e00', '#181200', '#1c1500', '#211900', '#251c00', '#2b2000', '#2f2300', '#332600', '#362800', '#3a2b01', '#3c2c00', '#402e00', '#443101', '#473200', '#4b3501', '#4e3700', '#523a01', '#543b00', '#583e01', '#5b3f00', '#5f4201', '#624400', '#664701', '#694800', '#6d4b01', '#704d01', '#734f01', '#775201', '#7a5301', '#7e5601', '#805701', '#845a01', '#875c01', '#8a5f02', '#8c6001', '#8f6402', '#916502', '#946803', '#966a03', '#996d03', '#9d7004', '#9e7204', '#a27505', '#a47604', '#a77a05', '#a97b05', '#ac7f06', '#ae8005', '#b18306', '#b38506', '#b78807', '#b98a06', '#bc8d08', '#bf9008', '#c19107', '#c49508', '#c69608', '#c99909', '#cb9b09', '#ce9e0a', '#d0a009', '#d4a30a', '#d6a50a', '#d9a80b', '#dbaa0a', '#dead0b', '#e0ae0b', '#e3b10b', '#e6b40c', '#e7b50c', '#e9b80e', '#eab80e', '#ebba10', '#ecbb10', '#eebd12', '#eebe12', '#f0c014', '#f0c014', '#f2c316', '#f2c316', '#f4c518', '#f5c719', '#f6c719', '#f8ca1b', '#f8ca1b', '#facc1c', '#facd1d', '#fccf1e', '#fcd01f', '#fed220', '#ffd321', '#ffd422', '#ffd522', '#ffd724', '#ffd724', '#ffd825', '#ffda27', '#ffda27', '#ffdc29', '#ffdc29', '#ffde2b', '#ffdf2b', '#ffe12d', '#ffe12d', '#ffe32f', '#ffe32f', '#ffe530', '#ffe531', '#ffe732', '#ffe833', '#ffe93e', '#ffeb4e', '#ffec5b', '#ffef6b', '#fff078', '#fff288', '#fff395', '#fff5a5', '#fff6b2', '#fff8c2', '#fff9cf', '#fffcdf', '#fffdec', '#fffefc']
    cc.kmrpw = ['#010000', '#090201', '#100201', '#180402', '#1f0402', '#280704', '#2f0703', '#370905', '#3e0904', '#470b06', '#4e0c06', '#530d07', '#550c07', '#5a0e08', '#5d0d08', '#610e0a', '#640e0a', '#690f0b', '#6c0f0b', '#70100c', '#73100c', '#77110e', '#7b110e', '#7d110e', '#821210', '#851210', '#891311', '#8c1311', '#911412', '#931412', '#981514', '#9b1413', '#9f1615', '#a21515', '#a61616', '#a81616', '#ab1616', '#ad1515', '#af1616', '#b11515', '#b41616', '#b51515', '#b81515', '#ba1515', '#bc1414', '#be1515', '#c01414', '#c31515', '#c41414', '#c71414', '#c91313', '#cc1414', '#cd1313', '#d01414', '#d11313', '#d41313', '#d61313', '#d81313', '#da1212', '#dd1313', '#de1212', '#e11212', '#e31212', '#e61212', '#e71212', '#e91111', '#ec1212', '#ed1111', '#f01111', '#f21010', '#f51111', '#f61010', '#f91111', '#fa1010', '#fd1111', '#ff1010', '#ff1010', '#ff0e0e', '#ff0e0e', '#ff0c0c', '#ff0c0c', '#ff0b0b', '#ff0b0b', '#ff0909', '#ff0909', '#ff0808', '#ff0707', '#ff0606', '#ff0505', '#ff0505', '#ff0303', '#ff0303', '#ff0101', '#ff0101', '#ff0000', '#ff0007', '#ff000d', '#ff0115', '#ff001c', '#ff0024', '#ff002a', '#ff0132', '#ff0039', '#ff0041', '#ff0047', '#ff0150', '#ff0056', '#ff005d', '#ff0165', '#ff006b', '#ff0073', '#ff007a', '#ff0182', '#ff0088', '#ff0f91', '#ff2099', '#ff34a2', '#ff45aa', '#ff58b3', '#ff69ba', '#ff7dc3', '#ff8ecb', '#ffa1d4', '#ffb2dc', '#ffc6e5', '#ffd7ed', '#ffeaf6', '#fffcfd']
    cc.kmrypw = ['#000000', '#0a0201', '#140301', '#1f0502', '#290603', '#340804', '#3e0905', '#480b06', '#500c06', '#520c06', '#530c06', '#560c06', '#570c07', '#5a0c07', '#5b0c07', '#5d0c07', '#5f0c07', '#610c07', '#630c07', '#650c07', '#670c07', '#690d08', '#6a0c07', '#6c0c08', '#6e0c08', '#700d08', '#720c08', '#740c08', '#760c08', '#780d09', '#7a0c08', '#7c0c09', '#7e0c09', '#800d09', '#810c09', '#830d09', '#850d09', '#870d09', '#890c09', '#8b0d09', '#8d0d0a', '#8f0d0a', '#900c0a', '#920d0a', '#950d0a', '#960c0a', '#980d0a', '#9a0d0a', '#9c0d0b', '#9e0d0a', '#a00d0b', '#a20d0b', '#a40d0b', '#a50d0b', '#a80d0b', '#a90d0b', '#ac0d0b', '#ad0d0b', '#af0d0b', '#b10d0c', '#b30d0c', '#b50d0c', '#b70d0c', '#b90d0c', '#bb0d0c', '#bc0d0c', '#be0d0c', '#c00d0c', '#c20d0d', '#c40d0c', '#c60d0d', '#c80d0d', '#ca0d0d', '#cc0d0d', '#ce0d0d', '#d00c0c', '#d20c0c', '#d40b0b', '#d70b0b', '#d90a0a', '#db0a0a', '#dd0909', '#df0808', '#e10808', '#e40707', '#e60707', '#e80606', '#ea0606', '#ec0505', '#ef0404', '#f10404', '#f30303', '#f50202', '#f70202', '#f90202', '#fc0101', '#fe0000', '#ff0603', '#ff1508', '#ff240e', '#ff3314', '#ff421a', '#ff5120', '#ff6026', '#ff6f2b', '#ff7e31', '#ff7d3b', '#ff6e49', '#ff5f56', '#ff5064', '#ff4171', '#ff327f', '#ff238c', '#ff149a', '#ff05a7', '#ff0db1', '#ff21b7', '#ff35be', '#ff4ac4', '#ff5ecb', '#ff72d1', '#ff86d8', '#ff9ade', '#ffaee5', '#ffc2eb', '#ffd6f2', '#ffeaf8', '#ffffff']
    cc.krpw = ['#010000', '#0a0201', '#140302', '#1f0503', '#290603', '#340804', '#3e0905', '#490b06', '#500c06', '#530c06', '#560c06', '#5a0c06', '#5c0b05', '#600b06', '#630b05', '#660b05', '#690a05', '#6c0a05', '#6f0a05', '#720a05', '#750905', '#780905', '#7b0904', '#7f0905', '#810904', '#850904', '#880804', '#8b0804', '#8e0804', '#910804', '#940704', '#970704', '#9a0703', '#9d0704', '#a00603', '#a40603', '#a60603', '#aa0603', '#ad0603', '#b00603', '#b30503', '#b60503', '#b90502', '#bc0502', '#bf0502', '#c20402', '#c50402', '#c80402', '#cb0402', '#ce0302', '#d20302', '#d40301', '#d80302', '#db0201', '#de0201', '#e10201', '#e40201', '#e70201', '#ea0201', '#ed0101', '#f00101', '#f30100', '#f70101', '#f90000', '#fd0000', '#ff0001', '#ff0005', '#ff0009', '#ff000d', '#ff0011', '#ff0016', '#ff0019', '#ff001e', '#ff0022', '#ff0026', '#ff002a', '#ff002e', '#ff0032', '#ff0036', '#ff003a', '#ff003f', '#ff0043', '#ff0047', '#ff004b', '#ff004f', '#ff0053', '#ff0057', '#ff005b', '#ff005f', '#ff0063', '#ff0067', '#ff006c', '#ff0070', '#ff0074', '#ff0078', '#ff007c', '#ff0080', '#ff0084', '#ff0088', '#ff008c', '#ff0090', '#ff0095', '#ff0098', '#ff009d', '#ff00a1', '#ff00a5', '#ff00a9', '#ff04ad', '#ff10b1', '#ff1db6', '#ff29b9', '#ff36be', '#ff42c2', '#ff4fc6', '#ff5bca', '#ff68ce', '#ff74d2', '#ff81d6', '#ff8dda', '#ff9ade', '#ffa7e2', '#ffb3e7', '#ffc0ea', '#ffcdef', '#ffd9f3', '#ffe6f7', '#fff2fb', '#ffffff']
    cc.cyclic_koyw = cc.koyw + cc.koyw[::-1]

    palette = [
        (r / 255., g / 255., b / 255.)
        if i >= len(cc.koyw) else
        (r / 255., b / 255., g / 255.)
        for i, (r, g, b) in enumerate(convert_palette(cc.cyclic_koyw))
    ]

    sw = Stopwatch()
    sw.lap()
    img = render_many_by_path(67108864, 0, 41475557, 4096, (W, H), threads, palette, 1)
    # img = render_many(np.array([
    # 	[-2.5, 1, -1, 1],
    # ]), (W, H), threads, palette, 1)
    print(f'{sw.lap()} ms')

    fname = 'goodbad.csv'
    goodbad = np.zeros((img.shape[0], 3))
    K = 0

    for im in img:
        cv2.imshow('mandelbrot', np.repeat(np.repeat(im[:, :, ::-1], 4, axis=0), 4, axis=1))
        # cv2.imshow('mandelbrot', im[:, :, ::-1])
        cv2.waitKey(1)
        im = cv2.resize(im, (11, 11))
        if is_roi_interesting(im):
            mean = im.mean()
            std = im.flatten().std()
            k = cv2.waitKey() & 0xFF
            if k == 27:
                return
            if k == ord('v'): # bad detection
                goodbad[K] = mean, std, 0
            else: # correct detection
                goodbad[K] = mean, std, 1
            K += 1
            with open(fname, 'w') as f:
                f.write('\n'.join(', '.join(map(str, r)) for r in goodbad[:K]))
        # sleep(0.01)

    print('Finished')


if __name__ == '__main__':
    main()


