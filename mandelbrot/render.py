
import numpy as np
from fischer.stopwatch import Stopwatch
import cv2
from fischer.colorpalettes import get_color_from_palette, convert_palette, shift_palette_hue, cycle_palette
import colorcet
import mpmath as mpm



colorcet.koyw = ['#000000', '#050400', '#090700', '#0f0b00', '#130e00', '#181200', '#1c1500', '#211900', '#251c00', '#2b2000', '#2f2300', '#332600', '#362800', '#3a2b01', '#3c2c00', '#402e00', '#443101', '#473200', '#4b3501', '#4e3700', '#523a01', '#543b00', '#583e01', '#5b3f00', '#5f4201', '#624400', '#664701', '#694800', '#6d4b01', '#704d01', '#734f01', '#775201', '#7a5301', '#7e5601', '#805701', '#845a01', '#875c01', '#8a5f02', '#8c6001', '#8f6402', '#916502', '#946803', '#966a03', '#996d03', '#9d7004', '#9e7204', '#a27505', '#a47604', '#a77a05', '#a97b05', '#ac7f06', '#ae8005', '#b18306', '#b38506', '#b78807', '#b98a06', '#bc8d08', '#bf9008', '#c19107', '#c49508', '#c69608', '#c99909', '#cb9b09', '#ce9e0a', '#d0a009', '#d4a30a', '#d6a50a', '#d9a80b', '#dbaa0a', '#dead0b', '#e0ae0b', '#e3b10b', '#e6b40c', '#e7b50c', '#e9b80e', '#eab80e', '#ebba10', '#ecbb10', '#eebd12', '#eebe12', '#f0c014', '#f0c014', '#f2c316', '#f2c316', '#f4c518', '#f5c719', '#f6c719', '#f8ca1b', '#f8ca1b', '#facc1c', '#facd1d', '#fccf1e', '#fcd01f', '#fed220', '#ffd321', '#ffd422', '#ffd522', '#ffd724', '#ffd724', '#ffd825', '#ffda27', '#ffda27', '#ffdc29', '#ffdc29', '#ffde2b', '#ffdf2b', '#ffe12d', '#ffe12d', '#ffe32f', '#ffe32f', '#ffe530', '#ffe531', '#ffe732', '#ffe833', '#ffe93e', '#ffeb4e', '#ffec5b', '#ffef6b', '#fff078', '#fff288', '#fff395', '#fff5a5', '#fff6b2', '#fff8c2', '#fff9cf', '#fffcdf', '#fffdec', '#fffefc']
colorcet.kmrpw = ['#010000', '#090201', '#100201', '#180402', '#1f0402', '#280704', '#2f0703', '#370905', '#3e0904', '#470b06', '#4e0c06', '#530d07', '#550c07', '#5a0e08', '#5d0d08', '#610e0a', '#640e0a', '#690f0b', '#6c0f0b', '#70100c', '#73100c', '#77110e', '#7b110e', '#7d110e', '#821210', '#851210', '#891311', '#8c1311', '#911412', '#931412', '#981514', '#9b1413', '#9f1615', '#a21515', '#a61616', '#a81616', '#ab1616', '#ad1515', '#af1616', '#b11515', '#b41616', '#b51515', '#b81515', '#ba1515', '#bc1414', '#be1515', '#c01414', '#c31515', '#c41414', '#c71414', '#c91313', '#cc1414', '#cd1313', '#d01414', '#d11313', '#d41313', '#d61313', '#d81313', '#da1212', '#dd1313', '#de1212', '#e11212', '#e31212', '#e61212', '#e71212', '#e91111', '#ec1212', '#ed1111', '#f01111', '#f21010', '#f51111', '#f61010', '#f91111', '#fa1010', '#fd1111', '#ff1010', '#ff1010', '#ff0e0e', '#ff0e0e', '#ff0c0c', '#ff0c0c', '#ff0b0b', '#ff0b0b', '#ff0909', '#ff0909', '#ff0808', '#ff0707', '#ff0606', '#ff0505', '#ff0505', '#ff0303', '#ff0303', '#ff0101', '#ff0101', '#ff0000', '#ff0007', '#ff000d', '#ff0115', '#ff001c', '#ff0024', '#ff002a', '#ff0132', '#ff0039', '#ff0041', '#ff0047', '#ff0150', '#ff0056', '#ff005d', '#ff0165', '#ff006b', '#ff0073', '#ff007a', '#ff0182', '#ff0088', '#ff0f91', '#ff2099', '#ff34a2', '#ff45aa', '#ff58b3', '#ff69ba', '#ff7dc3', '#ff8ecb', '#ffa1d4', '#ffb2dc', '#ffc6e5', '#ffd7ed', '#ffeaf6', '#fffcfd']
colorcet.kmrypw = ['#000000', '#0a0201', '#140301', '#1f0502', '#290603', '#340804', '#3e0905', '#480b06', '#500c06', '#520c06', '#530c06', '#560c06', '#570c07', '#5a0c07', '#5b0c07', '#5d0c07', '#5f0c07', '#610c07', '#630c07', '#650c07', '#670c07', '#690d08', '#6a0c07', '#6c0c08', '#6e0c08', '#700d08', '#720c08', '#740c08', '#760c08', '#780d09', '#7a0c08', '#7c0c09', '#7e0c09', '#800d09', '#810c09', '#830d09', '#850d09', '#870d09', '#890c09', '#8b0d09', '#8d0d0a', '#8f0d0a', '#900c0a', '#920d0a', '#950d0a', '#960c0a', '#980d0a', '#9a0d0a', '#9c0d0b', '#9e0d0a', '#a00d0b', '#a20d0b', '#a40d0b', '#a50d0b', '#a80d0b', '#a90d0b', '#ac0d0b', '#ad0d0b', '#af0d0b', '#b10d0c', '#b30d0c', '#b50d0c', '#b70d0c', '#b90d0c', '#bb0d0c', '#bc0d0c', '#be0d0c', '#c00d0c', '#c20d0d', '#c40d0c', '#c60d0d', '#c80d0d', '#ca0d0d', '#cc0d0d', '#ce0d0d', '#d00c0c', '#d20c0c', '#d40b0b', '#d70b0b', '#d90a0a', '#db0a0a', '#dd0909', '#df0808', '#e10808', '#e40707', '#e60707', '#e80606', '#ea0606', '#ec0505', '#ef0404', '#f10404', '#f30303', '#f50202', '#f70202', '#f90202', '#fc0101', '#fe0000', '#ff0603', '#ff1508', '#ff240e', '#ff3314', '#ff421a', '#ff5120', '#ff6026', '#ff6f2b', '#ff7e31', '#ff7d3b', '#ff6e49', '#ff5f56', '#ff5064', '#ff4171', '#ff327f', '#ff238c', '#ff149a', '#ff05a7', '#ff0db1', '#ff21b7', '#ff35be', '#ff4ac4', '#ff5ecb', '#ff72d1', '#ff86d8', '#ff9ade', '#ffaee5', '#ffc2eb', '#ffd6f2', '#ffeaf8', '#ffffff']
colorcet.krpw = ['#010000', '#0a0201', '#140302', '#1f0503', '#290603', '#340804', '#3e0905', '#490b06', '#500c06', '#530c06', '#560c06', '#5a0c06', '#5c0b05', '#600b06', '#630b05', '#660b05', '#690a05', '#6c0a05', '#6f0a05', '#720a05', '#750905', '#780905', '#7b0904', '#7f0905', '#810904', '#850904', '#880804', '#8b0804', '#8e0804', '#910804', '#940704', '#970704', '#9a0703', '#9d0704', '#a00603', '#a40603', '#a60603', '#aa0603', '#ad0603', '#b00603', '#b30503', '#b60503', '#b90502', '#bc0502', '#bf0502', '#c20402', '#c50402', '#c80402', '#cb0402', '#ce0302', '#d20302', '#d40301', '#d80302', '#db0201', '#de0201', '#e10201', '#e40201', '#e70201', '#ea0201', '#ed0101', '#f00101', '#f30100', '#f70101', '#f90000', '#fd0000', '#ff0001', '#ff0005', '#ff0009', '#ff000d', '#ff0011', '#ff0016', '#ff0019', '#ff001e', '#ff0022', '#ff0026', '#ff002a', '#ff002e', '#ff0032', '#ff0036', '#ff003a', '#ff003f', '#ff0043', '#ff0047', '#ff004b', '#ff004f', '#ff0053', '#ff0057', '#ff005b', '#ff005f', '#ff0063', '#ff0067', '#ff006c', '#ff0070', '#ff0074', '#ff0078', '#ff007c', '#ff0080', '#ff0084', '#ff0088', '#ff008c', '#ff0090', '#ff0095', '#ff0098', '#ff009d', '#ff00a1', '#ff00a5', '#ff00a9', '#ff04ad', '#ff10b1', '#ff1db6', '#ff29b9', '#ff36be', '#ff42c2', '#ff4fc6', '#ff5bca', '#ff68ce', '#ff74d2', '#ff81d6', '#ff8dda', '#ff9ade', '#ffa7e2', '#ffb3e7', '#ffc0ea', '#ffcdef', '#ffd9f3', '#ffe6f7', '#fff2fb', '#ffffff']

# GOOD_CMAPS = ('koyw', 'krpw', 'kmrpw', 'kmrypw', 'CET_L16', 'CET_CBTL1', 'CET_L1', 'CET_L2', 'kbc', 'bmw', 'CET_L17')
# FAVORITE_CMAPS = ('koyw', 'krpw', 'CET_L16', 'CET_L2', 'kbc', 'bmw')



def lerp(a:float, b:float, t:float) -> float:
	return a*(1-t)+b*t

def inv_lerp(a:float, b:float, x:float) -> float:
	return (x-a)/(b-a)

def viewport_corners(re:float, im:float, zoom:float, width:int, height:int) -> tuple[float, float, float, float]:
	w = width / height * zoom
	return re - w, im - zoom, re + w, im + zoom

def render_results(results:np.ndarray, width:int, height:int) -> np.ndarray:
	img = np.zeros((height, width, 3), dtype=np.int8)
	for i in range(height):
		for j in range(width):
			if results[i, j, 0] == -1:
				img[i, j] = 0, 0, 0
			else:
				# m = (results[i, j, 1] * results[i, j, 1] + results[i, j, 2] * results[i, j, 2]) ** .5
				# p = atan2(results[i, j, 2], results[i, j, 1])
				r, g, b = get_color_from_palette(palette, COLOR_RATE * (results[i, j, 0] * results[i, j, 0]))
				img[i, j, 0] = b
				img[i, j, 1] = g
				img[i, j, 2] = r
	img -= 128
	return img

def render(re:float, im:float, zoom:float, width:int, height:int, max_iterations:int=1000, workers=8, use_ap=False) -> tuple[np.ndarray, float, float]:
	global current_re_min, current_im_min, current_re_max, current_im_max
	current_re_min, current_im_min, current_re_max, current_im_max = viewport_corners(re, im, zoom, WIDTH, HEIGHT)
	results = test_region_mp(current_re_min, current_im_min, current_re_max, current_im_max, WIDTH, HEIGHT, workers=workers, max_iterations=max_iterations, prec=current_zoom+6 if use_ap else 16)
	et0 = sw.lap()
	img = render_results(results, WIDTH, HEIGHT)
	et1 = sw.lap()
	return img, et0, et1

def img_click(event, x, y, flags, param):
	global img, current_zoom
	view_adjusted = False
	if event == cv2.EVENT_LBUTTONDOWN:
		re = lerp(current_re_min, current_re_max, inv_lerp(0, WIDTH-1, x))
		im = lerp(current_im_min, current_im_max, inv_lerp(0, HEIGHT-1, y))
		current_zoom += ZOOM_RATE
		if using_ap:
			z = mpm.mpf(10)**-current_zoom
		else:
			z = 10**-current_zoom
		img, et0, et1 = render(re, im, z, WIDTH, HEIGHT, use_ap=using_ap, workers=NUM_WORKERS)
		view_adjusted = True
	elif event == cv2.EVENT_RBUTTONDOWN:
		current_zoom -= ZOOM_RATE
		if using_ap:
			z = mpm.mpf(10)**-current_zoom
		else:
			z = 10**-current_zoom
		img, et0, et1 = render((current_re_min + current_re_max) * 0.5, (current_im_min + current_im_max) * 0.5, z, WIDTH, HEIGHT, use_ap=using_ap, workers=NUM_WORKERS)
		view_adjusted = True
	if view_adjusted:
		print(f'{et0:0f} ms / {et1:0f} ms')
		print(f'\tCenter: {re} + {im}*i')
		print(f'\tZoom: 10^-{current_zoom}')
		print()


class MandelbrotRenderer:
	def __init__(self, dps: int = 16):
		mpm.mp.dps = dps


		self.using_ap = dps != 16 # arbitrary precision mode
		self.current_zoom = 13
		self.current_re_min, self.current_im_min, self.current_re_max, self.current_im_max = None, None, None, None

		sw.start()
		# results = test_region_mp(-2.5, -1, 1, 1, 1920, 1080, workers=8, max_iterations=1000)
		self.img, self.et0, self.et1 = render(mpm.mpf(-0.562439654643187), mpm.mpf(0.6423877496042704), 10**-current_zoom, WIDTH, HEIGHT, use_ap=using_ap, workers=NUM_WORKERS)
		print(f'{self.et0:0f} ms / {self.et1:0f} ms')
		cv2.namedWindow('Render')
		cv2.setMouseCallback('Render', img_click)
		# cv2.imshow(f'Render ({et0:0f} ms / {et1:0f} ms)', img)
		# cv2.waitKey()
	def display(self):
		while True:
			cv2.imshow('Render', img)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('r'):
				self.img, self.et0, self.et1 = render(-0.75, 0, 1, WIDTH, HEIGHT)
				print(f'{self.et0:0f} ms / {self.et1:0f} ms')
			elif key == ord('a'):
				using_ap = not using_ap
			elif key == 27: # ESC key
				break


if __name__ == '__main__':
	sw = Stopwatch()

	palette = convert_palette(colorcet.koyw)
	palette = cycle_palette(palette)
	p = palette
	for i in range(1, 6):
		p.extend(shift_palette_hue(palette, i/6.))
	palette = p
	del p

	COLOR_RATE = 0.05
	WIDTH = 256
	HEIGHT = 256
	# ZOOM_RATE = 0.301029996
	ZOOM_RATE = 1
	NUM_WORKERS = 15

	rend = MandelbrotRenderer()
	rend.display()

