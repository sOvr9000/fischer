
import os
import cv2
import numpy as np
from itertools import product


offsets = (1, 0, -1)

def pack(path, destination_fpath, cell_width, cell_height, source_format='png', aspect_ratio='13:8', verbose=True, stretch=True):
	ratio_w, ratio_h = map(int,aspect_ratio.split(':'))
	ims = []
	if verbose:
		print('Searching directories for sprites...')
	for curpath, _, fnames in os.walk(path):
		if verbose:
			print(curpath)
		found_here = 0
		for fname in fnames:
			if fname[-len(source_format):] != source_format:
				continue
			im = cv2.imread(curpath + '/' + fname, cv2.IMREAD_UNCHANGED)
			if im.shape[0] != cell_height or im.shape[1] != cell_width:
				if stretch:
					im = cv2.resize(im, (cell_width, cell_height), interpolation=cv2.INTER_CUBIC)
				else:
					# print(im.shape)
					if im.shape[0] > cell_height:
						im = cv2.resize(im, (int(.5 + im.shape[1] * cell_width / im.shape[0]), cell_height), interpolation=cv2.INTER_CUBIC)
					# print(im.shape)
					if im.shape[1] > cell_width:
						im = cv2.resize(im, (cell_width, int(.5 + im.shape[0] * cell_height / im.shape[1])), interpolation=cv2.INTER_CUBIC)
					# print(im.shape)
					if im.shape[0] < cell_height or im.shape[1] < cell_width:
						A = np.zeros((cell_height, cell_width, 4))
						x = int(.5+(cell_width-im.shape[1])*.5)
						y = int(.5+(cell_height-im.shape[0])*.5)
						A[y:y+im.shape[0], x:x+im.shape[1]] = im
						im = A
			# print(im.shape)
			ims.append(im)
			found_here += 1
		if verbose:
			print(f'Found here: {found_here}')
	total = len(ims)
	if verbose:
		print(f'Total sprites found: {total}')
	rows = int(.5 + (total * ratio_h / ratio_w) ** .5)
	cols = int(.5 + total / rows)
	bestr = rows
	bestc = cols
	best = total - bestr * bestc
	for dr, dc in product(offsets, offsets):
		a = (rows + dr) * (cols + dc)
		if a >= total and total - a < best:
			best = a
			bestr = rows + dr
			bestc = cols + dc
			if best == 0:
				break
	rows = bestr
	cols = bestc
	ss = np.zeros((cell_height*rows, cell_width*cols, 4))
	for (row, col), im in zip(product(range(rows), range(cols)), ims):
		ss[row*cell_height:(row+1)*cell_height, col*cell_width:(col+1)*cell_width] = im
	cv2.imwrite(destination_fpath, ss)
	if verbose:
		print(f'Generated spritesheet at {destination_fpath} with {rows} rows and {cols} columns.')

def unpack(sprite_sheet_fpath, rows, cols, destination_path, destination_name_prefix, pad_width=None, pad_height=None, destination_format='png', verbose=True):
	'''
	Unpack a sprite sheet into multiple image files.

	`destination_name_prefix` is what each generated image file name is prefixed with to avoid overwriting previously generated image files.

	`pad_width` is the width of the newly generated images after padding evenly on both sides with transparent pixels.  If `pad_width` is None, then no padding is done.  Same thing for `pad_height` but for height.
	'''
	if not os.path.isdir(destination_path):
		if verbose:
			print(f'Destination path not found: {destination_path}.  Creating it automatically.')
		os.makedirs(destination_path)
	if verbose:
		print(f'Unpacking sprite sheet at {sprite_sheet_fpath}')
	im = cv2.imread(sprite_sheet_fpath, cv2.IMREAD_UNCHANGED)
	h, w, _ = im.shape
	for r in range(rows):
		y = int(0.5 + r * h / rows)
		ny = int(0.5 + (r+1) * h / rows)
		for c in range(cols):
			x = int(0.5 + c * w / cols)
			nx = int(0.5 + (c+1) * w / cols)
			new_im = im[y:ny,x:nx]
			if pad_width is not None:
				if new_im.shape[1] < pad_width:
					amount = (pad_width - new_im.shape[1]) // 2
					new_im = np.concatenate((np.zeros((new_im.shape[0], amount, new_im.shape[2])), new_im, np.zeros((new_im.shape[0], pad_width - new_im.shape[1] - amount, new_im.shape[2]))), axis=1)
			if pad_height is not None:
				if new_im.shape[0] < pad_height:
					amount = (pad_height - new_im.shape[0]) // 2
					new_im = np.concatenate((np.zeros((amount, new_im.shape[1], new_im.shape[2])), new_im, np.zeros((pad_height - new_im.shape[0] - amount, new_im.shape[1], new_im.shape[2]))), axis=0)
			fname = f'{destination_path}/{destination_name_prefix}_{r}_{c}.{destination_format}'
			if verbose:
				if pad_width is not None or pad_height is not None:
					print(f'Sprite size after padding: {new_im.shape}')
					if new_im.shape[0] != pad_height or new_im.shape[1] != pad_width:
						print(f'(NOT PADDED)')
				print(f'Saving sprite at row={r}, col={c} as {fname}')
			cv2.imwrite(fname, new_im)


