

import cv2
import glob
import numpy as np




def directory_to_mp4(directory, destination_file_name, file_extension = 'png', verbose = 0, frame_interval = 1, **kwargs):
	'''
	Combine all images immediately under a directory into a single MP4.

	frame_interval can be set to a higher number to create a timelapse.  For example, frame_interval=2 means the video is twice as fast, skipping every other frame.
	'''
	if type(directory) is not str:
		raise Exception('directory must be a string')
	if type(file_extension) is not str or len(file_extension) == 0:
		raise Exception('file_extension must be a nonempty string')
	if type(frame_interval) is not int or frame_interval <= 0:
		raise Exception('frame_interval must be a positive integer')
	if 'channel_order' in kwargs and kwargs['channel_order'] != 'bgr':
		raise Exception('channel_order cannot be overridden in directory_to_mp4().  It must be \'bgr\'.')
	try:
		file_names = glob.glob(directory + '\\*.' + file_extension)
	except Exception:
		raise Exception(f'Invalid directory: {directory}')
	try:
		images = []
		k = 0
		for file_name in file_names:
			k += 1
			if verbose and k % 5 == 0:
				print(f'{k} / {len(file_names)}')
			if k % frame_interval == 0:
				images.append(cv2.imread(file_name))
				#cv2.imshow('abc', images[-1])
				#cv2.waitKey(0)
	except Exception as e:
		raise Exception(f'Failed to read an image under the provided directory. Error: {e}')
	return images_to_mp4(images, destination_file_name, verbose = verbose, channel_order = 'bgr', **kwargs)


def images_to_mp4(images, destination_file_name, frame_rate = 60, verbose = 0, channel_order = 'bgr'):
	'''
	Convert a list or NumPy array of images (each being NumPy arrays of equal dimensions) to a .mp4 file.

	channel_order is only used if each image is a 3D array.  The character permutation reflects the channel order of RGB values in each image.  The cv2 module expects BGR ordering, whereas PIL expects RGB ordering.  Basically, channel_order should be 'bgr' if the provided images were obtained with cv2, and it should be 'rgb' if they were obtained with PIL.  channel_order can be any permutation of the string 'rgb', non-case-sensitive.
	'''
	if type(images) is not list and type(images) is not np.ndarray:
		raise Exception('images must either be a numpy.ndarray or list')
	if len(images) <= 1:
		raise Exception('images list/array must have two or more images')
	if type(destination_file_name) is not str or len(destination_file_name) <= 4:
		raise Exception('destination_file_name must be a string of five or more characters, ending in \'.mp4\'.')
	if '.mp4' != destination_file_name[-4:].lower():
		raise Exception(f'destination_file_name = \'{destination_file_name}\' does not have the .mp4 file extension.')
	imshape = images[0].shape
	if not all(im.shape == imshape for im in images[1:]):
		raise Exception('All provided images must be of the same dimensions')
	vid = cv2.VideoWriter(destination_file_name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, images[0].shape[1::-1])
	if len(images[0].shape) == 3:
		channel_order = channel_order.lower()
		channel_order = [channel_order.index('b'), channel_order.index('g'), channel_order.index('r')]
		for im in images:
			vid.write(im[:,:,channel_order])
	else:
		for im in images:
			vid.write(im)
	vid.release()


