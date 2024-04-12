
import os
import numpy as np
import cv2
import pandas as pd
import datashader as ds
import colorcet as cc
import matplotlib.pyplot as plt
import mpmath as mp
from fischer.dt import dt

from multiprocessing import Pool

from gifgen import images_to_mp4



def trajectory(f, v0, iterations, clip_std, clip_iterations, *f_args):
	traj = np.empty((iterations+1,2), dtype=float)
	traj[0,0] = v0.real
	traj[0,1] = v0.imag
	v = v0
	for i in range(1, iterations+1):
		v = f(v, *f_args)
		if v is None:
			break
		traj[i,0] = v.real
		traj[i,1] = v.imag
	if clip_iterations > 0 and clip_iterations < len(traj):
		traj = traj[clip_iterations:]
	if clip_std is not None:
		# print(f'Clipping points outside of clip_std={clip_std} standard deviations.')
		Re, Im = traj.T
		re_center, im_center = np.mean(traj, axis=0)
		dist = np.sqrt(np.square(Re - re_center) + np.square(Im - im_center))
		dist_mean = np.mean(dist)
		dist_std = np.std(dist)
		dist_z = (dist - dist_mean) / dist_std
		non_outliers = dist_z < clip_std
		traj = traj[non_outliers]
		# print(f'Removed {np.sum(non_outliers==False)} outliers.')
	return traj

def trajectory_mp(f, iterations_per_worker, batches, clip_std = None, clip_iterations = 16, workers = 8) -> list[complex]:
	'''
	Call many iterations of trajectory() with multiprocessing to improve computation times.  One trajectory can only be processed by one process at a time, but multiple workers allow for multiple processes to simulate many trajectories, greatly increasing overall performance.

	Each worker simulates a trajectory with exactly `iterations_per_worker` steps, and there are `batches * workers` trajectories to be simulated overall. The returned array has the shape `(workers * batches * iterations_per_worker, 2)`.
	'''
	print(f'Total points to simulate among trajectories: {combined.shape[0]}')
	print('Simulating...')
	with Pool(workers) as pool:
		results = pool.starmap(trajectory, (
			(f, complex(float(np.random.normal(0, 1)), float(np.random.normal(0, 1))), iterations_per_worker, clip_std, clip_iterations)
			for _ in range(workers * batches)
		))
	print('Indexing...')
	combined = np.empty((sum(len(t) for t in results), 2), dtype=float)
	i = 0
	for t in results:
		combined[i:i+t.shape[0]] = t
		i += t.shape[0]
	return combined

def trajectory_end_behaviors(f, v0s, iterations, extra_iterations, *args):
	traj = []
	for v0 in v0s:
		traj.extend(trajectory(f, v0, iterations+extra_iterations, *args)[iterations:])
	return traj


def trajectory_graph(traj):
	plt.scatter([z.real for z in traj], [z.imag for z in traj], color='black', marker='.')
	plt.show()


def trajectory_image(traj, width = 200, height = 200, rotation = 0, cmap = cc.kg, background_color = 'black', save_file_name = None, show_image = True):
	if save_file_name is None: save_file_name = dt()
	if type(traj[0]) is complex or type(traj[0]) is mp.mpc:
		A = np.array([[float(z.real), float(z.imag)] for z in traj])
	elif type(traj[0]) is tuple:
		A = np.array(traj)
	elif type(traj) is np.ndarray:
		A = traj
	else:
		raise TypeError(f'Invalid type received for traj: {type(traj)}')
	if rotation != 0: A = np.dot(A, np.array([[np.math.cos(rotation), -np.math.sin(rotation)], [np.math.sin(rotation), np.math.cos(rotation)]]))
	if show_image:
		cv2.namedWindow('Trajectory Image', cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty('Trajectory Image',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	df = pd.DataFrame(data = {'x': A[:,0], 'y': A[:,1]})
	agg = ds.Canvas(width, height).points(df, 'x', 'y')
	im = ds.tf.set_background(ds.tf.shade(agg, cmap=cmap), background_color)
	ds.utils.export_image(im, save_file_name)
	im = cv2.imread(save_file_name + '.png')
	if show_image:
		cv2.imshow('Trajectory Image', im)
		cv2.waitKey(0)
	return im


def trajectory_animation(f, v0f, iterations, argf, frames, image_directory, destination_file_name, **kwargs):
	if isinstance(v0f, (float, complex)):
		z = v0f
		v0f = lambda i: z
	if isinstance(argf, tuple):
		a0, a1 = argf
		inv_frames = 1./(frames-1)
		argf = lambda i: (a0 * (1-i*inv_frames) + a1 * (i*inv_frames),)
	if not os.path.isdir(image_directory):
		os.makedirs(image_directory)
	images = []
	for frame in range(frames):
		images.append(trajectory_image(trajectory(f, v0f(frame), iterations, *argf(frame)), show_image = False, save_file_name = f'{image_directory}/{str(frame).zfill(5)}', **kwargs))
	images_to_mp4(images, destination_file_name = destination_file_name)


