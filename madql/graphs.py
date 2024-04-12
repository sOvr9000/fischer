
import numpy as np
import matplotlib.pyplot as plt


def first_differences(x):
	return x[1:] - x[:-1]

def second_differences(x):
	return first_differences(first_differences(x))

def third_differences(x):
	return second_differences(first_differences(x))


def plot_model_history(fpath, line_of_best_fit = True, grid_lines = False, start = 0, end = None, smoothing = 0, sim_history_floc = None, first_difs = True, second_difs = False, third_difs = False):
	'''
	Plot model history directly from a .npz file.

	History files are generated along with the .h5 files when a model gets saved during simulation.

	smoothing should be a positive integer.  It represents the radius around which each point is averaged with other points, to smooth out the curve.

	show_ids is a list of agent_id's.  The agent_id's found in this list are the ones for which this graph will show data.  If show_ids = None, then data will be shown for all agents provided in fpaths.
	'''
	if type(fpath) is str:
		plot_model_history([fpath], line_of_best_fit = line_of_best_fit, grid_lines = grid_lines, start = start, end = end, smoothing = smoothing, sim_history_floc = sim_history_floc, first_difs = first_difs, second_difs = second_difs, third_difs = third_difs)
	elif type(fpath) is list:
		plt.title('Mean accumulated reward over last 100 episodes')
		plt.xlabel('Episode')
		plt.ylabel('Reward')
		if grid_lines:
			plt.grid(grid_lines, c = (0, 0, 0))
		for fp in fpath:
			data = np.load(fp)
			if end is None:
				end = len(data['recent_reward_history'])
			x = np.arange(start, end)
			y = data['recent_reward_history'][start:end]
			l = len(y) - 1
			if smoothing > 0 and (type(smoothing) is int or (smoothing := int(smoothing) is not None)):
				print('Smoothing curve... ({} averages to be taken, {} iterations per average, approximately {} total iterations)'.format(l, 2 * smoothing + 1, l * (2 * smoothing + 1)))
				y = np.array([
					np.mean(y[max(i-smoothing, 0):min(i+smoothing+1, l)])
					for i in range(l + 1)
				])
			plt.plot(x, y, label = 'Agent {}'.format(data['agent_id']))
			if line_of_best_fit:
				m, b = np.polyfit(x, y, 1)
				plt.plot([x[0], x[-1]], [m * x[0] + b, m * x[-1] + b], label = 'Agent {} best fit'.format(data['agent_id']))
			if first_difs:
				plt.plot(x[:-1], first_differences(y), label = 'Agent {} first difference'.format(data['agent_id']))
			if second_difs:
				plt.plot(x[:-2], second_differences(y), label = 'Agent {} second difference'.format(data['agent_id']))
			if third_difs:
				plt.plot(x[:-3], third_differences(y), label = 'Agent {} third difference'.format(data['agent_id']))
		plt.legend(loc = 'best')
		plt.show()
		if sim_history_floc is not None and type(sim_history_floc) is str:
			fp = '{}/sim_history.npz'.format(sim_history_floc)
			try:
				data = np.load(fp)
			except:
				print('Could not find sim history at file path {}'.format(fp))
			finally:
				plt.title('Mean steps per episode over last 100 episodes')
				x = np.arange(start, end)
				y = data['recent_episode_length_history'][start:end]
				l = len(y) - 1
				if smoothing > 0 and (type(smoothing) is int or (smoothing := int(smoothing) is not None)):
					y = np.array([
						np.mean(y[max(i-smoothing, 0):min(i+smoothing+1, l)])
						for i in range(l + 1)
					])
				plt.xlabel('Episode')
				plt.ylabel('Steps')
				plt.plot(x, y, label = 'Steps')
				if first_difs:
					plt.plot(x[:-1], first_differences(y), label = 'First difference')
				if second_difs:
					plt.plot(x[:-2], second_differences(y), label = 'Second difference')
				if third_difs:
					plt.plot(x[:-3], third_differences(y), label = 'Third difference')
				plt.legend(loc = 'best')
				plt.show()
				if 'recent_episode_duration_history' in data:
					plt.title('Mean milliseconds per episode over last 100 episodes')
					x = np.arange(start, end)
					y = data['recent_episode_duration_history'][start:end]
					l = len(y) - 1
					if smoothing > 0 and (type(smoothing) is int or (smoothing := int(smoothing) is not None)):
						y = np.array([
							np.mean(y[max(i-smoothing, 0):min(i+smoothing+1, l)])
							for i in range(l + 1)
						])
					plt.xlabel('Episode')
					plt.ylabel('Milliseconds')
					plt.plot(x, y, label = 'Milliseconds')
					if first_difs:
						plt.plot(x[:-1], first_differences(y), label = 'First difference')
					if second_difs:
						plt.plot(x[:-2], second_differences(y), label = 'Second difference')
					if third_difs:
						plt.plot(x[:-3], third_differences(y), label = 'Third difference')
					plt.legend(loc = 'best')
					plt.show()
				else:
					print('recent_episode_duration_history not found in sim history data')
		for fp in fpath:
			data = np.load(fp)
			plt.title('Agent {} Q-value predictions over last 100 episodes'.format(data['agent_id']))
			for a,b in (('recent_min_q', 'Minimum Q'), ('recent_max_q', 'Maximum Q'), ('recent_mean_q', 'Mean Q')):
				if a not in data:
					print(f'Could not find data for \'{a}\' ({b}) from agent history data at path {fp}')
					continue
				end = min(end,len(data[a]))
				x = np.arange(start, end)
				y = data[a][start:end]
				l = len(y) - 1
				if smoothing > 0 and (type(smoothing) is int or (smoothing := int(smoothing) is not None)):
					y = np.array([
						np.mean(y[max(i-smoothing, 0):min(i+smoothing+1, l)])
						for i in range(l + 1)
					])
				plt.xlabel('Episode')
				plt.ylabel('Q value')
				plt.plot(x, y, label = b)
				if first_difs:
					plt.plot(x[:-1], first_differences(y), label = f'{b} first difference')
				if second_difs:
					plt.plot(x[:-2], second_differences(y), label = f'{b} second difference')
				if third_difs:
					plt.plot(x[:-3], third_differences(y), label = f'{b} third difference')
			plt.legend(loc = 'best')
			plt.show()
	else:
		raise TypeError('fpath must be a file path or list of file paths')


