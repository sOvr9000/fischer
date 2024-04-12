
import numpy as _np
import json as _json
import matplotlib.pyplot as _plt
import tensorflow as _tf
import gym


class SimpleDQL:
	def __init__(self, model=None, memory_capacity=10000, steps_per_experience_replay=128, memory_sample_size=256, experience_replay_delay=100000, do_rendering=False):
		self.set_model(model)
		self.gamma = 0.98
		self.double_q_tau = 0.75
		self.do_rendering = do_rendering
		self.memory_capacity = memory_capacity
		self.steps_per_experience_replay = steps_per_experience_replay
		self.memory_sample_size = memory_sample_size
		self.experience_replay_delay = experience_replay_delay
		self.set_epsilon(1.0, 0.1, 1000, 0, 0.2)
	def full_reset(self):
		self.old_states = _np.zeros((self.memory_capacity, *self.state_shape))
		self.new_states = _np.zeros((self.memory_capacity, *self.state_shape))
		self.actions_taken = _np.zeros(self.memory_capacity, dtype=_np.int16)
		self.rewards_observed = _np.zeros(self.memory_capacity)
		self.termination = _np.zeros(self.memory_capacity)
		self.new_valid_action_masks = _np.zeros((self.memory_capacity, self.num_actions), dtype=_np.bool)
		self.er_rewards = [0] * self.steps_per_experience_replay # rewards received since last experience replay
		self.epsilon = self.epsilon_max
		self.mean_reward_per_step_history = []
		self.mean_er_reward_history = []
		self.mean_episode_length_history = []
		self.total_steps = 0
		self.total_reward = 0
		self.total_episodes = -1
		self.total_experience_replays = 0
		self.last_prediction = None
		self.action_symmetries = self.generate_action_symmetries().astype(_np.int32)
		self.action_symmetries_inv = _np.array([
			[
				_np.where(action_map == n)[0][0]
				for n in range(action_map.shape[0])
			]
			for action_map in self.action_symmetries
		], dtype=_np.int32)
		self.reset()
	def reset(self, env_index = 0):
		self.total_episodes += 1
		self.current_step = 0
		self.on_reset(env_index = env_index)
		self._current_state = self.get_state()
		self.current_reward = 0
		self.current_prediction = None
	def set_model(self, model):
		if type(model) is str:
			self.model = _tf.keras.models.load_model(model + '.h5')
		else:
			self.model = model
		self.target_model = _tf.keras.models.clone_model(self.model)
		self.target_model.set_weights(self.model.get_weights())
		self.state_shape = self.model.input_shape[1:]
		self.num_actions = self.model.output_shape[1]
	def on_reset(self, env_index = 0):
		raise NotImplementedError
	def _record_transition(self, old_state, new_state, action_taken, reward_observed, termination, new_valid_action_mask):
		termination = 1-int(termination)
		i = (self.total_steps * self.action_symmetries.shape[0]) % self.memory_capacity
		old_state_symmetries = self.get_state_symmetries(old_state)
		new_state_symmetries = self.get_state_symmetries(new_state)
		for n in range(self.action_symmetries.shape[0]):
			j = (i+n)%self.memory_capacity
			self.old_states[j] = old_state_symmetries[n]
			self.new_states[j] = new_state_symmetries[n]
			self.actions_taken[j] = self.action_symmetries[n, action_taken]
			self.rewards_observed[j] = reward_observed
			self.termination[j] = termination
			self.new_valid_action_masks[j] = new_valid_action_mask
	def _sample_memory(self, count):
		i = _np.random.randint(0, min(self.total_steps * self.action_symmetries.shape[0], self.memory_capacity), size = count)
		return self.old_states[i], self.new_states[i], self.actions_taken[i], self.rewards_observed[i], self.termination[i], self.new_valid_action_masks[i]
	def _experience_replay(self):
		self.total_experience_replays += 1
		old_states, new_states, actions_taken, rewards_observed, termination, new_valid_action_mask = self._sample_memory(count = self.memory_sample_size)
		prediction_old_states_online = self.model.predict(old_states, batch_size=256)
		prediction_new_states_online = self.model.predict(new_states, batch_size=256)
		prediction_new_states_target = self.target_model.predict(new_states, batch_size=256)
		adjusted_prediction_new_states_online = prediction_new_states_online - prediction_new_states_online.min(axis=1).reshape((-1,1))
		adjusted_prediction_new_states_online *= new_valid_action_mask
		indices = _np.arange(prediction_old_states_online.shape[0],dtype=_np.int32)
		prediction_old_states_online[indices,actions_taken] = rewards_observed + self.gamma * termination * prediction_new_states_target[indices,_np.argmax(adjusted_prediction_new_states_online, axis=1)]
		self.model.fit(old_states, prediction_old_states_online, batch_size=256, epochs=1, verbose=False)
		self.target_model.set_weights(self.model.get_weights())
		tw = self.target_model.get_weights()
		mw = self.model.get_weights()
		self.target_model.set_weights([
			tw[i] * (1-self.double_q_tau) + mw[i] * self.double_q_tau
			for i in range(len(tw))
		])
		self.on_experience_replay()
	def run_step(self):
		self.total_steps += 1
		self.current_step += 1

		va = self._get_valid_actions()
		self.last_prediction = self.current_prediction
		if _np.random.random() < self.epsilon:
			old_state = self._current_state
			action_taken = va[_np.random.randint(len(va))]
			self.current_prediction = None
			used_sym = False
		else:
			state_symmetries = self.get_state_symmetries(self._current_state)
			symmetry_index = _np.random.randint(len(state_symmetries))
			old_state = state_symmetries[symmetry_index]
			va = self.action_symmetries_inv[symmetry_index, va]
			self.current_prediction = self.predict(old_state)
			action_taken = va[_np.argmax(self.current_prediction[va])]
			action_taken = self.action_symmetries[symmetry_index, action_taken]
			used_sym = True

		reward_observed, termination = self.step(action_taken)

		self.current_reward = reward_observed
		new_valid_action_mask = _np.zeros(self.num_actions, dtype=_np.bool)
		new_valid_action_mask[self._get_valid_actions()] = True

		self.total_reward += reward_observed
		self.mean_reward_per_step = self.total_reward / self.total_steps
		self.er_rewards[self.total_steps % self.steps_per_experience_replay] = reward_observed

		self._current_state = self.get_state()
		if used_sym:
			new_state = self.get_state_symmetries(self._current_state)[symmetry_index]
		else:
			new_state = self._current_state

		if self.do_rendering:
			self.render()

		self._record_transition(old_state, new_state, action_taken, reward_observed, termination, new_valid_action_mask)
		if self.total_steps >= self.epsilon_delay:
			if (self.total_steps - self.epsilon_delay) % self.epsilon_reset_interval == 0:
				self.epsilon = self.epsilon_max
			else:
				self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
		if termination:
			self.reset()
		if self.total_steps % self.steps_per_experience_replay == 0:
			if self.total_steps >= self.experience_replay_delay: # only train when ready
				self._experience_replay()
			self.mean_reward_per_step_history.append(self.mean_reward_per_step)
			self.mean_er_reward = sum(self.er_rewards) / self.steps_per_experience_replay
			self.mean_er_reward_history.append(self.mean_er_reward)
			self.mean_episode_length = self.total_steps if self.total_episodes == 0 else self.total_steps / self.total_episodes
			self.mean_episode_length_history.append(self.mean_episode_length)
	def predict(self, state):
		return self.model.predict(state.reshape((1,*self.state_shape))).reshape((-1,))
	def predict_repr(self, state):
		pred = self.predict(state)
		m = _np.argmax(pred)
		return ' '.join(f'{v:.4f}' for v in pred) + ' | ' + str(m)
	def step(self, action):
		raise NotImplementedError
	def get_state(self):
		raise NotImplementedError
	def render(self):
		raise NotImplementedError
	def on_experience_replay(self):
		pass
	def _get_valid_actions(self):
		return _np.sort(self.get_valid_actions().astype(_np.int32))
	def get_valid_actions(self):
		return _np.arange(self.num_actions, dtype=_np.int32)
	def generate_action_symmetries(self):
		'''
		The number of actions is typically low, and is always fixed.  For this reason, the symmetric actions are pregenerated and serve as a look-up table.

		The returned array must be 2D.  The dimensions correspond to (symmetry, action).

		For example, if there is (pi/4) rotational and reflective symmetry (i.e. the D4 group), then this function returns an array of shape (8, num_actions).  The result is that each row is a mapping of the actions 0, 1, ..., num_actions-1 to the corresponding symmetric actions a0, a1, ... .

		By default, this function returns the array [[0, 1, ..., num_actions-1]], which represents the environment having only the identity symmetry.
		'''
		return _np.arange(self.num_actions, dtype=_np.int32).reshape((1,-1))
	def get_state_symmetries(self, state):
		'''
		Return all symmetries of the given state, applying the symmetric operations to the state in the same order as that which was used to generate the action symmetries.

		For example, if there is (pi/4) rotational and reflective symmetry (i.e. the D4 group), then this function returns a state for each of the eight symmetries. Then, if the action symmetries follow the form [identity, rotate pi/4, rotate pi/2, rotate 3pi/4, reflect, reflect rotate pi/4, reflect rotate pi/2, reflect rotate 3pi/4] along axis 0, the eight symmetries returned by this function must be from those corresponding symmetries, in the same order.
		'''
		return state.reshape((1,*state.shape))
	def save_model(self, fp=None):
		if fp is None:
			fp = './simplebot'
		self.model.save(fp + '.h5')
		with open(fp + '_data.json', 'w') as f:
			f.write(_json.dumps(self._modify_json_data({
				'mean_reward_per_step': self.mean_reward_per_step_history,
				'mean_er_reward': self.mean_er_reward_history,
				'mean_episode_length': self.mean_episode_length_history,
			}), indent = 4))
	def _modify_json_data(self, d):
		self.modify_json_data(d)
		return d
	def modify_json_data(self, d):
		pass
	def auto_save(self, fp = None, step_interval = None):
		if self.total_steps < self.experience_replay_delay: # model is not training so don't save yet
			return
		if step_interval is None:
			step_interval = self.steps_per_experience_replay*5
		if self.total_steps % step_interval == 0:
			self.save_model(fp = fp)
	def set_epsilon(self, start_value = 1.0, end_value = 0.1, decay_interval = 1000, delay = 0, reset_rate = 0.2):
		'''
		Set the constraints of epsilon (exploration parameter).  Initially, it is set to start_value.  Over the course of decay_interval steps, epsilon exponentially decays down to end_value.  This assumes end_value <= start_value.  Epsilon does not decrease until total_steps >= delay.
		'''
		self.epsilon_max = start_value
		self.epsilon_min = end_value
		self.epsilon_decay = (self.epsilon_min / self.epsilon_max) ** (1. / max(decay_interval,1))
		self.epsilon_delay = delay
		self.epsilon_reset_interval = decay_interval / reset_rate if reset_rate is not None and (reset_rate > 0 or decay_interval > 0) else _np.inf
		self.epsilon = self.epsilon_max





def smooth_data(data, smoothing=2):
	if len(data) <= 2:
		return data
	smoothing = max(1, min(len(data)-2, smoothing))
	return [
		_np.mean(data[i:i+smoothing+1])
		for i in range(len(data)-smoothing-1)
	]



def plot_data(fp = None, smoothing = 1, start = 0, end = _np.inf):
	if fp is None:
		fp = './simplebot'
	data = _json.load(open(fp + '_data.json', 'r'))

	s = max(0,start)
	e = min(len(data['mean_reward_per_step']),end)
	_plt.title('Mean reward per step')
	_plt.xlabel('Experience replays')
	_plt.ylabel('Reward')
	_plt.plot(data['mean_reward_per_step'][s:e])
	_plt.show()

	smoothed = smooth_data(data['mean_er_reward'], smoothing)
	e = min(len(smoothed),end)
	_plt.title(f'Mean reward since last {smoothing} experience replays')
	_plt.xlabel('Experience replays')
	_plt.ylabel('Reward')
	_plt.plot(smoothed[s:e])
	_plt.show()

	e = min(len(data['mean_episode_length']),end)
	_plt.title('Mean episode length')
	_plt.xlabel('Experience replays')
	_plt.ylabel('Steps')
	_plt.plot(data['mean_episode_length'][s:e])
	_plt.show()

	for k,v in data.items():
		if k not in ('mean_reward_per_step', 'mean_er_reward', 'mean_episode_length'):
			smoothed = smooth_data(v, smoothing)
			e = min(len(smoothed),end)
			_plt.title(f'"{k}" | Mean over previous {smoothing} values')
			_plt.xlabel('Data points')
			_plt.ylabel('Value')
			_plt.plot(smoothed[s:e])
			_plt.show()



class GymDQL(SimpleDQL):
	'''
	A SimpleDQL with easy Open AI Gym environment integration.  Instances can be created directly from this class, and everything will work.

	Instead of overriding SimpleDQL.get_state(), you should override GymDQL.convert_gym_state().  There is no need to override SimpleDQL.on_reset(), SimpleDQL.step(), or SimpleDQL.render().
	'''
	def __init__(self, env, convert_gym_state = lambda gym_state: gym_state, **kwargs):
		'''
		env can be the name of a registered Open AI Gym environment, such as 'Taxi-v3' or 'CartPole-v1', or it can be an instance of the Gym class, which is created with gym.make() or custom-made.
		'''
		super().__init__(**kwargs)
		if type(env) is str:
			self.gym_env = gym.make(env)
		else:
			self.gym_env = env
		self.convert_gym_state = convert_gym_state # default to leaving it unchanged
	def on_reset(self):
		self.gym_state = self.gym_env.reset()
		self.converted_gym_state = self.convert_gym_state(self.gym_state)
		self.gym_step_info = {}
	def step(self, action):
		self.gym_state, reward, done, self.gym_step_info = self.gym_env.step(action)
		self.converted_gym_state = self.convert_gym_state(self.gym_state)
		return reward, done
	# def convert_gym_state(self, gym_state):
	# 	'''
	# 	Convert the gym state to one that is better suited for a neural network.  For example, discrete states are represented as integers, but a neural network is more easily able to understand the expanded form of such information.
		
	# 	Conversion from integers to tuples of integers can be done with numpy.unravel_index(), and converted back with numpy.ravel_multi_index().

	# 	Not only can the forms of states be converted.  Normalization can (and should) also happen here.  State values should not have a mean far from zero, and keeping standard deviation near one helps to prevent vanishing/exploding gradients in parameter updating.

	# 	By default, this function leaves the state alone, not converting its form nor normalizing it.
	# 	'''
	# 	return gym_state
	def get_state(self):
		return self.converted_gym_state
	def render(self):
		self.gym_env.render()





class TwoPlayerGameDQL:
	def __init__(self, env):
		self.env = env




