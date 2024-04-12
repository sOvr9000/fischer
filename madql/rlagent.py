

'''
The following methods MUST be overridden:
- RLAgent.get_state()
- RLAgent.get_reward()

The following methods can be overridden for better control:
- RLAgent.get_current_valid_actions()		| For most implementations, you will want to override this.
- RLAgent.on_reset()
- RLAgent.on_pre_step()
- RLAgent.on_step()							| For most implementations, you will want to override this.
- RLAgent.on_post_step()
- RLAgent.do_actions()						| Logic here can be handled in RLAgent.on_step() instead.  This is just to further organize code.
-													| Better to implement do_actions() if RLEnvironment.parallel_agent_stepping = True  (that is the default setting)
- RLAgent.episode_ended()
- RLAgent.on_pre_experience_replay()
- RLAgent.on_post_experience_replay()
- RLAgent.get_heuristic_actions()
- RLAgent.on_kill()
- RLAgent.get_state_symmetries()
- RLAgent.get_action_symmetries()
- RLAgent.show_evaluations()				| Required if RLAgent.do_show_evaluations = True
'''



import numpy as np
from keras.models import clone_model, load_model as load_keras_model, Sequential
from madql.rlmem import RLMemory
import tensorflow as tf

class RLAgent:
	def __init__(self,
		env,
		state_shape,
		action_space_shape,

		memory_capacity = None,
		model = None,
		human_input = False,
		do_show_evaluations = False,

		minibatch_size = None,
		minibatch_adaptive_rate = 0.95,
		gamma = 0.98,
		model_fit_batch_size = 2048,
		epsilon_min = 0.05,
		epsilon_start = 0.98,
		epsilon_decay = 0.98,
		epsilon_restart_interval = None,
		heuristic_chance = 0.4,
		heuristic_decay = 0.99,

		random_actions = False,
		random_actions_are_valid = False,
		decided_actions_must_be_valid = False,
		experience_replay = True,
		record_transitions = True,
		temporal_learning = False,
		episodes_per_experience_replay = 1,

		do_model_saving = True,

		double_q_tau = None,
		double_q_update_interval = None,

		munchausen = False,
		munchausen_tau = 0.5,
		munchausen_alpha = 0.5,

	):
		self.env = env
		self.memory = self.env.memory if memory_capacity is None else RLMemory(capacity = memory_capacity)
		self.id = len(self.env.agents)
		self.model = model
		self.state_shape = state_shape
		self.model_input_shape = (-1, *self.state_shape)
		self.action_space_shape = action_space_shape
		self.random_actions = random_actions
		self.random_actions_are_valid = random_actions_are_valid
		self.decided_actions_must_be_valid = decided_actions_must_be_valid
		self.experience_replay = experience_replay
		self.record_transitions = record_transitions
		self.temporal_learning = temporal_learning
		self.episodes_per_experience_replay = episodes_per_experience_replay
		self.minibatch_size = minibatch_size
		self.minibatch_adaptive_rate = minibatch_adaptive_rate
		self.gamma = gamma
		self.model_fit_batch_size = model_fit_batch_size
		self.epsilon_start = epsilon_start
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay
		self.epsilon = self.epsilon_start
		self.do_model_saving = do_model_saving
		self.resets = 0
		self.total_accumulated_reward = 0
		self.current_accumulated_reward = 0
		self.total_transitions_saved = 0
		self.current_transitions_saved = 0
		self.accumulated_reward_history = []
		self.recent_mean_accumulated_reward_history = []

		self.min_q_values = [] # q value predictions from the network from each step, cleared after each episode
		self.max_q_values = []
		self.mean_q_values = []
		self.mean_min_q_value_history = [] # mean of self.min_q_values, computed after each episode (useful for verifying that the model is learning when it doesn't look like it is)
		self.mean_max_q_value_history = []
		self.mean_mean_q_value_history = []
		self.recent_mean_mean_min_q_value_history = [] # mean of latest 100 values of self.mean_min_q_value_history
		self.recent_mean_mean_max_q_value_history = []
		self.recent_mean_mean_mean_q_value_history = []

		self.heuristic_chance_start = heuristic_chance
		self.heuristic_chance = heuristic_chance
		self.heuristic_decay = heuristic_decay
		self.epsilon_restart_interval = epsilon_restart_interval if epsilon_restart_interval is not None else np.inf
		self.target_model = None
		self.double_q_update_interval = None
		self.double_q_tau = 1
		self.set_double_q(double_q_tau, double_q_update_interval)
		self.munchausen = munchausen
		self.munchausen_tau = munchausen_tau
		self.munchausen_alpha = munchausen_alpha
		self.human_input = human_input
		self.do_show_evaluations = do_show_evaluations

	def reset(self):
		self.total_accumulated_reward += self.current_accumulated_reward
		self.total_transitions_saved += self.current_transitions_saved
		if self.resets > 0:
			self.accumulated_reward_history.append(self.current_accumulated_reward)
			self.recent_mean_accumulated_reward = sum(self.accumulated_reward_history[max(0,len(self.accumulated_reward_history)-100):]) / min(100, len(self.accumulated_reward_history))
			self.recent_mean_accumulated_reward_history.append(self.recent_mean_accumulated_reward)
			midpoint = len(self.accumulated_reward_history) // 2
			if midpoint > 0:
				self.accumulated_reward_trend, _ = np.polyfit(np.arange(len(self.accumulated_reward_history)), self.accumulated_reward_history, 1)
			else:
				self.accumulated_reward_trend = 0
		if self.env.current_episode % self.epsilon_restart_interval == 0:
			self.epsilon = self.epsilon_start
		self.current_accumulated_reward = 0
		self.current_transitions_saved = 0
		self.mean_accumulated_reward = self.total_accumulated_reward / max(1, self.resets)
		self.dead = False
		self.memory.on_reset(clear_recently_added = self.env.current_episode % self.episodes_per_experience_replay == 0)
		self.steps_since_last_reset = 0
		self.current_observation = np.zeros(self.state_shape)
		self.last_observation = self.current_observation
		self.last_last_observation = self.last_observation # I don't like this variable name, but this is what it is
		self.transition_to_save = None
		self.skip_timer = 0
		if len(self.min_q_values) > 0:
			self.mean_min_q_value_history.append(np.mean(self.min_q_values))
			self.mean_max_q_value_history.append(np.mean(self.max_q_values))
			self.mean_mean_q_value_history.append(np.mean(self.mean_q_values))
			self.recent_mean_mean_min_q_value_history.append(np.mean(self.mean_min_q_value_history[max(len(self.mean_min_q_value_history)-100,0):]))
			self.recent_mean_mean_max_q_value_history.append(np.mean(self.mean_max_q_value_history[max(len(self.mean_max_q_value_history)-100,0):]))
			self.recent_mean_mean_mean_q_value_history.append(np.mean(self.mean_mean_q_value_history[max(len(self.mean_mean_q_value_history)-100,0):]))
			self.min_q_values.clear()
			self.max_q_values.clear()
			self.mean_q_values.clear()
		self.on_reset()
		self.resets += 1

	def on_reset(self):
		pass

	def get_state(self):
		raise Exception('RLAgent.get_state() is not overridden.')

	def get_reward(self):
		raise Exception('RLAgent.get_reward() is not overridden.')

	def get_random_actions(self):
		'''
		Return a randomly selected tuple of actions which represents how to take the next step in the environment.
		'''
		if self.random_actions_are_valid:
			va = self.get_current_valid_actions()
			return va[np.random.randint(0, len(va))]
		return tuple(np.random.randint(0, self.action_space_shape))

	def get_heuristic_actions(self):
		# default is to take random actions, but override this if you want to make models explore the actions that this function generates
		# if the heuristic actions are actually good, then the models will find high reward by taking these actions and learn to generalize it for other states on their own
		# otherwise, the models will learn that it isn't good (or good enough), and they will predict better moves after more training
		# implementing a heuristic can speed up training by showing agents what they should try, but they will still explore randomly if they don't like these heuristic actions
		return self.get_random_actions()

	def get_actions_prediction(self):
		'''
		Use the agent's model to predict the next action(s) to take in the environment.
		If the agent has no model, then this function returns a tuple of random actions, sampled with uniform distribution.
		'''
		if self.model is None:
			return self.get_random_actions()
		else:
			# ssym = self.get_state_symmetries(self.current_observation)
			# self.observed_symmetry_index = np.random.randint(len(ssym))
			# co = ssym[self.observed_symmetry_index]
			# TODO
			# Make the model observe a random symmetry instead of the given one.
			# The action will need to be unmapped somehow (generating a map of actions and an inverse map of actions for each possible symmetry is one way to solve this problem)
			self.state_evaluation = self.model.predict(np.reshape(self.current_observation, newshape = self.model_input_shape))
			self.min_q_values.append(np.min(self.state_evaluation))
			self.max_q_values.append(np.max(self.state_evaluation))
			self.mean_q_values.append(np.mean(self.state_evaluation))
			if self.decided_actions_must_be_valid:
				va = self.get_current_valid_actions()
				ev = np.zeros_like(self.state_evaluation)
				ev[(0, *zip(*va))] = self.state_evaluation[(0, *zip(*va))]
				ev = np.where(ev == 0, -np.inf, ev)
				# TODO: unmap action based on chosen symmetry
				return np.unravel_index(np.argmax(ev), shape = self.action_space_shape)
			return np.unravel_index(np.argmax(self.state_evaluation), shape = self.action_space_shape)

	def get_actions_from_human_input(self):
		raise NotImplementedError('get_actions_from_human_input() needs to be overridden to use RLAgent.human_input = True')

	def decide_actions(self):
		'''
		This function returns the tuple of actions (can be a 1-tuple) that the agent will take next in the environment.
		This function can be overridden to allow for, for example, user input, but the agent uses its neural network by default to decide on its actions (by returning the return value of `get_actions_prediction()`).
		'''
		if self.human_input:
			return self.get_actions_from_human_input()
		return self.get_actions_prediction()

	def get_current_valid_actions(self):
		'''
		Return a list of tuples which represent all of the possible "valid" actions that can be taken in the current state of the environment.
		'''
		raise NotImplementedError('RLAgent.get_current_valid_actions() must be overridden.')

	def pre_step(self):
		'''
		Called before the agent takes a step in the environment.
		'''
		if self.skip_timer > 0:
			return
		self.current_observation = self.get_state().reshape(self.state_shape)

		if self.steps_since_last_reset > 0:
			reward = self.get_reward()
			terminal = self.env.is_in_terminal_state or self.dead
			self.current_accumulated_reward += reward
			self.transition_to_save = [self.last_observation, self.decided_actions, reward, self.current_observation, 0 if terminal else 1]

		if (self.random_actions_are_forced() or np.random.random() < self.epsilon) and not self.env.test_mode:
			if self.model is not None and np.random.random() < self.heuristic_chance:
				self.decided_actions = self.get_heuristic_actions()
			else:
				self.decided_actions = self.get_random_actions()
		else:
			self.decided_actions = self.decide_actions()

		if self.do_show_evaluations and self.model is not None:
			self.show_evaluations()

		self.on_pre_step()

	def show_evaluations(self):
		raise NotImplementedError('Implement this method to be able to set RLAgent.do_show_evaluations = True')

	def on_pre_step(self):
		'''
		Called before the agent takes a step in the environment.
		Overwrite this method to let the agent decide on what to do in the environment.
		First (pre_step), every agent "thinks."
		Then (step), the environment executes the actions.
		Finally (post_step), the environment returns the rewards for each agent.
		'''
		pass

	def step(self):
		'''
		Called when the agent takes a step in the environment.
		'''
		if self.skip_timer > 0:
			return
		self.do_actions(actions = self.decided_actions)
		self.on_step()

	def on_step(self):
		'''
		Called when the agent takes a step in the environment.
		Overwrite this method to make the environment execute the agent's action.
		First (pre_step), every agent "thinks."
		Then (step), the environment executes the actions.
		Finally (post_step), the environment returns the rewards for each agent.
		'''
		pass

	def post_step(self):
		'''
		Called after the agent takes a step in the environment.
		'''
		if self.skip_timer > 0:
			self.skip_timer -= 1
		else:
			self.steps_since_last_reset += 1
			self.on_post_step()
			self.last_last_observation = self.last_observation
			self.last_observation = self.current_observation
			if self.transition_to_save is not None and not self.dead:
				self.record_transition(self.transition_to_save)
			if self.temporal_learning and (not self.env.autofilter_experience_replay or self != self.env.best_performers_history[-1]):
				self.do_experience_replay()

	def on_post_step(self):
		'''
		Called after the agent takes a step in the environment.
		Overwrite this method to make the environment compute (return) the agent's reward.
		First (pre_step), every agent "thinks."
		Then (step), the environment executes the actions.
		Finally (post_step), the environment returns the rewards for each agent.
		'''
		pass

	def do_experience_replay(self, full_replay = False, epochs = None):
		'''
		Fit on a random sample of previously collected data.
		'''
		#rm = self.env.render_mode
		#self.env.set_render_mode(None)
		if self.experience_replay and self.model is not None and self.env.random_episodes_simulated >= self.env.initial_random_episodes and self.env.current_episode % self.episodes_per_experience_replay == 0:
			sample = None
			if full_replay:
				print('Fitting on random transition history for agent {}...'.format(self.id))
				sample = self.memory.sample_all()
			else:
				bs = self.minibatch_size if self.minibatch_size is not None else int(self.minibatch_adaptive_rate * self.memory.recently_added + 0.5)
				if bs <= 0:
					#self.env.set_render_mode(rm)
					return
				sample = self.memory.sample(size = bs)
			if len(sample) == 0:
				#self.env.set_render_mode(rm)
				return
			self.pre_experience_replay()
			print('Experience replay for Agent {} on minibatch of size {}...'.format(self.id, len(sample)))
			old_states = []
			actions = []
			rewards = []
			new_states = []
			terminal = []
			for s in sample:
				old_states.append(s[0])
				actions.append(s[1])
				rewards.append(s[2])
				new_states.append(s[3])
				terminal.append(s[4])
			old_states = np.array(old_states).reshape(self.model_input_shape)
			rewards = np.array(rewards)
			new_states = np.array(new_states).reshape(self.model_input_shape)
			terminal = np.array(terminal)

			pred_old_states = self.apply_invalid_action_prediction(self.model.predict(old_states))

			if self.use_double_q:
				pred_new_states_eval = self.apply_invalid_action_prediction(self.model.predict(new_states))
				pred_new_states_target = self.apply_invalid_action_prediction(self.target_model.predict(new_states))
				action_indices = np.unravel_index(np.argmax(pred_new_states_eval, axis = 1), shape = pred_new_states_eval.shape[1:])
				# if self.munchausen:
				# 	q_k_targets = np.copy(pred_new_states_target)
				# 	q_k_targets -= np.max(q_k_targets, axis = 1)[0]
				# 	log_sum_exp = np.log(np.sum(np.exp(q_k_targets / self.munchausen_tau), axis = 1))
				# 	log_sum_exp = log_sum_exp.reshape(*log_sum_exp.shape, 1)
				# 	tau_log_pi = q_k_targets - self.munchausen_tau * log_sum_exp
				# 	ex = np.exp(pred_new_states_target / self.munchausen_tau)
				# 	s = np.sum(ex, axis = 1)
				# 	pi_target = ex / np.reshape(s, (*s.shape, 1))
				# 	terminal = terminal.reshape(-1, 1)
				# 	updated_values = rewards + self.gamma * np.sum(pi_target * (pred_new_states_target - tau_log_pi) * terminal, axis = 1)
				# else:
				# TODO: use slow indexing to calculate updated_values
				updated_values = rewards + self.gamma * terminal * pred_new_states_target[(np.arange(len(sample)), *action_indices)]
			else:
				pred_new_states = self.apply_invalid_action_prediction(self.model.predict(new_states))
				updated_values = rewards + self.gamma * terminal * np.max(pred_new_states, axis = 1)

			if self.munchausen:
				q_k_targets = self.apply_invalid_action_prediction(self.target_model.predict(old_states)) if self.use_double_q else pred_old_states
				q_k_targets -= np.max(q_k_targets, axis = 1)[0]
				log_sum_exp = np.log(np.sum(np.exp(q_k_targets / self.munchausen_tau), axis = 1))
				updated_values += self.munchausen_alpha * np.clip((q_k_targets - self.munchausen_tau * log_sum_exp.reshape(*log_sum_exp.shape, 1))[(np.arange(len(sample)), *tuple(zip(*actions)))], -1, 0)

			if len(sample) < 10000:
				pred_old_states[(np.arange(len(sample)), *tuple(zip(*actions)))] = updated_values
			else:
				l = len(sample)
				for i in range(0, l, 4096):
					action_indices = tuple(zip(*actions[i:i+4096]))
					pred_old_states[(np.arange(i, min(l,i+4096), dtype=np.int8), *action_indices)] = updated_values[i:i+4096]

			self.model.fit(old_states, pred_old_states, verbose = 1 if full_replay else 0, epochs = epochs if epochs is not None else 2 if full_replay else 1, batch_size = self.model_fit_batch_size)

			if self.use_double_q:
				if self.double_q_tau is not None:
					ew = self.model.get_weights()
					tw = self.target_model.get_weights()
					self.target_model.set_weights([
						ew[i] * self.double_q_tau + tw[i] * (1 - self.double_q_tau)
						for i in range(len(ew))
					])
				elif self.double_q_update_interval is not None:
					if self.env.current_step % self.double_q_update_interval == 0:
						self.target_model.set_weights(self.model.get_weights())

			self.post_experience_replay()
		#self.env.set_render_mode(rm)

	def apply_invalid_action_prediction(self, pred):
		return pred

	def pre_experience_replay(self):
		'''
		Called before the agent does experience replay.
		'''
		self.on_pre_experience_replay()

	def on_pre_experience_replay(self):
		'''
		Called before the agent does experience replay.
		Overwrite this method if necessary.
		'''
		pass

	def post_experience_replay(self):
		'''
		Called after the agent does experience replay.
		'''
		self.on_post_experience_replay()

	def on_post_experience_replay(self):
		'''
		Called after the agent does experience replay.
		Overwrite this method if necessary.
		'''
		pass

	def do_actions(self, actions):
		pass

	def _episode_ended(self):
		if not self.dead:
			reward = self.get_reward()
			self.current_accumulated_reward += reward
			self.transition_to_save = None
			self.record_transition([self.last_last_observation, self.decided_actions, reward, self.current_observation, 0])
		if not self.temporal_learning and (not self.env.autofilter_experience_replay or self != self.env.best_performers_history[-1]):
			self.do_experience_replay()
		if not self.random_actions_are_forced():
			if not self.env.test_mode:
				self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
			self.heuristic_chance *= self.heuristic_decay
			if self.heuristic_chance < 0.001:
				self.heuristic_chance = 0
		self.episode_ended()

	def episode_ended(self):
		pass

	def random_actions_are_forced(self):
		return self.random_actions or self.model is None or self.env.random_episodes_simulated < self.env.initial_random_episodes

	def set_double_q(self, tau = None, update_interval = None):
		'''
		Set whether the agent should use the deep double Q algorithm in training.

		Tau is the rate at which the target model parameters approach the evaluation model parameters.
		It should be a number between 0 and 1, inclusive.
		If tau = 1, then it is identical to standard (single model) deep Q learning.
		If tau = 0, then the agent will almost certainly learn nothing.
		In general, the more stochastic transitions there are in an environment, the lower tau should be, but the longer it will take to train the agent.
		'''
		if update_interval is not None:
			self.use_double_q = True
			self.double_q_update_interval = update_interval
		elif tau is not None:
			self.use_double_q = True
			self.double_q_tau = tau
			if self.use_double_q and self.model is not None:
				self.target_model = clone_model(self.model)
				self.target_model.set_weights(self.model.get_weights())
		else:
			self.use_double_q = False

	def set_model(self, new_model):
		self.model = new_model
		if self.model is not None and self.use_double_q and self.target_model is None:
			if type(self.model) is Sequential:
				self.target_model = clone_model(self.model)
			else:
				self.target_model = tf.keras.models.clone_model(self.model)
			self.target_model.set_weights(self.model.get_weights())

	def kill(self):
		if self.dead:
			return
		self.dead = True
		self.on_kill()
		reward = self.get_reward()
		self.current_accumulated_reward += reward
		self.transition_to_save = None
		self.record_transition([self.last_last_observation, self.decided_actions, reward, self.current_observation, 0])

	def on_kill(self):
		pass

	def skip(self, steps = 1):
		self.skip_timer += steps

	def skip_other_agents(self, steps = 1):
		for agent in self.env.agents:
			if agent != self:
				agent.skip(steps)

	def load_model(self, fpath):
		model = None
		try:
			model = load_keras_model(filepath = fpath)
		except ImportError:
			raise Exception('The model at {} could not be loaded. Is h5py installed?'.format(fpath))
		else:
			self.set_model(model)

	def save_model(self):
		if self.env.test_mode or self.env.random_episodes_simulated < self.env.initial_random_episodes:
			return
		self.env.save_sim_history()
		if self.model is None or not self.do_model_saving:
			return
		if self.model in self.env.models_currently_saved:
			return
		self.env.models_currently_saved.append(self.model)
		model_fname = self.get_model_save_file_name()
		try:
			self.model.save('{}/{}.h5'.format(self.env.model_save_location, model_fname), overwrite = True)
		except OSError:
			pass
		np.savez('{}/{}_history.npz'.format(self.env.model_save_location, model_fname),
			recent_reward_history = self.recent_mean_accumulated_reward_history,
			recent_min_q = self.recent_mean_mean_min_q_value_history,
			recent_max_q = self.recent_mean_mean_max_q_value_history,
			recent_mean_q = self.recent_mean_mean_mean_q_value_history,
			agent_id = self.id
		)

	def get_model_save_file_name(self):
		return 'agent_{}'.format(self.id)

	def record_transition(self, transition):
		if self.record_transitions:
			for t in self.get_transition_symmetries(transition):
				self.current_transitions_saved += 1
				self.env.current_transitions_saved += 1
				self.memory.add_transition(t)

	def get_transition_symmetries(self, transition):
		'''
		Return a list of transitions that are symmetrical to the one provided, including the identity symmetry.
		'''
		s = self.get_state_symmetries(transition[0])
		a = self.get_actions_symmetries(transition[1])
		s2 = self.get_state_symmetries(transition[3])
		return [
			[
				s[i],
				a[i],
				transition[2],
				s2[i],
				transition[4]
			]
			for i in range(len(s))
		]

	def get_state_symmetries(self, state):
		'''
		Return a list of state symmetries, including the provided state itself (identity symmetry).
		The returned list should iterate through the symmetries in the same order as the list returned from get_actions_symmetries().

		This is particularly useful for training models in environments where states or actions are invariant under rotation and/or reflection.
		For example, states and actions in the game of Go (piece locations on the grid) are invariant under rotation and reflection.
		In that case, states and actions have a total of eight symmetries (seven extra per transition) that can be added to the transition memory for training,
		effectively octupling the rate at which training data can be collected per episode.

		If multiple symmetries exist for states and actions in this environment, then this method should be overridden to account for that.
		NumPy's rot90() and flip() functions are recommended here.
		Agents can be trained regardless of whether symmetries are considered, but the agents will generalize much better (thus, train faster) when symmetries are considered.
		'''
		return [state]

	def get_actions_symmetries(self, actions):
		return [actions]

