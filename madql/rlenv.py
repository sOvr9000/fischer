
'''
The following methods MUST be overridden:
- RLEnvironment.new_agent()

The following methods can be overridden for better control:
- RLEnvironment.on_pre_reset()
- RLEnvironment.on_reset()						| For most implementations, you will want to override this.
- RLEnvironment.on_step()
- RLEnvironment.is_state_terminal()				| For most implementations, you will want to override this.
- RLEnvironment.episode_ended()
- RLEnvironment.on_random_episodes_complete()
- RLEnvironment.on_print_simulation_info()
- RLEnvironment.render_as_image()				| Required if RLEnvironment.set_render_mode('image')
- RLEnvironment.render_as_text()				| Required if RLEnvironment.set_render_mode('text')
'''



import os
from datetime import datetime
import numpy as np
from keras.models import clone_model
from madql.rlagent import RLAgent
from madql.rlmem import RLMemory



def sort_by_value(l, v):
	return [e for _,_,e in sorted([(v(l[i]), i, l[i]) for i in range(len(l))])]

def millis():
	return int((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds() * 1000)



class RLEnvironment:
	def __init__(self,
		max_agents = 1,
		max_steps = np.inf,

		memory_capacity = None,
		parallel_agent_stepping = True,

		initial_random_episodes = 0,
		train_after_initial_random_episodes = True, # TODO: implement toggleability (this is always True, even when you set this to False)
		model_save_interval = 100,
		model_save_location = None,
		render_mode = None,

		autofilter_experience_replay = False, # if True, do experience replay only for the agent which is currently performing the worst out of the last 100 episodes

		agent_cls = RLAgent,
	):
		self.memory = RLMemory(capacity = memory_capacity if memory_capacity is not None else 100000)
		self.agents = []
		self.inactive_agents = []
		self.agents_by_id = {}
		self.max_agents = max_agents
		self.max_steps = max(1, max_steps)
		for i in range(self.max_agents):
			self.add_agent(self.new_agent())
		if type(parallel_agent_stepping) is bool:
			self.parallel_agent_stepping = [[n for n in range(self.max_agents)]] if parallel_agent_stepping else [[n] for n in range(self.max_agents)]
		elif type(parallel_agent_stepping) is list:
			self.parallel_agent_stepping = list(parallel_agent_stepping)
		self.num_parallel_steps = len(self.parallel_agent_stepping)
		self.agent_cls = agent_cls
		self.random_episodes_simulated = -1
		self.current_step = 0
		self.current_episode = -1
		self.total_transitions_saved = 0
		self.total_steps_simulated = 0
		self.current_transitions_saved = 0
		self.episode_length_history = []
		self.recent_mean_episode_length_history = []
		self.episode_length_trend = 0
		self.transitions_saved_history = []
		self.recent_mean_transitions_saved_history = []
		self.episode_ms_duration_history = []
		self.mean_episode_ms_duration_history = []
		self.recent_mean_episode_ms_duration_history = []
		self.transitions_saved_trend = 0
		self.initial_random_episodes = initial_random_episodes
		self.set_render_mode(render_mode)
		self.render_time_scale = 1
		self.model_save_interval = model_save_interval
		self.recent_mean_transitions_saved = 0
		self.test_mode = False
		self.paused = False
		self.autofilter_experience_replay = autofilter_experience_replay
		self.current_best_performer = None
		self.best_performers_history = []
		self.models_currently_saved = []
		if any([agent.do_model_saving for agent in self.agents]):
			if model_save_location is None:
				new_dir = './models_{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
				try:
					os.mkdir(new_dir)
				except FileExistsError:
					pass
				except:
					new_dir = './'
				self.model_save_location = new_dir
			else:
				try:
					os.mkdir(model_save_location)
				except:
					pass
				self.model_save_location = model_save_location

	def reset(self):
		self.on_pre_reset()
		self.memory.on_reset(clear_recently_added = self.current_episode % max(agent.episodes_per_experience_replay for agent in self.agents + self.inactive_agents) == 0)
		self.episode_progress = 0
		if self.random_episodes_simulated < self.initial_random_episodes:
			self.random_episodes_simulated += 1
			if self.random_episodes_simulated >= self.initial_random_episodes:
				self.random_episodes_complete()
		self.current_episode += 1
		if len(self.inactive_agents) > 0:
			self.agents += self.inactive_agents
			self.agents = sort_by_value(self.agents, lambda agent: agent.id)
			self.inactive_agents.clear()
		for agent in self.agents:
			agent.reset()
		s = self.current_step
		self.current_step = -1
		self.has_saved_sim_history = False
		self.is_in_terminal_state = False
		self.total_transitions_saved += self.current_transitions_saved
		self.mean_episode_length = self.total_steps_simulated / max(1, self.current_episode)
		self.mean_transitions_saved = self.total_transitions_saved / max(1, self.current_episode)
		if self.current_episode > 0:
			self.episode_length_history.append(s)
			self.recent_mean_episode_length = np.mean(self.episode_length_history[max(len(self.episode_length_history)-100,0):])
			self.recent_mean_episode_length_history.append(self.recent_mean_episode_length)
			if len(self.episode_length_history) > 1:
				self.episode_length_trend, _ = np.polyfit(np.arange(len(self.episode_length_history)), self.episode_length_history, 1)
				self.recent_episode_length_trend, _ = np.polyfit(np.arange(min(100,len(self.episode_length_history))), self.episode_length_history[max(0,len(self.episode_length_history)-100):], 1)
			else:
				self.episode_length_trend = 0
				self.recent_episode_length_trend = 0
			self.transitions_saved_history.append(self.current_transitions_saved)
			self.recent_mean_transitions_saved = np.mean(self.transitions_saved_history[max(len(self.transitions_saved_history)-100,0):])
			self.recent_mean_transitions_saved_history.append(self.recent_mean_transitions_saved)
			midpoint = len(self.transitions_saved_history) // 2
			if midpoint > 0:
				self.transitions_saved_trend, _ = np.polyfit(np.arange(len(self.transitions_saved_history)), self.transitions_saved_history, 1)
				self.recent_transitions_saved_trend, _ = np.polyfit(np.arange(min(100,len(self.transitions_saved_history))), self.transitions_saved_history[max(0,len(self.transitions_saved_history)-100):], 1)
			else:
				self.transitions_saved_trend = 0
				self.recent_transitions_saved_trend = 0
			self.episode_ms_duration_history.append(millis() - self.episode_start)
			self.mean_episode_ms_duration_history.append(np.mean(self.episode_ms_duration_history))
			self.recent_mean_episode_ms_duration_history.append(np.mean(self.episode_ms_duration_history[max(len(self.episode_ms_duration_history)-100,0):]))
			if len(self.recent_mean_episode_ms_duration_history) > 1:
				self.episode_ms_duration_trend, _ = np.polyfit(np.arange(len(self.episode_ms_duration_history)), self.episode_ms_duration_history, 1)
				self.recent_episode_ms_duration_trend, _ = np.polyfit(np.arange(min(100,len(self.episode_ms_duration_history))), self.episode_ms_duration_history[max(0,len(self.episode_ms_duration_history)-100):], 1)
			else:
				self.episode_ms_duration_trend = 0
				self.recent_episode_ms_duration_trend = 0
		self.current_transitions_saved = 0
		self.parallel_agent_stepping_index = 0
		if self.current_episode != 0 and self.current_episode % self.model_save_interval == 0 and self.random_episodes_simulated >= self.initial_random_episodes:
			self.save_agent_models()
		self.episode_start = millis()
		self.on_reset()
		self.render()

	def add_agent(self, agent):
		'''
		Add an agent to the environment.
		'''
		self.agents.append(agent)
		self.agents_by_id[agent.id] = agent

	def get_agent(self, agent_id):
		'''
		Get an agent by its id.
		'''
		return self.agents_by_id[agent_id] if agent_id in self.agents_by_id else None

	def new_agent(self):
		'''
		Create a new instance of an agent.
		'''
		return self.agent_cls(self)

	def toggle_pause(self):
		self.paused = not self.paused

	def on_pre_reset(self):
		pass

	def on_reset(self):
		pass

	def step(self):
		'''
		Simulate the next step of the environment.
		'''
		if True:#not self.paused:
			if self.parallel_agent_stepping_index == 0:
				self.current_step += 1
				self.total_steps_simulated += 1
			self.episode_progress = self.current_step / self.max_steps
			parallel_stepping = self.parallel_agent_stepping[self.parallel_agent_stepping_index]
			for agent_id in parallel_stepping:
				agent = self.get_agent(agent_id)
				if agent is not None and not agent.dead:
					agent.pre_step()
			for agent_id in parallel_stepping:
				agent = self.get_agent(agent_id)
				if agent is not None and not agent.dead:
					agent.step()
			for i in range(len(self.agents)-1,-1,-1):
				if self.agents[i].dead:
					self.inactive_agents.append(self.agents[i])
					del self.agents[i]
		self.on_step()
		self.is_in_terminal_state = self._is_state_terminal()
		if True:#not self.paused:
			for agent_id in parallel_stepping:
				agent = self.get_agent(agent_id)
				if agent is not None and not agent.dead:
					agent.post_step()
			for i in range(len(self.agents)-1,-1,-1):
				if self.agents[i].dead:
					self.inactive_agents.append(self.agents[i])
					del self.agents[i]
		if self.current_step % self.render_time_scale == 0:
			self.render()
		if True:#not self.paused:
			self.parallel_agent_stepping_index = (self.parallel_agent_stepping_index + 1) % self.num_parallel_steps
			if self.is_in_terminal_state:
				self._episode_ended()

	def on_step(self):
		pass

	def _is_state_terminal(self):
		return self.current_step >= self.max_steps - 1 or len(self.agents) == 0 or self.is_state_terminal()

	def is_state_terminal(self):
		'''
		Is the current state of the environment terminal?
		
		If True, then the current episode will terminate and the next episode will commence.

		Max steps per episode overrides this return value, but if this returns True before the end of the current episode, then it will end early.
		'''
		return False

	def get_nonrandom_agents(self):
		return [agent for agent in self.agents + self.inactive_agents if not agent.random_actions_are_forced()]

	def determine_best_performer(self):
		'''
		Return the agent which performed the best during the episode.  This method is called at the end of each episode.
		
		By default, it returns the agent (among the ones that do not forcibly execute random actions) which accumulated the most reward over the episode.  Safe to override.

		RLEnvironment.best_performer is updated before RLEnvironment.reset() gets called to start the next episode.
		'''
		nra = self.get_nonrandom_agents()
		if len(nra) == 0:
			return None
		l = [(nra[i].accumulated_reward if hasattr(nra[i], 'accumulated_reward') else 0, i, nra[i]) for i in range(len(nra))]
		return max(l)[2]

	def _episode_ended(self):
		self.best_performers_history.append(self.determine_best_performer())
		if self.current_episode >= 100 and self.current_episode % 100 == 0:
			last100 = self.best_performers_history[-100:]
			#self.current_best_performer =  # TODO Count number of times that each agent was best performer, then select the one that is the best overall
		for agent in self.agents + self.inactive_agents:
			agent._episode_ended()
		if self.test_mode:
			if self.do_check_winning:
				self._check_winning()
		self.episode_ended()
		self.reset()

	def episode_ended(self):
		'''
		Called at the end of each episode. The default behavior is to clear the output (via the command 'cls') and print simulation info.

		Safe to override.
		'''
		os.system('cls')
		self.print_simulation_info()

	def random_episodes_complete(self):
		trained_models = []
		for agent in self.agents + self.inactive_agents:
			if agent.model is not None and agent.model not in trained_models:
				trained_models.append(agent.model)
				agent.do_experience_replay(full_replay = True)
		self.on_random_episodes_complete()

	def on_random_episodes_complete(self):
		pass

	def _check_winning(self):
		winner = self.check_winning()
		if type(winner) is int:
			if winner >= self.max_agents or winner < 0:
				raise ValueError('RLEnvironment.check_winning() returned an integer out of bounds')
			self.test_models_result['winners'][winner] += 1
		elif type(winner) is tuple:
			ties = self.test_models_result['ties']
			for w in winner:
				if type(w) is not int or w < 0 or w >= self.max_agents:
					raise TypeError('One of the values in the tuple returned by RLEnvironment.check_winning() is not an integer within bounds')
			winner = tuple(sorted(winner))
			if winner in ties:
				ties[winner] += 1
			else:
				ties[winner] = 1
		else:
			raise TypeError('RLEnvironment.check_winning() returned invalid type: ' + str(type(winner)))

	def check_winning(self):
		raise NotImplementedError('To use RLenvironment.test_models(check_winning = True), you must implement RLEnvironment.check_winning(), and it must return the id of the agent that won, or a tuple of id\'s if tied.')

	def print_simulation_info(self):
		self.all_agents = self.agents + self.inactive_agents
		print('-' * 128)
		print('Episode: {}'.format(self.current_episode))
		if self.initial_random_episodes > 0:
			print('Random episodes simulated: {} / {}'.format(self.random_episodes_simulated, self.initial_random_episodes))
		else:
			print('.' * 4)
		if self.current_episode > 0:
			print('[STATS] Previous / Average / Trend / Recent Average / Recent Trend ...')
			print('[STEPS PER EPISODE]\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(self.episode_length_history[-1], self.mean_episode_length, self.episode_length_trend, self.recent_mean_episode_length, self.recent_episode_length_trend))
			if not self.test_mode:
				print('[TRANSITIONS PER EPISODE]\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(self.transitions_saved_history[-1], self.mean_transitions_saved, self.transitions_saved_trend, self.recent_mean_transitions_saved, self.recent_transitions_saved_trend))
			print('[EPISODE DURATION -- MILLISECONDS]\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(self.episode_ms_duration_history[-1], self.mean_episode_ms_duration_history[-1], self.episode_ms_duration_trend, self.recent_mean_episode_ms_duration_history[-1], self.recent_episode_ms_duration_trend))
			print('[STEP DURATION -- MILLISECONDS]:\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format('--', self.mean_episode_ms_duration_history[-1] / self.mean_episode_length, '--', self.recent_mean_episode_ms_duration_history[-1] / self.recent_mean_episode_length, '--'))
		else:
			print('\n'.join(['.' * 4] * 15))
		if not self.test_mode:
			print('Transitions stored in shared memory: {} / {}'.format(len(self.memory.transitions), self.memory.capacity))
			own_memory_agents = [agent for agent in self.all_agents if agent.memory != self.memory]
			if len(own_memory_agents) > 0:
				print('Transitions stored in personal memory:')
				for agent in own_memory_agents:
					print('\tAgent {}: {} / {}'.format(agent.id, len(agent.memory.transitions), agent.memory.capacity))
			else:
				print('\n'.join(['.' * 4] * (self.max_agents + 1)))
		if self.current_episode > 0:
			print('Average accumulated reward per reset:') # per reset because agents can reset mid-episode, without the episode resetting also
			for agent in reversed(sort_by_value(self.all_agents, lambda agent: agent.mean_accumulated_reward)):
				print('\tAgent {}: {}'.format(agent.id, agent.mean_accumulated_reward))
			if not self.test_mode:
				print('Recent average accumulated reward per reset:')
				for agent in reversed(sort_by_value(self.all_agents, lambda agent: agent.recent_mean_accumulated_reward)):
					print('\tAgent {}: {}'.format(agent.id, agent.recent_mean_accumulated_reward))
			print('Accumulated reward trend:')
			for agent in reversed(sort_by_value(self.all_agents, lambda agent: agent.accumulated_reward_trend)):
				print('\tAgent {}: {}'.format(agent.id, agent.accumulated_reward_trend))
			if not self.test_mode:
				print('Mean episode Q values, MIN / MEAN / MAX (previous, average, trend, recent average, recent trend):')
				# These are helpful to see how the model is learning to evaluate the environment throughout an episode over time (as training progresses).
				# For example, if the recent trend of MAX episode q value is higher than the recent trend of MIN episode q value, then that agent is learning how to better distinguish good actions from bad ones.
				# For another example, if the recent average MEAN episode q value is closer to the recent average MIN episode q value than it is to the recent average MAX episode q value, then the agent currently understands the value of good actions more than the value of bad actions.
				# ... whereas if r.a. MEAN is closer to r.a. MAX (instead of r.a. MIN), then the agent currently understands the value of bad actions more than the value of good actions.
				# In the former case, where mean is closer to min, this generally indicates that the model knows what the few good actions are if there are any at all in a given state of the environment.  For example, in chess, most legal moves are bad moves, but a few moves are actually good moves, which is why the model would predict with a mean that is near the min and with a max that is much higher than the mean.
				# SO... while running the training loop and interpreting this section of the simulation info, you want to see (eventually) r.a. MEAN closer to r.a. MIN than r.a. MAX, and you want to see (overall) trend MIN to be lower than trend MAX.
				for agent in sort_by_value(self.all_agents, lambda agent: agent.id):
					if len(agent.mean_min_q_value_history) >= 1:
						print(f'\tAgent {agent.id}:')
						print('\t\t{:.3f} / {:.3f} / {:.3f}'.format(
							agent.mean_min_q_value_history[-1],
							agent.mean_mean_q_value_history[-1],
							agent.mean_max_q_value_history[-1]
						))
						print('\t\t{:.3f} / {:.3f} / {:.3f}'.format(
							np.mean(agent.mean_min_q_value_history),
							np.mean(agent.mean_mean_q_value_history),
							np.mean(agent.mean_max_q_value_history)
						))
						if len(agent.mean_min_q_value_history) >= 2:
							print('\t\t{:.3f} / {:.3f} / {:.3f}'.format(
								np.polyfit(np.arange(len(agent.mean_min_q_value_history)), agent.mean_min_q_value_history, 1)[0],
								np.polyfit(np.arange(len(agent.mean_mean_q_value_history)), agent.mean_mean_q_value_history, 1)[0],
								np.polyfit(np.arange(len(agent.mean_max_q_value_history)), agent.mean_max_q_value_history, 1)[0]
							))
						else:
							print('\t\t.... / .... / ....')
						print('\t\t{:.3f} / {:.3f} / {:.3f}'.format(
							np.mean(agent.mean_min_q_value_history[max(0,len(agent.mean_min_q_value_history)-100):]),
							np.mean(agent.mean_mean_q_value_history[max(0,len(agent.mean_mean_q_value_history)-100):]),
							np.mean(agent.mean_max_q_value_history[max(0,len(agent.mean_max_q_value_history)-100):])
						))
						if len(agent.mean_min_q_value_history) >= 2:
							print('\t\t{:.3f} / {:.3f} / {:.3f}'.format(
								np.polyfit(np.arange(min(100,len(agent.mean_min_q_value_history))), agent.mean_min_q_value_history[max(0,len(agent.mean_min_q_value_history)-100):], 1)[0],
								np.polyfit(np.arange(min(100,len(agent.mean_mean_q_value_history))), agent.mean_mean_q_value_history[max(0,len(agent.mean_mean_q_value_history)-100):], 1)[0],
								np.polyfit(np.arange(min(100,len(agent.mean_max_q_value_history))), agent.mean_max_q_value_history[max(0,len(agent.mean_max_q_value_history)-100):], 1)[0],
							))
						else:
							print('\t\t.... / .... / ....')
					else:
						print('\n'.join(['\t\t.... / .... / ....'] * 5))
		else:
			print('\n'.join(['.' * 4] * (3 * self.max_agents + 3)))
		if not self.test_mode:
			print('Epsilon values:')
			for agent in sort_by_value(self.all_agents, lambda agent: agent.id):
				print('\tAgent {}: {}'.format(agent.id,
					'1.0 (overriding {})'.format(agent.epsilon)
					if agent.random_actions_are_forced() else
					agent.epsilon
				))
		print('-' * 128)
		self.on_print_simulation_info()

	def on_print_simulation_info(self):
		pass

	def set_render_mode(self, mode, time_scale = 1.0):
		'''
		Set the rendering mode.

		mode = None disables rendering after each step.

		mode = "image" will call render_as_image() after each step.

		mode = "text" will call render_as_text() after each step.

		time_scale adjusts approximately how many frames are skipped in rendering
		'''
		self.render_mode = mode
		self.render_time_scale = time_scale

	def render(self):
		if self.render_mode is None:
			return
		if self.render_mode == 'text':
			self.render_as_text()
		elif self.render_mode == 'image':
			self.render_as_image()

	def render_as_image(self):
		raise Exception('render_as_image() is not overridden.')

	def render_as_text(self):
		raise Exception('render_as_text() is not overridden.')

	def unify_agent_models(self, model):
		'''
		Enforce the currently existing agents to use the same instance of a Keras model.
		'''
		for agent in self.agents + self.inactive_agents:
			agent.set_model(model)

	def suppress_agent_models(self):
		'''
		Enforce the currently existing agents to take random actions with uniform distribution. The affected agents will need to load a Keras model to be able to use a model again.

		Agents that are added after calling this function will not automatically take random actions.
		'''
		for agent in self.agents + self.inactive_agents:
			agent.set_model(None)

	def set_agent_models(self, models, clone = False):
		'''
		models can be a list of Keras models, having the same length as the number of agents currently existing in the environment.

		models can also be a Keras model (not in a list).  Then, every agent gets that model.  If clone = True, then every agent gets a unique instance of the given model.
		'''
		if type(models) is list:
			if len(models) != len(self.agents) + len(self.inactive_agents):
				raise Exception('Cannot load {} models for {} agents.'.format(len(models), len(self.agents) + len(self.inactive_agents)))
			i = 0
			for agent in self.agents + self.inactive_agents:
				agent.set_model(models[i])
				i += 1
		else:
			if clone:
				self.set_agent_models([clone_model(models) for _ in range(self.max_agents)])
				for agent in self.agents + self.inactive_agents:
					agent.model.compile(models.optimizer, models.loss)
			else:
				self.set_agent_models([models] * self.max_agents)

	def load_agent_models(self, fpaths):
		'''
		fpaths should be a list of strings, each being a valid file path of a Keras model.

		The number of strings in fpaths should be equal to the number of currently existing agents in the environment.
		'''
		if len(fpaths) != len(self.agents) + len(self.inactive_agents):
			raise Exception('Cannot load {} models for {} agents.'.format(len(fpaths), len(self.agents) + len(self.inactive_agents)))
		i = 0
		for agent in self.agents + self.inactive_agents:
			agent.load_model(fpaths[i])
			i += 1

	def save_agent_models(self):
		'''
		Save each agent's model.
		'''
		self.models_currently_saved.clear()
		for agent in self.agents + self.inactive_agents:
			agent.save_model()

	def save_sim_history(self):
		if self.has_saved_sim_history:
			return
		if self.random_episodes_simulated < self.initial_random_episodes:
			return
		np.savez('{}/sim_history.npz'.format(self.model_save_location),
			recent_episode_length_history = self.recent_mean_episode_length_history,
			recent_episode_duration_history = self.recent_mean_episode_ms_duration_history,
			recent_transitions_saved_history = self.recent_mean_transitions_saved_history,
		)
		self.has_saved_sim_history = True

	def memory_ram_estimate(self):
		'''
		Print in the output the estimated total RAM usage when transition memory is full.
		This is estimated by calculating the total number of bits used in storing each number of the state.
		However, this is not a perfectly accurate estimation of RAM usage by the transition memory.
		'''
		print('Estimated RAM usage of full transition memory: {} GB'.format(np.product(self.agents[0].model_input_shape) * -1e-3 * (self.memory.capacity * 1.6e-5)))

	def run_loop(self, max_total_environment_steps = np.inf, max_episodes = np.inf):
		'''
		Start training the agents with a multiple agent deep Q-learning (MADQL) algorithm.

		max_total_environment_steps, max_episodes, or both can be set to limit the duration of training.
		'''
		self.reset()
		while self.total_steps_simulated < max_total_environment_steps and self.current_episode < max_episodes:
			self.step()

	def test_models(self, fpaths = None, episodes = 100, show_evaluations = False, check_winning = False):
		'''
		Simulate the environment with the agents having zero epsilon value (no randomly chosen actions).
		fpaths can be a list of file paths for each of the agent's models.  If fpaths = None, then the models currently set for each agent are used.
		The list can contain None values, which would make the corresponding agent choose actions randomly during the test.
		This list can also have values that are the "human" string, which makes the corresponding agent ask for human input before taking each action.
		Use print_simulation_info() while overriding episode_ended() to see the performance of each model during the test.
		Use set_render_mode() to render the steps of the environment as images or text while the models are being tested.

		During the test, agents do not record transitions, do experience replay, nor save their current model in use to a file.

		Return a list of the mean accumulated rewards for each agent once the test is completed.
		'''
		if fpaths is not None:
			if type(fpaths) is not list:
				raise TypeError('fpaths must be a list of file paths for Keras models')
			if len(fpaths) != self.max_agents:
				raise Exception('fpaths is expected to have a length of max_agents')
		self.test_mode = True
		self.do_check_winning = check_winning
		self.test_models_result = {
			'winners': [0] * self.max_agents,
			'ties': {},
		}
		self.reset()
		self.initial_random_episodes = 0
		for agent in self.agents + self.inactive_agents:
			agent.epsilon = 0
			agent.epsilon_restart_interval = np.inf
			agent.record_transitions = False
			agent.experience_replay = False
			agent.do_model_saving = False
			agent.do_show_evaluations = show_evaluations
			if fpaths is not None:
				if fpaths[agent.id] is None or fpaths[agent.id] == 'random':
					agent.set_model(None)
					agent.random_actions_are_valid = True
				elif fpaths[agent.id] == 'human':
					agent.human_input = True
				else:
					print(f'Set agent {agent.id} to model \'{fpaths[agent.id]}\'')
					agent.load_model(fpaths[agent.id])
			if agent.do_show_evaluations:
				agent.show_evaluations()
		while self.current_episode < episodes:
			self.step()
		self.print_simulation_info()
		self.test_mode = False
		rewards = [self.get_agent(agent_id).mean_accumulated_reward for agent_id in range(self.max_agents)]
		print('TEST RESULTS:\nOverall mean accumulated rewards per agent:')
		enumlist = list(enumerate(rewards))
		for agent_id, reward in reversed(sort_by_value(enumlist, lambda t: t[1])):
			print(f'\t{agent_id} | {reward}')
		if self.do_check_winning:
			print('Winners:')
			enumlist = list(enumerate(self.test_models_result['winners']))
			for agent_id, times_won in reversed(sort_by_value(enumlist, lambda t: t[1])):
				print(f'\t{agent_id} | {times_won}   ({100.*times_won/episodes:02.0f}%)')
			if len(self.test_models_result['ties']) > 0:
				print('Tied:')
				tiestrs = [('-'.join(str(k) for k in t), times_tied) for t, times_tied in self.test_models_result['ties'].items()]
				for s, times_tied in sort_by_value(tiestrs, lambda t: len(t[0])):
					print(f"\t{s} | {times_tied}")
			else:
				print('Tied: None')
		return rewards, dict(self.test_models_result)


