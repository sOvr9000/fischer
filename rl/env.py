
import numpy as np
from typing import Tuple, Union
import tensorflow as tf



class Env:
	def __init__(self, memory_capacity:int, model:tf.keras.Model, fit_batch_size:int, predict_batch_size:Union[int, None], tau:float, gamma:float, processors:int):
		self.state_shape = model.input_shape[1:]
		self.num_actions = model.output_shape[1]
		self.memory_capacity = memory_capacity
		self.memory_from_states = np.zeros((memory_capacity, *self.state_shape))
		self.memory_to_states = np.zeros((memory_capacity, *self.state_shape))
		self.memory_actions = np.zeros((memory_capacity,), dtype=int)
		self.memory_rewards = np.zeros((memory_capacity,))
		self.memory_terminal = np.zeros((memory_capacity,), dtype=int)
		self.memory_nva = np.zeros((memory_capacity, self.num_actions), dtype=bool)
		self.memory_index = 0
		self.model = model
		self.target_model = tf.keras.models.clone_model(self.model)
		self.target_model.set_weights(self.model.get_weights())
		self.fit_batch_size = fit_batch_size
		self.predict_batch_size = predict_batch_size if predict_batch_size is not None else int(.5+1.5*fit_batch_size)
		self.tau = tau
		self.gamma = gamma
		self.processors = processors
		self.use_multiprocessing = processors > 1
		self.current_states = None
	def add_transition(self, from_state:np.ndarray, to_state:np.ndarray, action:int, reward:float, terminal:bool, next_valid_actions:np.ndarray):
		i = self.memory_index % self.memory_capacity
		self.memory_from_states[i] = from_state
		self.memory_to_states[i] = to_state
		self.memory_actions[i] = action
		self.memory_rewards[i] = reward
		self.memory_terminal[i] = int(not terminal)
		self.memory_nva[i] = next_valid_actions
		self.memory_index += 1
	def sample_transitions(self, n:int):
		indices = np.random.randint(min(self.memory_index, self.memory_capacity), size=n)
		return self.memory_from_states[indices], self.memory_to_states[indices], self.memory_actions[indices], self.memory_rewards[indices], self.memory_terminal[indices], self.memory_nva[indices]
	def experience_replay(self, n:int):
		if self.memory_index == 0:
			return
		from_states, to_states, actions, rewards, terminal, next_valid_actions = self.sample_transitions(n)
		arange = np.arange(n, dtype=int)
		pred_from = self.model.predict(from_states, batch_size=self.predict_batch_size, workers=self.processors, use_multiprocessing=self.use_multiprocessing)
		pred_to = self.model.predict(to_states, batch_size=self.predict_batch_size, workers=self.processors, use_multiprocessing=self.use_multiprocessing)
		pred_to_target = self.target_model.predict(to_states, batch_size=self.predict_batch_size, workers=self.processors, use_multiprocessing=self.use_multiprocessing)
		pred_to[np.logical_not(next_valid_actions)] = -np.inf

		pred_from[arange, actions] = rewards + self.gamma * terminal * pred_to_target[arange, np.argmax(pred_to, axis=1)]

		self.model.fit(from_states, pred_from, batch_size=self.fit_batch_size, epochs=self.fit_epochs, shuffle=False, workers=self.processors, use_multiprocessing=self.use_multiprocessing)
		w = self.model.get_weights()
		tw = self.target_model.get_weights()
		self.target_model.set_weights([_w * self.tau + _tw * (1 - self.tau) for _w, _tw in zip(w, tw)])
	def pre_step(self, states:np.ndarray, valid_actions:np.ndarray=None):
		if valid_actions is None:
			valid_actions = np.ones((states.shape[0], self.num_actions), dtype=bool)
		from_states = self.current_states
		self.current_states = states
		self.current_valid_actions = valid_actions
		if from_states is not None:
			for t in zip(from_states, states, self.predicted_actions, self.rewards, self.terminal, self.current_valid_actions):
				self.add_transition(*t)
	def post_step(self, rewards:np.ndarray, terminal:np.ndarray):
		self.rewards = rewards
		self.terminal = terminal
	def reset(self):
		self.current_states = None
	def predict_actions(self) -> np.ndarray:
		pred = self.model.predict(self.current_states, batch_size=self.predict_batch_size, workers=self.processors, use_multiprocessing=self.use_multiprocessing)
		pred[np.logical_not(self.current_valid_actions)] = -np.inf
		self.predicted_actions = np.argmax(pred, axis=1)
		return self.predicted_actions


