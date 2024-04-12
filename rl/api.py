
import numpy as np
from typing import Tuple, Union
from .env import Env
import tensorflow as tf



class NotInitializedError(Exception):
	pass



_fischer_rl_system:Env = None



def init(
	memory_capacity:int,
	model:tf.keras.Model,
	fit_batch_size:int,
	predict_batch_size:Union[int, None]=None,
	tau:float=0.2,
	gamma:float=0.98,
	processors:int=8,
):
	'''
	Initialize the reinforcement learning system.

	`processors` is the number of logical processors to use while making model predictions.  If `processors=1`, then multiprocessing is not utilized.
	'''
	global _fischer_rl_system
	_fischer_rl_system = Env(
		memory_capacity=memory_capacity,
		model=model,
		fit_batch_size=fit_batch_size,
		predict_batch_size=predict_batch_size,
		tau=tau,
		gamma=gamma,
		processors=processors,
	)

def add_transition(
	from_state:np.ndarray,
	to_state:np.ndarray,
	action:int,
	reward:float,
	terminal:bool,
	next_valid_actions:np.ndarray,
):
	'''
	Save a transition from one state to another state.
	'''
	if _fischer_rl_system is None:
		raise NotInitializedError('Please call rl.init() before using other functions!')
	_fischer_rl_system.add_transition(from_state, to_state, action, reward, terminal, next_valid_actions)

def add_transitions(
	from_states:np.ndarray,
	to_states:np.ndarray,
	actions:np.ndarray,
	rewards:np.ndarray,
	terminal:np.ndarray,
	next_valid_actions:np.ndarray,
):
	'''
	Save many transitions from previous states to current states.
	'''
	if _fischer_rl_system is None:
		raise NotInitializedError('Please call rl.init() before using other functions!')
	for t in zip(from_states, to_states, actions, rewards, terminal, next_valid_actions):
		_fischer_rl_system.add_transition(*t)

def experience_replay(
	num_transitions:int,
):
	'''
	Train the model on its own performance.
	'''
	if _fischer_rl_system is None:
		raise NotInitializedError('Please call rl.init() before using other functions!')
	_fischer_rl_system.experience_replay(num_transitions)

def pre_step(
	states:np.ndarray,
	valid_actions:np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
	'''
	Given `states`, return `(from_states, to_states)`.  `from_states` will be None after each `rl.reset()`.

	`valid_actions` is an array of bools to indicate which actions are valid in `states` (where True means an action is valid).
	'''
	if _fischer_rl_system is None:
		raise NotInitializedError('Please call rl.init() before using other functions!')
	return _fischer_rl_system.pre_step(states, valid_actions)

def reset():
	'''
	Reset the environment.  Training data is not reset.
	'''
	if _fischer_rl_system is None:
		raise NotInitializedError('Please call rl.init() before using other functions!')
	_fischer_rl_system.reset()

def predict_actions() -> np.ndarray:
	'''
	Use the model to make a prediction of the next actions to take in the current states of the environment.
	'''
	if _fischer_rl_system is None:
		raise NotInitializedError('Please call rl.init() before using other functions!')
	return _fischer_rl_system.predict_actions()


