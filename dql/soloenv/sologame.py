
import numpy as np
from typing import Union



__all__ = ['SoloGame']

class SoloGame:
    '''
    Models a one-player game where the goal is to maximize the score (the "static" reward).

    You must override `SoloGame.reset()`, `SoloGame.step()`, `SoloGame.get_state()`, `SoloGame.get_static_reward()`.
    
    For further control, you may override `SoloGame.is_terminal()` and `SoloGame.get_action_mask()`.
    '''
    def __init__(self):
        self.sim_index: int = -1
    def reset(self):
        '''
        Reset the environment.
        '''
        raise NotImplementedError
    def step(self, action: Union[int, np.ndarray]):
        '''
        Take one step in the game given action `action`.
        '''
        raise NotImplementedError
    def get_state(self) -> np.ndarray[float]:
        '''
        Return the current state of the environment with values normalized so that the model is capable of interpreting the state.
        '''
        raise NotImplementedError
    def get_action_mask(self) -> np.ndarray[bool]:
        '''
        Return a boolean mask over the action space that describes which actions are legal in the current state of the environment.

        By default, this function returns `None`, which means all actions are assumed to be legal.
        '''
        return None
    def get_valid_actions(self) -> list[tuple]:
        '''
        Return a list of the possible actions which are each a tuple of integers.

        This only needs to be overridden when the model has multiple 1D outputs.  In this case, `SoloGame.get_action_mask()` does not need to be overridden.
        '''
        raise NotImplementedError
    def get_static_reward(self) -> float:
        '''
        Return an absolute evaluation of the current state.

        The first differences of the values returned by this function are used as the reward function in the DQL algorithm.
        '''
        raise NotImplementedError
    def is_terminal(self) -> bool:
        '''
        Return whether the game is in a terminal state.

        By default, this function returns `False`.
        '''
        return False
    def get_heuristic_action(self) -> Union[int, np.ndarray]:
        return None
    def _reset(self):
        self.total_steps = 0
        self.accumulated_reward = 0
        self.reset()
    def _step(self, action: Union[int, np.ndarray]):
        self.step(action)
        self.total_steps += 1
    def _get_valid_actions(self) -> list[tuple]:
        return self.get_valid_actions()
