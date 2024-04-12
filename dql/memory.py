
import numpy as np
from typing import Union



class TransitionMemory:
    def __init__(self, capacity: int, state_shape: Union[tuple, list[tuple]], action_shape: Union[tuple, list[int]]):
        self.capacity = capacity
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_dims = len(self.action_shape)
        self.is_multi_action = isinstance(action_shape, list)
        self.is_multi_state = isinstance(self.state_shape, list) and all(isinstance(v, tuple) for v in self.state_shape)
        if self.is_multi_state:
            self.old_states = [
                np.zeros((capacity, *s))
                for s in self.state_shape
            ]
            self.new_states = [
                np.zeros((capacity, *s))
                for s in self.state_shape
            ]
        else:
            self.old_states = np.zeros((capacity, *state_shape))
            self.new_states = np.zeros_like(self.old_states)
        self.rewards = np.zeros(capacity)
        self.actions = np.zeros((capacity, self.action_dims), dtype=int)
        self.terminal = np.zeros(capacity, dtype=bool)
        if self.is_multi_action:
            self.next_valid_actions = []
        else:
            self.next_valid_actions = np.zeros((capacity, *action_shape), dtype=bool)
        self.clear()
    def record_transition(self, old_state: Union[np.ndarray, list[np.ndarray]], new_state: Union[np.ndarray, list[np.ndarray]], reward: float, action: np.ndarray, terminal: bool, next_valid_actions: Union[np.ndarray, list[tuple]] = None, multi_action: bool = False):
        '''
        Record a transition from state `old_state` to state `new_state` by taking action `action`, which yields the reward `reward`, where `terminal` is whether `new_state` is a terminal state and `next_valid_actions` is a binary mask of which actions are legal to take in state `new_state`.

        States are allowed to be lists of arrays if the each environment state consists of multiple arrays.

        `next_valid_actions` can only be left as `None` if the environment is not multi-action. It can be an array of boolean values to function as a mask over possible actions in the environment. Otherwise, if the environment is not multi-action, it must be a list of tuples and cannot be left as `None`.
        '''
        if next_valid_actions is None:
            next_valid_actions = np.ones(self.next_valid_actions.shape[1:], dtype=bool)
        ctm = self.current_transition % self.capacity
        if self.is_multi_state:
            for i in range(len(self.old_states)):
                self.old_states[i][ctm] = old_state[i]
                self.new_states[i][ctm] = new_state[i]
        else:
            self.old_states[ctm] = old_state
            self.new_states[ctm] = new_state
        self.rewards[ctm] = reward
        self.actions[ctm] = action
        self.terminal[ctm] = terminal
        if self.is_multi_action:
            if len(self.next_valid_actions) >= self.capacity:
                self.next_valid_actions[ctm] = next_valid_actions
            else:
                self.next_valid_actions.append(next_valid_actions)
        else:
            self.next_valid_actions[ctm] = next_valid_actions
        self.current_transition += 1
    def record_transitions(self, old_states: Union[np.ndarray[float], list[np.ndarray[float]]], new_states: Union[np.ndarray[float], list[np.ndarray[float]]], rewards: np.ndarray, actions: np.ndarray, terminals: np.ndarray, next_valid_actions: Union[np.ndarray[bool], list[list[tuple]]] = None):
        '''
        Record a series of transitions stored in contiguous arrays.
        '''
        if next_valid_actions is None:
            if self.is_multi_action:
                raise ValueError(f'next_valid_actions must be provided if actions consist of multiple arrays.')
            else:
                next_valid_actions = np.ones((rewards.shape[0], *self.next_valid_actions.shape[1:]), dtype=bool)
        if self.is_multi_state:
            for i in range(rewards.shape[0]):
                self.record_transition(
                    old_state=[
                        s[i]
                        for s in old_states
                    ],
                    new_state=[
                        s[i]
                        for s in new_states
                    ],
                    reward=rewards[i],
                    action=actions[i],
                    terminal=terminals[i],
                    next_valid_actions=next_valid_actions[i]
                )
        else:
            for t in zip(old_states, new_states, rewards, actions, terminals, next_valid_actions):
                self.record_transition(*t)
        del old_states, new_states, rewards, actions, terminals, next_valid_actions
    def sample(self, count: int) -> tuple[Union[np.ndarray[float], list[np.ndarray[float]]], Union[np.ndarray[float], list[np.ndarray[float]]], np.ndarray[float], np.ndarray[int], np.ndarray[bool], Union[np.ndarray[bool], list[list[tuple]]]]:
        indices = np.random.randint(min(self.capacity, self.current_transition), size=count)
        if self.is_multi_state:
            return [
                old_states[indices]
                for old_states in self.old_states
            ], [
                new_states[indices]
                for new_states in self.new_states
            ], self.rewards[indices], self.actions[indices], self.terminal[indices], self.next_valid_actions[indices]
        if self.is_multi_action:
            nvas = list(map(self.next_valid_actions.__getitem__, indices))
        else:
            nvas = self.next_valid_actions[indices]
        return self.old_states[indices], self.new_states[indices], self.rewards[indices], self.actions[indices], self.terminal[indices], nvas
    def clear(self):
        self.current_transition = 0
        if self.is_multi_state:
            for arr in self.old_states + self.new_states:
                arr[:] = 0
        else:
            self.old_states[:] = 0
            self.new_states[:] = 0
        self.rewards[:] = 0
        self.actions[:] = 0
        self.terminal[:] = False
        if self.is_multi_action:
            self.next_valid_actions.clear()
        else:
            self.next_valid_actions[:] = True


