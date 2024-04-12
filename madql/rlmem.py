
import numpy as np

class RLMemory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.transitions = []
		self.transition_index = 0
		self.was_reset = False
		self.recently_added = 0

	def sample(self, size):
		'''
		Randomly sample from previously experienced transitions.
		'''
		if len(self.transitions) == 0:
			return []
		return [self.transitions[i] for i in np.random.randint(0, len(self.transitions), size = size)]

	def sample_all(self):
		return list(self.transitions)

	def add_transition(self, transition):
		'''
		Add a transition to the memory.
		'''
		self.was_reset = False
		self.recently_added += 1
		if len(self.transitions) < self.capacity:
			self.transitions.append(transition)
		else:
			self.transition_index = (self.transition_index + 1) % len(self.transitions)
			self.transitions[self.transition_index] = transition

	def on_reset(self, clear_recently_added):
		if self.was_reset:
			return
		self.was_reset = True
		if clear_recently_added:
			self.recently_added = 0
