import numpy as np

class WeightedChoice:
	def __init__(self, choices):
		'''
		choices is a either a list of tuples of the form (weight, item).  Cannot be empty.
		'''
		self.set_choices(choices)
	def set_choices(self, choices):
		if type(choices) is not list or len(choices) == 0:
			raise TypeError('choices must be a nonempty list')
		if any(type(e) is not tuple or len(e) != 2 for e in choices):
			raise TypeError('Not all elements of choices are tuples of length two')
		if not all(isinstance(weight, (int, float)) and not isinstance(weight, bool) for weight,_ in choices):
			raise TypeError('Not all weights are floats or integers')
		if not all(weight > 0 for weight,_ in choices):
			raise ValueError('Not all weights are greater than zero (a zero-weight item should be excluded altogether if the zero weight is intended)')
		# items can be of any type, for the sake of versatility
		self.choices = choices
		self.weight_sum = sum(w for w,_ in self.choices)
	def choice(self):
		r = np.random.random() * self.weight_sum
		s = 0.
		for w,i in self.choices:
			s += w
			if r <= s:
				return i
		return self.choices[-1][1]
	def __call__(self):
		return self.choice()
