
from tetrislib import *

class Polyomino:
	def __init__(self, squares):
		self.squares = squares
	def clone(self):
		return Polyomino([vec(s[0], s[1]) for s in self.squares])
	def set_position(self, pos):
		pos = vec_rounded(pos)
		center = vec_rounded(self.get_center())
		delta = pos - center
		for si in range(len(self.squares)):
			self.squares[si] += delta
	def get_center(self):
		return vec_center(self.squares)
	def translate(self, offset):
		for si in range(len(self.squares)):
			self.squares[si] += offset
		for s in self.squares:
			if s[0] < 0 or s[0] >= self.width or s[1] >= self.height:
				for si in range(len(self.squares)):
					self.squares[si] -= offset
				return False
			if s[1] < 0 or contains_vec(self.grounded_squares, s):
				for si in range(len(self.squares)):
					self.squares[si] -= offset
				return True # polyomino should stop falling now
		return False
	def rotate(self):
		center = self.get_center()
		for si in range(len(self.squares)):
			self.squares[si] = vec_rounded(center + vec_rotated270(self.squares[si] - center))
		for s in self.squares:
			if s[0] < 0 or s[0] >= self.width or s[1] < 0 or s[1] >= self.height or contains_vec(self.grounded_squares, s):
				for si in range(len(self.squares)):
					self.squares[si] = vec_rounded(center + vec_rotated90(self.squares[si] - center))
				break

