
import numpy as np
from tetrislib import *
from polyomino import Polyomino



POLY_I = Polyomino([vec(0,0), vec(1,0), vec(2,0), vec(3,0)])
POLY_L = Polyomino([vec(0,0), vec(1,0), vec(2,0), vec(2,1)])
POLY_J = Polyomino([vec(0,0), vec(1,0), vec(2,0), vec(0,1)])
POLY_T = Polyomino([vec(0,0), vec(1,0), vec(2,0), vec(1,1)])
POLY_O = Polyomino([vec(0,0), vec(1,0), vec(0,1), vec(1,1)])
POLY_S = Polyomino([vec(0,0), vec(1,0), vec(1,1), vec(2,1)])
POLY_Z = Polyomino([vec(0,1), vec(1,1), vec(1,0), vec(2,0)])

POLYOMINOES = [POLY_I, POLY_L, POLY_J, POLY_T, POLY_O, POLY_S, POLY_Z]



class Tetris:
	def __init__(self, size):
		self.grounded_squares = []
		self.width = size[0]
		self.height = size[1]
	def next_piece(self):
		p = np.random.choice(POLYOMINOES).clone()
		p.set_position(vec(int(self.width/2), self.height-1))
		while True:
			for s in p.squares:
				if s[1] >= self.height:
					for si in range(len(p.squares)):
						p.squares[si] += VEC_DOWN
					break
			else:
				break # break the while loop, only when the for loop has iterated completely without breaking
		return p
	def reset(self):
		self.next_pieces = [self.next_piece()]
		self.current_polyomino = self.next_piece()
		self.grounded_squares.clear()
		self.score = 0
		self.steps = 0
		return self.get_state()
	def get_state(self):
		state = [0] * self.width * self.height
		for p in self.grounded_squares:
			state[p[0]+p[1]*self.width] = -1
		for p in self.current_polyomino.squares:
			state[p[0]+p[1]*self.width] = 1
		return np.array(state)
	def step(self, action):
		self.steps += 1
		self.hit_ground = False
		c = self.current_polyomino.get_center()
		if action == 1:
			self.translate_polyomino(self.current_polyomino, VEC_LEFT)
		elif action == 2:
			self.translate_polyomino(self.current_polyomino, VEC_RIGHT)
		elif action == 3:
			self.rotate_polyomino(self.current_polyomino)
		self.hit_ground = self.translate_polyomino(self.current_polyomino, VEC_DOWN)
		if np.array_equal(self.current_polyomino.get_center(), c):
			self.hit_ground = self.hit_ground or self.translate_polyomino(self.current_polyomino, VEC_DOWN)
		last_polyomino = self.current_polyomino
		if self.hit_ground:
			self.grounded_squares += self.current_polyomino.squares
			self.current_polyomino = self.next_pieces[0]
			del self.next_pieces[0]
			self.next_pieces.append(self.next_piece())
			for s in self.current_polyomino.squares:
				if contains_vec(self.grounded_squares, s):
					return self.get_state(), -10, True, {}
		new_state = self.get_state()
		cleared_lines = []
		if self.hit_ground:
			_,poly_ys = zip(*last_polyomino.squares)
			upper = max(poly_ys)
			lower = min(poly_ys)
			for y in range(upper,lower-1,-1):
				for x in range(self.width):
					if new_state[y*self.width+x] != -1:
						break
				else: # if for loop does not break (i.e. this entire line is full)...
					cleared_lines.append(y)
					for si in range(len(self.grounded_squares)-1,-1,-1):
						if self.grounded_squares[si][1] == y:
							del self.grounded_squares[si]
			self.score += len(cleared_lines)
			if len(cleared_lines) > 0:
				lowest = min(cleared_lines)
				for s in self.grounded_squares:
					if s[1] > lowest:
						for y in cleared_lines:
							if s[1] > y:
								s[1] -= 1
				new_state = self.get_state()
		cleared_lines = len(cleared_lines)
		done = False
		if self.hit_ground:
			for s in last_polyomino.squares:
				if s[1] >= self.height - 1:
					done = True
					break
		reward = -10 if done else cleared_lines * 5# if cleared_lines > 0 else 0.05
		return new_state, reward, done, {}




	# def fix_position(self, poly):
	# 	hit_ground = False
	# 	while True:
	# 		for s in poly.squares:
	# 			if s[0] >= self.width:
	# 				for si in range(len(poly.squares)):
	# 					poly.squares[si] += VEC_LEFT
	# 				break # break the for loop
	# 			if s[0] < 0:
	# 				for si in range(len(poly.squares)):
	# 					poly.squares[si] += VEC_RIGHT
	# 				break # break the for loop
	# 			if s[1] < 0:
	# 				for si in range(len(poly.squares)):
	# 					poly.squares[si] += VEC_UP
	# 				hit_ground = True
	# 				break # break the for loop
	# 			if s[1] >= self.height:
	# 				for si in range(len(poly.squares)):
	# 					poly.squares[si] += VEC_DOWN
	# 				break
	# 		else:
	# 			break # break the while loop, only when the for loop has iterated completely without breaking
	# 	return hit_ground
