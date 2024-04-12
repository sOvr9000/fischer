

import numpy as np





class ConnectFour:
	def __init__(self, columns = 7, rows = 6):
		self.columns = columns
		self.rows = rows
	def reset(self):
		self.grid = np.zeros((self.rows, self.columns), dtype=np.int32)
		self.pieces_per_column = [0] * self.columns
		self.p1_to_move = True
		self.done = False
		self.winner = 0
	def drop_location(self, n):
		return self.rows - 1 - self.pieces_per_column[n]
	def can_move(self, n):
		return self.drop_location(n) >= 0
	def move(self, n):
		assert self.can_move(n), f'The move n={n} is illegal in the following position:\n{self}'
		self._push_move(n)
	def valid_moves(self):
		# for n in (3,2,4,1,5,0,6):
		# 	if self.can_move(n):
		# 		yield n
		for n in range(self.columns):
			if self.can_move(n):
				yield n
	def random_move(self):
		va = list(self.valid_moves())
		return va[np.random.randint(len(va))]
	def __repr__(self):
		return '\n'.join(' '.join('\u25a0' if self.grid[i,j] == 1 else '\u25a1' if self.grid[i,j] == -1 else ' ' for j in range(self.columns)) for i in range(self.rows))
	def _push_move(self, n):
		self.grid[self.drop_location(n),n] = int(self.p1_to_move)*2-1
		self.pieces_per_column[n] += 1
		self.p1_to_move = not self.p1_to_move
		self.done = all(c == self.rows for c in self.pieces_per_column)
		if not self.done:
			for i in range(self.rows):
				for j in range(self.columns-1):
					t = self.has_diagonal1(i,j)
					if t != 0 and self.has_diagonal1(i+1,j+1) == t and self.has_diagonal1(i+2,j+2) == t:
						self.done = True
						self.winner = t
					t = self.has_diagonal2(i,j)
					if t != 0 and self.has_diagonal2(i-1,j+1) == t and self.has_diagonal2(i-2,j+2) == t:
						self.done = True
						self.winner = t
					t = self.has_horizontal(i,j)
					if t != 0 and self.has_horizontal(i,j+1) == t and self.has_horizontal(i,j+2) == t:
						self.done = True
						self.winner = t
					t = self.has_vertical(i,j)
					if t != 0 and self.has_vertical(i+1,j) == t and self.has_vertical(i+2,j) == t:
						self.done = True
						self.winner = t
	def _pop_move(self, n):
		self.grid[self.drop_location(n)+1,n] = 0
		self.pieces_per_column[n] -= 1
		self.p1_to_move = not self.p1_to_move
		self.done = False
		self.winner = 0
	def has_diagonal1(self, i, j):
		if i+1 >= self.rows or j+1 >= self.columns:
			return 0
		return self.grid[i, j] * int(self.grid[i, j] == self.grid[i+1, j+1])
	def has_diagonal2(self, i, j):
		if i <= 0 or j+1 >= self.columns:
			return 0
		return self.grid[i, j] * int(self.grid[i, j] == self.grid[i-1, j+1])
	def has_spaced_diagonal1(self, i, j):
		if i+2 >= self.rows or j+2 >= self.columns:
			return 0
		return self.grid[i, j] * int(self.grid[i, j] == self.grid[i+2, j+2])
	def has_spaced_diagonal2(self, i, j):
		if i <= 1 or j+2 >= self.columns:
			return 0
		return self.grid[i, j] * int(self.grid[i, j] == self.grid[i-2, j+2])
	def has_horizontal(self, i, j):
		if j+1 >= self.columns:
			return 0
		return self.grid[i, j] * int(self.grid[i, j] == self.grid[i, j+1])
	def has_vertical(self, i, j):
		if i+1 >= self.rows:
			return 0
		return self.grid[i, j] * int(self.grid[i, j] == self.grid[i+1, j])




