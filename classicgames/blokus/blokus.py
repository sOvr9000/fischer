

import numpy as np



class Blokus:
	POLYOMINOES = [ # These are all polyominoes of 5 squares or fewer where no polyomino in this set is a reflection and/or rotation of another.
		((0,0),),
		((0,0),(1,0)),
		((0,0),(1,0),(2,0)),
		((0,0),(1,0),(0,1)),
		((0,0),(1,0),(2,0),(3,0)),
		((0,0),(1,0),(2,0),(0,1)),
		((0,0),(1,0),(2,0),(1,1)),
		((0,0),(1,0),(0,1),(1,1)),
		((0,0),(1,0),(1,1),(2,1)),
		((0,0),(1,0),(2,0),(3,0),(4,0)),
		((0,0),(1,0),(2,0),(3,0),(0,1)),
		((0,0),(1,0),(2,0),(2,1),(3,1)),
		((0,0),(1,0),(2,0),(0,1),(1,1)),
		((0,0),(1,0),(2,0),(0,1),(2,1)),
		((0,0),(1,0),(2,0),(3,0),(1,1)),
		((0,0),(1,0),(2,0),(1,1),(1,2)),
		((0,0),(1,0),(2,0),(0,1),(0,2)),
		((0,0),(1,0),(1,1),(2,1),(2,2)),
		((0,0),(1,0),(1,1),(1,2),(2,2)),
		((0,0),(1,0),(1,1),(1,2),(2,1)),
		((0,0),(1,0),(2,0),(1,1),(1,-1)),
		#((0,0),(1,0),(1,1),(2,1),(2,2),(3,2),(3,3)),
	]
	PLAYER_SYMBOLS = [
		# '\u16DC',
		# '\u16ED',
		# '\u2836',
		# '\u16BC',
		'-',
		'/',
		'|',
		'\\',
	]
	EMPTY_SYMBOL = ' '#'\u16EB'
	def __init__(self):
		self.piece_set = []
		for i,p in enumerate(self.POLYOMINOES):
			s = []
			for ref in (p,tuple((-x,y) for x,y in p)):
				rot = ref
				for _ in range(4):
					rot = tuple((-y,x) for x,y in rot)
					for x,y in rot:
						if (x-1,y) not in rot and (x,y-1) not in rot:
							t = tuple(sorted((_x-x,_y-y) for _x,_y in rot))
							if t not in s:
								s.append(t)
			self.piece_set.append(s)
		self.generate_placement_map()
		self.num_polyominoes = len(self.POLYOMINOES)
	def generate_placement_map(self):
		self.placement_map = []
		self.placement_map_inv = {}
		self.polyomino_indices = {}
		for i,s in enumerate(self.piece_set):
			for p in s:
				self.placement_map_inv[p] = len(self.placement_map)
				self.placement_map.append(p)
				self.polyomino_indices[p] = i
		print(f'Total number of ways to place pieces: {len(self.placement_map)}')
	def reset(self):
		self.board = -np.ones((22,22), dtype=np.int8) # -1 = empty, N = player N
		self.turn = 0 # player 0 begins
		self.pieces_placed = np.zeros((4,self.num_polyominoes), dtype=np.bool)
		self.vmoves = None
		self.done = False
		self.dead = [False] * 4
	def can_move(self, p, r, x, y):
		'''
		Return whether it's currently possible to call Blokus.move(p, r, x, y).
		'''
		if p < 0 or p >= len(self.placement_map):
			return False
		poly = self.placement_map[p]
		if self.pieces_placed[self.turn,self.polyomino_indices[poly]]:
			return False
		if r == 1:
			poly = tuple((-y,x) for x,y in poly)
		elif r == 2:
			poly = tuple((-x,-y) for x,y in poly)
		elif r == 3:
			poly = tuple((y,-x) for x,y in poly)
		for px,py in poly:
			_x,_y = x+px,y+py
			if _x < 0 or _x >= 22 or _y < 0 or _y >= 22:
				return False
			if self.board[_y,_x] != -1:
				return False
			if (_x < 21 and self.board[_y,_x+1] == self.turn) or (_x > 0 and self.board[_y,_x-1] == self.turn) or (_y < 21 and self.board[_y+1,_x] == self.turn) or (_y > 0 and self.board[_y-1,_x] == self.turn):
				return False
		return True
	def move(self, p, r, x, y):
		'''
		At position (x, y), which must be touching corners with one of the current player's placed pieces, place the piece p with rotation r.

		The piece cannot be placed if it touches sides with any of the current player's placed pieces or if it overlaps with any placed piece on the board.

		Note: All pieces p assume the currently placed corner is to the lower left, at relative position (-1,-1), and the newly placed corner (to touch the currently placed one) will be at relative position (0,0).
		To branch off of corners that point in different angles from NE, use a different r.  In terms of the direction in which the currently placed corner points, r = 0 means NE, r = 1 means NW, r = 2 means SW, and r = 3 means SE.
		'''
		assert self.can_move(p, r, x, y), f'The move {p, r, x, y} is invalid in the current state:\n{self}'
		self.vmoves = None
		poly = self.placement_map[p]
		self.pieces_placed[self.turn, self.polyomino_indices[tuple(sorted(poly))]] = True
		if r == 1:
			poly = tuple((-y,x) for x,y in poly)
		elif r == 2:
			poly = tuple((-x,-y) for x,y in poly)
		elif r == 3:
			poly = tuple((y,-x) for x,y in poly)
		for px,py in poly:
			self.board[y+py,x+px] = self.turn
		while not self.done:
			self.turn = (self.turn + 1) % 4
			if not self.dead[self.turn]:
				if len(self.valid_moves()) == 0:
					self.dead[self.turn] = True
					if all(self.dead):
						self.done = True
				else:
					break
	def _valid_moves(self):
		va = []
		if self.turn == 0 and self.board[0,0] == -1:
			return [
				(p,0,0,0)
				for p,poly in enumerate(self.placement_map)
				if self.can_move(p, 0, 0, 0)
			]
		elif self.turn == 1 and self.board[0,21] == -1:
			return [
				(p,1,21,0)
				for p,poly in enumerate(self.placement_map)
				if self.can_move(p, 1, 21, 0)
			]
		elif self.turn == 2 and self.board[21,21] == -1:
			return [
				(p,2,21,21)
				for p,poly in enumerate(self.placement_map)
				if self.can_move(p, 2, 21, 21)
			]
		elif self.turn == 3 and self.board[21,0] == -1:
			return [
				(p,3,0,21)
				for p,poly in enumerate(self.placement_map)
				if self.can_move(p, 3, 0, 21)
			]
		for y in range(22):
			for x in range(22):
				if self.board[y,x] == self.turn:
					if y < 21 and self.board[y+1,x] == -1:
						if x < 21 and self.board[y,x+1] == -1:
							for p,poly in enumerate(self.placement_map):
								if self.can_move(p, 0, x+1, y+1):
									va.append((p,0,x+1,y+1))
						if x > 0 and self.board[y,x-1] == -1:
							for p,poly in enumerate(self.placement_map):
								if self.can_move(p, 1, x-1, y+1):
									va.append((p,1,x-1,y+1))
					if y > 0 and self.board[y-1,x] == -1:
						if x > 0 and self.board[y,x-1] == -1:
							for p,poly in enumerate(self.placement_map):
								if self.can_move(p, 2, x-1, y-1):
									va.append((p,2,x-1,y-1))
						if x < 21 and self.board[y,x+1] == -1:
							for p,poly in enumerate(self.placement_map):
								if self.can_move(p, 3, x+1, y-1):
									va.append((p,3,x+1,y-1))
		return va
	def valid_moves(self):
		if self.vmoves is None:
			self.vmoves = self._valid_moves()
		return self.vmoves
	def random_move(self):
		va = self.valid_moves()
		return va[np.random.randint(len(va))]
	def __repr__(self):
		return '\n'.join(' '.join(self.PLAYER_SYMBOLS[self.board[y,x]] if self.board[y,x] >= 0 else self.EMPTY_SYMBOL for x in range(22)) for y in range(21,-1,-1))



if __name__ == '__main__':
	game = Blokus()
	game.reset()

	while not game.done:
		print()
		print(game)
		va = game.valid_moves()
		p,r,x,y = va[np.random.randint(len(va))]
		game.move(p,r,x,y)
		print(f'Move made: {p,r,x,y}')
