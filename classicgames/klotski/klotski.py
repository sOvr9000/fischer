



import numpy as np



class Klotski:
	def __init__(self, board_size: tuple[int, int], goal_positions: list[tuple[int, int]], biaxial_sliding: bool = False):
		# goal_positions is a list of positions that can accept goal blocks; once all goal blocks are in goal positions, the game is won
		# if biaxial_sliding=False, then blocks can only slide along their major axis, i.e. a block that is skinny in the y-axis can only slide along the x-axis and vice versa
		# otherwise, block sliding is not limited by the size of each block
		# a square block is always able to slide in both directions, for either sliding mode
		if not isinstance(board_size, tuple):
			raise TypeError
		if len(board_size) != 2:
			raise ValueError
		if not all(isinstance(v, int) for v in board_size):
			raise TypeError
		if not isinstance(goal_positions, list):
			raise TypeError
		if any(len(v) != 2 for v in goal_positions):
			raise ValueError
		if any(not all(isinstance(v, int) for v in u) for u in goal_positions):
			raise TypeError
		if not isinstance(biaxial_sliding, bool):
			raise TypeError
		self.height, self.width = board_size
		self.biaxial_sliding = biaxial_sliding
		self.grid = -np.ones(board_size, dtype=int)
		self.blocks: list[tuple[int, int, int, int]] = []
	def add_block(self, x: int, y: int, w: int, h: int) -> int:
		'''
		Add a block to the puzzle.  Return the index in Klotski.blocks that represents the new block.
		'''
		if not isinstance(x, int) or not isinstance(y, int) or not isinstance(w, int) or not isinstance(h, int):
			raise TypeError
		if x < 0 or x >= self.width or y < 0 or y >= self.height or w <= 0 or h <= 0 or x + w >= self.width or y + h >= self.height:
			raise ValueError
		i = len(self.blocks)
		self.blocks.append((x, y, w, h))
		for _y in range(y, y + h):
			for _x in range(x, x + w):
				self.grid[_y, _x] = i
		return i
	def __str__(self) -> str:
		a = [['\u25a1'] * self.width for _ in range(self.height)]
		for x, y, w, h in self.blocks:
			for _y in range(y, y + h):
				for _x in range(x, x + w):
					a[_y][_x] = '\u25a0'
		return '\n'.join(
			'  '.join(row)
			for row in a
		)


game = Klotski(board_size=(5, 6), goal_positions=[(0, 0)], biaxial_sliding=False)

print(game)
