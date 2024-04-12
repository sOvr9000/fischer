

from random import random, shuffle, randint



class Izzi:
	def __init__(self, size=4):
		self.size = size
	def reset(self):
		'''
		Clear the board and generate a new set of tiles.
		'''
		self.board = [[None]*self.size for _ in range(self.size)]
		self.generate_tiles()
	def clear_board(self):
		'''
		Clear the board and keep the same set of tiles.
		'''
		for y in range(self.size):
			for x in range(self.size):
				if not self.is_empty(x, y):
					self.pick_up_tile(x, y)
		shuffle(self.tiles)
		for tile in self.tiles:
			tile.rotate(randint(0,3))
	def generate_tiles(self):
		tiles = [[Tile(x, y) for x in range(self.size)] for y in range(self.size)]
		for y in range(self.size):
			for x in range(self.size):
				t = tiles[y][x]
				if x == 0:
					t.colors[4] = int(random() < 0.5)
					t.colors[5] = int(random() < 0.5)
				if y == 0:
					t.colors[6] = int(random() < 0.5)
					t.colors[7] = int(random() < 0.5)
				t.colors[0] = int(random() < 0.5)
				t.colors[1] = int(random() < 0.5)
				t.colors[2] = int(random() < 0.5)
				t.colors[3] = int(random() < 0.5)
				if x+1 < self.size:
					tiles[y][x+1].colors[4] = t.colors[1]
					tiles[y][x+1].colors[5] = t.colors[0]
				if y+1 < self.size:
					tiles[y+1][x].colors[6] = t.colors[3]
					tiles[y+1][x].colors[7] = t.colors[2]
		for y in range(self.size):
			for x in range(self.size):
				t = tiles[y][x]
				v = 0 if all(t.colors) else 1 if not any(t.colors) else -1
				if v == 0 or v == 1:
					k = randint(0, 7)
					t.colors[k] = v
					if (k==0 or k==1) and x+1 < self.size:
						tiles[y][x+1].colors[5-k] =  v
					elif (k==2 or k==3) and y+1 < self.size:
						tiles[y+1][x].colors[9-k] = v
					elif (k==4 or k==5) and x > 0:
						tiles[y][x-1].colors[5-k] = v
					elif (k==6 or k==7) and y > 0:
						tiles[y-1][x].colors[9-k] = v
		self.tiles = [t for row in tiles for t in row]
		self.shuffle_tiles()
		for tile in self.tiles:
			tile.rotate(randint(0,3))
	def shuffle_tiles(self):
		shuffle(self.tiles)
	def is_solved(self):
		return all(self.board[y][x] is not None and self.is_valid_placement(self.board[y][x], x, y, False) for y in range(self.size) for x in range(self.size))
	def is_valid_placement(self, tile, x, y, check_all_sides = True):
		if self.board[y][x] is not None:
			return False
		if x + 1 < self.size:
			if self.board[y][x+1] is not None:
				if self.board[y][x+1].colors[5] != tile.colors[0] or self.board[y][x+1].colors[4] != tile.colors[1]:
					return False
		if y + 1 < self.size:
			if self.board[y+1][x] is not None:
				if self.board[y+1][x].colors[7] != tile.colors[2] or self.board[y+1][x].colors[6] != tile.colors[3]:
					return False
		if check_all_sides:
			if x > 0:
				if self.board[y][x-1] is not None:
					if self.board[y][x-1].colors[1] != tile.colors[4] or self.board[y][x-1].colors[0] != tile.colors[5]:
						return False
			if y > 0:
				if self.board[y-1][x] is not None:
					if self.board[y-1][x].colors[3] != tile.colors[6] or self.board[y-1][x].colors[2] != tile.colors[7]:
						return False
		return True
	def get_tile_index(self, tile):
		for i,t in enumerate(self.tiles):
			if t == tile:
				return i
	def place_tile(self, tiles_index, x, y):
		if not self.is_on_board(x, y) or not self.is_empty(x, y):
			return False
		if tiles_index >= len(self.tiles) or tiles_index < 0:
			return False
		if not self.is_valid_placement(self.tiles[tiles_index], x, y):
			return False
		self.board[y][x] = self.tiles.pop(tiles_index)
		return True
	def pick_up_tile(self, x, y):
		if not self.is_on_board(x, y) or self.is_empty(x, y):
			return False
		self.tiles.append(self.board[y][x])
		self.board[y][x] = None
		return True
	def rotate_tiles(self, k=1):
		for tile in self.tiles:
			tile.rotate(k=k)
	def is_on_board(self, x, y):
		return x >= 0 and x < self.size and y >= 0 and y < self.size
	def is_empty(self, x, y):
		return not self.is_on_board(x, y) or self.board[y][x] is None
	def apply_solution(self, solution):
		'''
		This function applies solutions found with Izzi.find_solutions().  This places all tiles at their respective locations with their respective rotations.
		'''
		for tile, (x, y, r) in solution.items():
			tile.set_rotation(r)
			self.place_tile(self.get_tile_index(tile), x, y)
	def solve(self):
		'''
		Return the pieces to their originally generated positions and rotations.
		'''
		for y in range(self.size):
			for x in range(self.size):
				self.pick_up_tile(x, y)
		while len(self.tiles) > 0:
			t = self.tiles[0]
			if not self.place_tile(0, t.solution_x, t.solution_y):
				break
			self.board[t.solution_y][t.solution_x].unrotate()
	def valid_placements(self, tile):
		'''
		Return the valid positions for the given tile in its current rotation.
		'''
		for y in range(self.size):
			for x in range(self.size):
				if self.is_valid_placement(tile, x, y, check_all_sides = True):
					yield x, y
	def all_valid_placements(self):
		'''
		Return a dictionary that maps each tile to four distinct lists (potentially all empty for some tile,
		in which case the puzzle is currently unsolvable) which contains all possible positions where it can be placed.
		'''
		d = {}
		for tile in self.tiles:
			l = [[],[],[],[]]
			for r in range(4):
				l[r] += list(self.valid_placements(tile))
				tile.rotate()
			d[tile] = l
		return d
	def is_potentially_solvable(self):
		'''
		Return whether all tiles have at least one position where they can possibly be placed.
		If this returns False, then that means the puzzle currently cannot possibly be solved.
		If this returns True, then that means the puzzle could be solvable, but is not proven.
		'''
		avp = self.all_valid_placements()
		b = all(any(len(l)>0 for l in p) for p in avp.values())
		if not b:
			#print('a tile cannot be placed somewhere')
			return False
		s = set(p for r in avp.values() for p in r[0]+r[1]+r[2]+r[3])
		b = all(
			(x,y) in s
			for y in range(self.size)
			for x in range(self.size)
			if self.board[y][x] is None
		)
		if not b:
			#print('a position cannot hold a tile')
			return False
		return True
	def attempt_autosolve(self):
		'''
		An implementation of a humanlike approach to solving the puzzle: work along the edges, and avoid leaving holes in the grid where pieces would be difficult to place.
		'''
		good_spots = []
		for y in range(self.size):
			if self.is_empty(0, y):
				good_spots.append((0, y))
				break
			for x in range(self.size):
				if y>0 and self.is_empty(x, y-1):
					break
				if self.is_empty(x, y):
					good_spots.append((x, y))
					break
		for i in range(4):
			for i,tile in enumerate(self.tiles):
				for x,y in self.valid_placements(tile):
					if (x,y) in good_spots:
						self.place_tile(i, x, y)
						return True
			self.rotate_tiles()
		return False
	def find_solutions(self, num_solutions = 1):
		'''
		Return a list of dictionaries where each dictionary maps each tile to a possible location and rotation that would solve the puzzle.

		The solutions are found with uniform distribution over all possible solutions with the current set of tiles.
		'''
		solutions = []
		for n in range(num_solutions):
			K = 0
			while True:
				if not self.attempt_autosolve():
					if len(self.tiles) > 0:
						self.clear_board()
						K += 1
					else:
						break
			print(f'Solution found in {K} attempts')
			solutions.append({
				self.board[y][x]: (x, y, self.board[y][x].r)
				for y in range(self.size)
				for x in range(self.size)
			})
		return solutions
	def __repr__(self):
		BLACK = '\u25a1'
		WHITE = '\u25a0'
		#FILLED = '+'
		COLORS = (BLACK, WHITE)
		aug_size = self.size*4
		s = [[' ' for x in range(aug_size)] for y in range(aug_size)]
		for y in range(self.size):
			_y = y*4
			for x in range(self.size):
				_x = x*4
				t = self.board[y][x]
				if t is None:
					continue
				s[_y][_x+1] = COLORS[t.colors[6]]
				s[_y][_x+2] = COLORS[t.colors[7]]
				s[_y+1][_x] = COLORS[t.colors[4]]
				s[_y+1][_x+3] = COLORS[t.colors[1]]
				s[_y+2][_x] = COLORS[t.colors[5]]
				s[_y+2][_x+3] = COLORS[t.colors[0]]
				s[_y+3][_x+1] = COLORS[t.colors[3]]
				s[_y+3][_x+2] = COLORS[t.colors[2]]
				s[_y+1][_x+1] = '/'#FILLED
				s[_y+2][_x+1] = '\\'#FILLED
				s[_y+1][_x+2] = '\\'#FILLED
				s[_y+2][_x+2] = '/'#FILLED
		return '\n'.join(' '.join(_s) for _s in reversed(s))



class Tile:
	def __init__(self, solution_x, solution_y):
		self.colors = [0]*8
		self.solution_x = solution_x
		self.solution_y = solution_y
		self.r = 0 # solution rotation is zero
	def rotate(self, k=1):
		k %= 4
		if k == 0:
			return
		v = 8-k*2
		self.colors = self.colors[v:] + self.colors[:v]
		self.r = (self.r + k) % 4
	def set_rotation(self, r):
		self.rotate(r-self.r)
	def unrotate(self):
		self.set_rotation(0)
	def __hash__(self):
		return hash((self.solution_x, self.solution_y))
	def __eq__(self, tile):
		if type(tile) is not Tile:
			return False
		return tile.solution_x == self.solution_x and tile.solution_y == self.solution_y
	def __repr__(self):
		return ''.join(map(str,self.colors))


