
import numpy as np



__all__ = ['random_setup', 'Mancala']

def random_setup(num_pieces=36):
	setup = [abs(np.random.normal()) for _ in range(6)]
	s = sum(setup)
	setup = [int(v * num_pieces / s + 0.5) for v in setup]
	while sum(setup) > num_pieces:
		i = np.random.randint(6)
		if setup[i] > 0:
			setup[i] -= 1
	return setup



class MancalaGeneral:
	def setup(self, pieces: np.ndarray[int], turn: int = 0, num_players: int = None):
		'''
		Define the initial position of the board.
		
		If `num_players is None`, then it is inferred from the number of rows in `pieces`.  In this case, if `pieces` is one-dimensional, then an error is thrown.
		
		If `pieces` is two-dimensional, then `num_players` is ignored.  Otherwise, `num_players` is used to determine the number of players, and each player gets the symmetrically identical `pieces` setup.
		'''
		if not isinstance(pieces, np.ndarray):
			raise TypeError('pieces must be a NumPy array of positive integers.')
		if pieces.dtype not in (int, np.int_):
			raise TypeError('pieces must be a NumPy array of positive integers.')
		if np.any(pieces <= 0):
			raise ValueError('pieces must be a NumPy array of positive integers.')
		if len(pieces.shape) == 1:
			if num_players is None:
				raise ValueError('num_players cannot be None if pieces is one-dimensional.  Please either provide a two-dimensional pieces array or define num_players as an integer.')
			if not isinstance(num_players, (int, np.int_)):
				raise TypeError('num_players must be an integer >= 2.')
			if num_players < 2:
				raise TypeError('num_players must be an integer >= 2.')
			pieces = np.repeat(np.expand_dims(pieces, axis=0), repeats=num_players, axis=0)
		if np.any(np.sum(pieces, axis=1) == 0):
			raise ValueError('')
		self.num_slots_per_player = len(pieces)
		self.pieces = np.hstack((pieces, 0))
		self.pieces = np.vstack((self.pieces, self.pieces)).astype(int)
		self.num_players = len
		self.turn = turn
		self.piece_sum = self.pieces.sum()
		self.done = False
	def can_move(self, n, p=None):
		if p is None:
			p = self.turn
		return n < 6 and n >= 0 and self.pieces[p,n] > 0
	def move(self, n, p=None):
		'''
		From the perspective of player `p`, move pieces from slot `n`.  For example, if `p=1` and `n=1`, then the slot at position (in the `MancalaGeneral.pieces` array)
		```
		O O O O O O
		O X O O O O
		```
		is moved.  If `p` is None, then `p` is the player whose turn it is in the current position.
		'''
		if type(n) is tuple and len(n) == 2:
			n,p = n
		if p is None:
			p = self.turn
		assert self.can_move(n, p), f'The move (n={n}, p={p}) in the following state is illegal:{self}'
		self.free_turn = False
		v = self.pieces[p,n]
		self.pieces[p,n] = 0
		i = p
		j = n
		for _ in range(v):
			j += 1
			if j == (7 if i == p else 6):
				j = 0
				i = 1 - i
			self.pieces[i,j] += 1
		if j == 6:
			# free turn
			self.free_turn = True
		else:
			if i == p and self.pieces[i,j] == 1:
				# capture
				self.pieces[i,6] += 1 + self.pieces[1-i,5-j]
				self.pieces[i,j] = 0
				self.pieces[1-i,5-j] = 0
		if not self.free_turn:
			self.turn = 1 - self.turn
		# check for termination
		self.check_termination()
	def set_pieces(self, n, p, v):
		ov = self.pieces[p,n]
		self.pieces[p,n] = v
		self.piece_sum += v - ov
		self.check_termination()
	def check_termination(self):
		if not self.pieces[0,:-1].any():
			self.pieces[1,6] += self.pieces[1,:-1].sum()
			self.pieces[1,:-1] = 0
			self.done = True
		elif not self.pieces[1,:-1].any():
			self.pieces[0,6] += self.pieces[0,:-1].sum()
			self.pieces[0,:-1] = 0
			self.done = True
		if 2*self.pieces[0,6] > self.piece_sum:
			self.winner = 0
			self.done = True
		elif 2*self.pieces[1,6] > self.piece_sum:
			self.winner = 1
			self.done = True
		elif self.done:
			if self.pieces[0,6] > self.pieces[1,6]:
				self.winner = 0
			elif self.pieces[0,6] < self.pieces[1,6]:
				self.winner = 1
			else:
				self.winner = None
	def get_result(self) -> float:
		if not self.done:
			return None
		if self.winner is None:
			return .5
		return 1. - self.winner
	def valid_moves(self, p=None):
		'''
		Return a generator for all possible moves (tuples of the form (n,p)) in the current position from the perspective of player p.
		
		If p is None, then p is the player whose turn it is in the current position.
		'''
		if p is None:
			p = self.turn
		for n in range(6):
			if self.can_move(n, p):
				yield n,p
	def sorted_moves_by_likely_best(self, move_quality_heuristic=mcts_basic_move_quality_heuristic): # primarily intended to increase pruning rate in alpha-beta pruning
		va = list(self.valid_moves())
		va.sort(key = lambda m: move_quality_heuristic(self, m), reverse=True)
		return va
	def random_move(self, p=None):
		'''
		Return a randomly selected valid move (a tuple of the form (n,p)) in the current position from the perspective of player p.
		
		If p is None, then p is the player whose turn it is in the current position.
		'''
		moves = list(self.valid_moves(p=p))
		return moves[np.random.randint(len(moves))]
	def home_pieces(self, p):
		return self.pieces[p,6]
	def score_lead(self):
		'''
		Return a very cheap and basic heuristic calculation of the evaluation of the position from the first player's perspective.

		This serves as a quick and dirty way to judge who's better in the current position.  It is not always accurate.
		'''
		return self.home_pieces(0) - self.home_pieces(1)
	def win_proximity(self, p):
		return self.piece_sum - 2 * self.home_pieces(p) + 1
	def argmin_move(self, values): # used when -inf makes player-1-winning moves indistinguishable from illegal moves
		m = min(v for i,v in enumerate(values) if self.can_move(i))
		for i,v in enumerate(values):
			if v == m and self.can_move(i):
				return i
		raise Exception(f'No argmin_move returned; given values {values} or maybe no moves are legal in the following position:\n{self}')
	def argmax_move(self, values): # particularly just here as a complement to Mancala.argmin_move()
		m = max(v for i,v in enumerate(values) if self.can_move(i))
		for i,v in enumerate(values):
			if v == m and self.can_move(i):
				return i
		raise Exception(f'No argmax_move returned; given values {values} or maybe no moves are legal in the following position:\n{self}')
	def __repr__(self, perspective = False):
		if perspective:
			p1 = self.turn
			p2 = 1-self.turn
		else:
			p1 = 0
			p2 = 1
		return f'\n.=:==-==-===--===--===-==-==:=.\n| {self.home_pieces(p2):2d} | ' + ' '.join(f'{v:2d}' for v in self.pieces[p2,-2::-1]) + f'      |\n|      ' + ' '.join(f'{v:2d}' for v in self.pieces[p1,:-1]) + f' | {self.home_pieces(p1):2d} |\n\'=:==-==-===--===--===-==-==:=\''
	def mcts(self, depth, p = None, heuristic=mcts_basic_heuristic, move_quality_heuristic=mcts_basic_move_quality_heuristic):
		if p is None:
			p = self.turn
		return np.array([
			self._mcts_move(n, depth, heuristic = heuristic, move_quality_heuristic = move_quality_heuristic)
			if self.can_move(n) else
			np.inf
			if p else
			-np.inf
			for n in range(6)
		])
	def _mcts_move(self, move, depth, heuristic, move_quality_heuristic):
		node = self.copy()
		node.move(move)
		return node._alphabeta(depth-1, heuristic = heuristic, move_quality_heuristic = move_quality_heuristic)
	def _alphabeta(self, depth, alpha=-np.inf, beta=np.inf, heuristic=mcts_basic_heuristic, move_quality_heuristic=mcts_basic_move_quality_heuristic):
		if depth <= 0 or self.done:
			return heuristic(self)
		if not self.turn:
			value = -np.inf
			for va in self.sorted_moves_by_likely_best(move_quality_heuristic = move_quality_heuristic):
				node = self.copy()
				node.move(va)
				value = max(value, node._alphabeta(depth - 1, alpha = alpha, beta = beta, heuristic=heuristic, move_quality_heuristic=move_quality_heuristic))
				alpha = max(alpha, value)
				if value >= beta:
					break
		else:
			value = np.inf
			for va in self.sorted_moves_by_likely_best(move_quality_heuristic = move_quality_heuristic):
				node = self.copy()
				node.move(va)
				value = min(value, node._alphabeta(depth - 1, alpha = alpha, beta = beta, heuristic=heuristic, move_quality_heuristic=move_quality_heuristic))
				beta = min(beta, value)
				if value <= alpha:
					break
		return value
	# def capture_amount(self, n, p=None):
	# 	if p is None:
	# 		p = self.turn
	# 	v = self.pieces[p,n]
	# 	if v == 13:
	# 		return self.pieces[1-p,5-n] + 2
	# 	elif v == 12:
	# 		return self.pieces[1-p,6-n] + 2
	# 	if self.pieces[n]
	def copy(self):
		m = Mancala(setup=False)
		m.pieces = self.pieces.copy()
		m.piece_sum = self.piece_sum
		m.turn = self.turn
		m.done = self.done
		return m
	def short_str(self):
		return ' '.join(str(v) for v in self.pieces.flatten()) + ' ' + str(self.turn)
	def __hash__(self):
		return hash(self.short_str())



if __name__ == '__main__':
	game = Mancala()

	game.setup(game.random_setup(60))
	print(game)

	while not game.done:
		if game.turn:
			vs = game.mcts(8, heuristic=mcts_advanced_heuristic)
		else:
			vs = game.mcts(4, heuristic=mcts_basic_heuristic)
		print((np.min if game.turn else np.max)(vs))
		m = (game.argmin_move if game.turn else game.argmax_move)(vs)
		print(m)
		game.move(m)
		print(game)
		print('Turn: ' + str(game.turn))

	if game.winner is None:
		print('Game ended in a draw!')
	else:
		print(f'The winner is Player {game.winner+1}! (player id = {game.winner})')
