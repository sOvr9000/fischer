
import numpy as np



__all__ = ['mcts_basic_heuristic', 'mcts_advanced_heuristic', 'mcts_basic_move_quality_heuristic', 'random_setup', 'Mancala', 'MancalaMoveRecorded']

def mcts_basic_heuristic(game: 'Mancala'):
	if game.done:
		return np.inf if game.winner == 0 else -np.inf if game.winner == 1 else 0
	wp0 = game.win_proximity(0)
	wp1 = game.win_proximity(1)
	v = game.score_lead() + 10. / (wp0*wp0) - 10. / (wp1*wp1)
	if wp0 <= 12 or wp1 <= 12:
		v += int(np.sum(game.pieces[0])>np.sum(game.pieces[1]))*6-3
	return v

def mcts_advanced_heuristic(game: 'Mancala'):
	v = mcts_basic_heuristic(game)
	free_turns0 = 0
	free_turns1 = 0
	for k in range(6):
		free_turns0 += int(game.pieces[0,k]%13+k==6)
		free_turns1 += int(game.pieces[1,k]%13+k==6)
		kp = game.pieces[0,k]+k
		if kp < 6 and game.pieces[0,kp] == 0:
			v += game.pieces[1,5-kp] + 1
		if kp >= 13 and kp <= 18 and game.pieces[0,kp-13]==0:
			v += game.pieces[1,18-kp] + 3
		kp = game.pieces[1,k]+k
		if kp < 6 and game.pieces[1,kp] == 0:
			v -= game.pieces[0,5-kp] + 1
		if kp >= 13 and kp <= 18 and game.pieces[1,kp-13]==0:
			v -= game.pieces[0,18-kp] + 3
	v += (free_turns0 * free_turns0 - free_turns1 * free_turns1) * 2
	return v

def mcts_basic_move_quality_heuristic(game: 'Mancala', n, p=None): # not the same as raw evaluation of the move; just score it on how likely it is to be good move, like if it capture pieces or earns a free move, etc.
	if type(n) is tuple:
		n,p = n
	if p is None:
		p = game.turn
	v = game.pieces[p,n]
	vn = v+n
	if n+v%13 == 6:
		return 300
	if vn < 6 and game.pieces[p,vn] == 0:
		return 200 + game.pieces[1-p,5-vn]
	if vn >= 13 and vn <= 18 and game.pieces[p,vn-13] == 0:
		return 200 + game.pieces[1-p,18-vn]
	if game.pieces[1-p,5-n] == 0:
		return 100 + game.pieces[p,n]
	if vn >= 6:
		return (vn-6)//13+1
	return 0

def random_setup(num_pieces=36):
	setup = [abs(np.random.normal()) for _ in range(6)]
	s = sum(setup)
	setup = [int(v * num_pieces / s + 0.5) for v in setup]
	while sum(setup) > num_pieces:
		i = np.random.randint(6)
		if setup[i] > 0:
			setup[i] -= 1
	return setup



class Mancala:
	def __init__(self, setup: bool = True):
		if setup:
			self.setup([6] * 6, turn=0) # Standard Mancala setup
	def setup(self, pieces: list[int], turn=0):
		'''
		Define the initial position of the board.
		'''
		assert len(pieces) == 6, 'Wrong number of slots.'
		assert all(isinstance(v,(int,np.int_)) for v in pieces), 'Some piece counts are not integers.'
		assert all(v >= 0 for v in pieces), 'All pieces must be non-negative.'
		assert turn == 0 or turn == 1, 'Turn can only be 0 or 1.'
		self.pieces = np.hstack((pieces, 0))
		self.pieces = np.vstack((self.pieces, self.pieces)).astype(int)
		self.turn = turn
		self.piece_sum = self.pieces.sum()
		self.done = False
	def can_move(self, n, p=None):
		if p is None:
			p = self.turn
		return n < 6 and n >= 0 and self.pieces[p,n] > 0
	def move(self, n, p=None):
		'''
		From the perspective of player `p`, move pieces from slot `n`.  For example, if `p=1` and `n=1`, then the slot at position (in the `Mancala.pieces` array)
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



class MancalaMoveRecorded(Mancala):
	def __init__(self, *args, setup: bool = False, **kwargs):
		super().__init__(*args, setup=setup, **kwargs)
		self.move_history = []
		self.initial_setup: tuple[list[int], int] = None
	def move(self, n, p=None):
		if p is None:
			if isinstance(n, tuple):
				n, p = n
			else:
				p = self.turn
		super().move(n, p=p)
		self.move_history.append((n, p))
	def setup(self, pieces: list[int], turn: int = 0):
		super().setup(pieces, turn=turn)
		self.initial_setup = pieces, turn
		self.move_history.clear()



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
