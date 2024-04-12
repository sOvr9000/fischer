import numpy as np

_minimax_mem = {}


def minimax(node, depth, o_even_turn = None, even_turn = True, use_mem = True):
	if o_even_turn is None:
		o_even_turn = node.is_even_turn()
	if use_mem:
		_minimax_mem.clear()
	return _minimax(node, depth, o_even_turn, even_turn, use_mem, -np.inf, np.inf)


# def minimax(node, depth, even_turn = None, use_mem = True):
# 	if even_turn is None:
# 		even_turn = node.is_even_turn()
# 	if use_mem:
# 		_minimax_mem.clear()
# 	return _minimax(node, depth, even_turn, use_mem, -np.inf, np.inf)

def _minimax(node, depth, o_even_turn, even_turn, use_mem, alpha, beta):
	if use_mem and node.__hash__() in _minimax_mem:
		return _minimax_mem[node.__hash__()]
	if depth == 0 or node.is_game_over():
		return node.heuristic() * (int(o_even_turn) * 2 - 1)
	if even_turn:
		value = -np.inf
		for m in node.get_available_moves():
			child_node = node.test_move(m)
			value = max(value, _minimax(child_node, depth - 1, o_even_turn, False, use_mem, alpha, beta))
			alpha = max(alpha, value)
			if alpha >= beta:
				break
	else:
		value = np.inf
		for m in node.get_available_moves():
			child_node = node.test_move(m)
			value = min(value, _minimax(child_node, depth - 1, o_even_turn, True, use_mem, alpha, beta))
			beta = min(beta, value)
			if alpha >= beta:
				break
	if use_mem:
		_minimax_mem[node.__hash__()] = value
	return value

# def _minimax(node, depth, even_turn, use_mem, alpha, beta):
# 	if use_mem and node.__hash__() in _minimax_mem:
# 		return _minimax_mem[node.__hash__()]
# 	if depth == 0 or node.is_game_over():
# 		return node.heuristic() * (int(node.is_even_turn()) * 2 - 1)
# 	if node.is_even_turn() == even_turn:
# 		value = -np.inf
# 		for m in node.get_available_moves():
# 			child_node = node.test_move(m)
# 			value = max(value, _minimax(child_node, depth - 1, even_turn, use_mem, alpha, beta))
# 			alpha = max(alpha, value)
# 			if alpha >= beta:
# 				break
# 	else:
# 		value = np.inf
# 		for m in node.get_available_moves():
# 			child_node = node.test_move(m)
# 			value = min(value, _minimax(child_node, depth - 1, even_turn, use_mem, alpha, beta))
# 			beta = min(beta, value)
# 			if alpha >= beta:
# 				break
# 	if use_mem:
# 		_minimax_mem[node.__hash__()] = value
# 	return value



class Game2P:
	'''
	A general model of a game which involves two players.

	The following functions can be overridden: on_reset(), copy(), get_available_moves(), process_move(m), is_game_over(), on_play_step(), render_as_text(), get_winner(), heuristic(), pre_simulate_step(), post_simulate_step(), pre_simulation(), post_simulation().

	Override __hash__() to allow faster tree search in minimax(depth).
	'''
	def __init__(self):
		pass

	def reset(self):
		self.even_turn = True
		self.on_reset()
	
	def is_even_turn(self):
		return self.even_turn

	def on_reset(self):
		'''
		Reset the game state.
		'''
		pass

	def _copy(self):
		game = self.copy()
		game.even_turn = self.even_turn
		return game

	def copy(self):
		'''
		Return a copy of the game, as a separate instance.
		'''
		raise NotImplementedError()

	def get_available_moves(self):
		'''
		Return a list of possible moves in the current state of the game.
		'''
		raise NotImplementedError()

	def do_move(self, m):
		self.process_move(m)
		self.even_turn = not self.even_turn # default to alternating turns
	
	def process_move(self, m):
		'''
		Process a move.
		'''
		raise NotImplementedError()

	def test_move(self, m):
		n = self._copy()
		n.do_move(m)
		return n

	def select_random_move(self):
		av = self.get_available_moves()
		return av[np.random.randint(len(av))]

	def is_game_over(self):
		raise NotImplementedError()

	def play(self, show_available_moves = True, p1_move_selection = ('minimax', 3, False), p2_move_selection = 'random'):
		if type(p1_move_selection) is str:
			p1_move_selection = (p1_move_selection,)
		if type(p2_move_selection) is str:
			p2_move_selection = (p2_move_selection,)
		self.reset()
		while not self.is_game_over():
			self.render_as_text()
			if show_available_moves:
				print('Available moves:\n{}'.format('\n'.join([
					' '.join(str(q) for q in _m)
					for _m in self.get_available_moves()
				])))
			inp = input('Player {}, play your move!\n'.format(1 if self.even_turn else 2))
			if inp == '':
				move_selection = p1_move_selection if self.even_turn else p2_move_selection
				m = self.select_random_move() if move_selection[0] == 'random' else self.minimax(depth = 3 if len(move_selection) < 2 else move_selection[1], use_mem = False if len(move_selection) < 3 else move_selection[2])
			else:
				l = inp.split(' ')
				for i in range(len(l)):
					try:
						l[i] = int(l[i])
					except:
						pass
				m = tuple(l)
			self.do_move(m)
			print('Move played: {}'.format(m))
			self.on_play_step()
			print('\n' + '=' * 96 + '\n')
		self.render_as_text()
		print('\n\nGame over!')
		w = self.get_winner()
		print('It\'s a draw!' if w == 0 else 'Player {} wins.'.format((3 - w) // 2))
		print('\n' + '=' * 96 + '\n')

	def on_play_step(self):
		pass

	def render_as_text(self):
		'''
		Render the game as text.
		'''
		raise NotImplementedError()

	def simulate(self, n = 1000, p1_move_selection = 'random', p2_move_selection = ('minimax', 3, False)):
		self.pre_simulation()
		if type(p1_move_selection) is str:
			p1_move_selection = (p1_move_selection,)
		if type(p2_move_selection) is str:
			p2_move_selection = (p2_move_selection,)
		wins = 0
		draws = 0
		for t in range(n):
			self.reset()
			print('{} / {}'.format(t + 1, n))
			while not self.is_game_over():
				self.pre_simulate_step()
				move_selection = p1_move_selection if self.even_turn else p2_move_selection
				self.do_move(self.select_random_move() if move_selection[0] == 'random' else self.minimax(depth = 3 if len(move_selection) < 2 else move_selection[1], use_mem = False if len(move_selection) < 3 else move_selection[2]))
				self.post_simulate_step()
			w = self.get_winner()
			wins += int(w == 0)
			draws += int(w == -1)
			if t % 50 == 0:
				print('Wins/Draws: {:.1f}% / {:.1f}% of {} {}'.format(wins * 100. / max(1, t), draws * 100. / max(1, t), t + 1, 'game' if t == 0 else 'games'))
		wr = float(wins) / n
		dr = float(draws) / n
		print('Wins/Draws: {:.1f}% / {:.1f}% of {} games'.format(wr * 100, dr * 100, n))
		self.post_simulation()
		return wr

	def get_winner(self):
		'''
		Return 0 (Draw), 1 (Player 1), or -1 (Player 2).
		'''
		raise NotImplementedError()

	def post_simulate_step(self):
		pass

	def pre_simulate_step(self):
		pass

	def pre_simulation(self):
		pass

	def post_simulation(self):
		pass

	def heuristic(self):
		'''
		Quickly return a best guess of how good the position is for Player 0.
		'''
		raise NotImplementedError()

	def minimax(self, depth, use_mem=True):
		m = max([
			(minimax(self.test_move(m), depth - 1, o_even_turn = self.is_even_turn(), even_turn = not self.is_even_turn(), use_mem=use_mem), m)
			for m in self.get_available_moves()
		])
		print('Minimax value for Player {}: {}'.format(2-int(self.even_turn), m[0]))
		return m[1]
	
	def __hash__(self):
		raise NotImplementedError()


