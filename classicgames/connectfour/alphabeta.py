

import numpy as np


def basic_heuristic(node):
	if node.done:
		return np.inf if node.winner == 1 else -np.inf if node.winner == -1 else 0
	h = 0
	d = 0
	sd = 0
	for i in range(node.rows):
		for j in range(node.columns-1):
			d += int(node.has_diagonal1(i,j)) + int(node.has_diagonal2(i,j))
			sd += int(node.has_spaced_diagonal1(i,j)) + int(node.has_spaced_diagonal2(i,j))
			h += int(node.has_horizontal(i,j)) + int(node.has_vertical(i,j))
	return h + d*2 + sd*3

def alphabeta(node, depth):
	return _alphabeta(node, depth, node.p1_to_move, -np.inf, np.inf)

def _alphabeta(node, depth, maximizing, alpha, beta):
	if depth <= 0 or node.done:
		return basic_heuristic(node)
	if maximizing:
		value = -np.inf
		for n in node.valid_moves():
			node._push_move(n)
			value = max(value, _alphabeta(node, depth-1, False, alpha, beta))
			node._pop_move(n)
			alpha = max(alpha, value)
			if value >= beta:
				break
	else:
		value = np.inf
		for n in node.valid_moves():
			node._push_move(n)
			value = min(value, _alphabeta(node, depth-1, True, alpha, beta))
			node._pop_move(n)
			beta = min(beta, value)
			if value <= alpha:
				break
	return value

def alphabeta_moves(node, depth):
	moves = []
	for n in node.valid_moves():
		node._push_move(n)
		moves.append((n, alphabeta(node, depth-1)))
		node._pop_move(n)
	return moves
