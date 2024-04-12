
import numpy as np

from connectfour import ConnectFour
from alphabeta import alphabeta_moves




game = ConnectFour(rows=6, columns=7)
game.reset()


while not game.done:
	print(game)
	if game.p1_to_move:
		moves = alphabeta_moves(game, 6)
		a = max(moves, key=lambda t: t[1])[0]
		print(moves)
	else:
		# a = game.random_move()
		moves = alphabeta_moves(game, 6)
		a = min(moves, key=lambda t: t[1])[0]
		print(moves)
	game.move(a)

print(game)


