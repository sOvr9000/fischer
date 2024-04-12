
import os

import numpy as np
from tetris import Tetris

game = Tetris((10, 20))

def print_state(state):
	os.system('cls')
	s = ''
	for y in range(game.height-1,-1,-1):
		s += ""
		for x in range(game.width):
			v = state[y*game.width+x]
			s += '\u25a0' if v == -1 else '\u25a0' if v == 1 else '\u25a1'
			s += ' '
		s += '\n'
	print(s)
	print('Current score: {}'.format(game.score))

while True:
	state = game.reset()
	print_state(state)
	while True:
		inp = input('0. stay put\n1. left\n2. right\n3. rotate\nEnter your choice: ')
		if inp == '':
			inp = '0'
		action = int(inp)
		state, reward, done, info = game.step(action)
		print_state(state)
		print('Reward: {}'.format(reward))
		if done:
			break

	print('Game over!')
	inp = input('Play again? (y/n)\n')
	if inp != 'y':
		break

print('Goodbye!')
