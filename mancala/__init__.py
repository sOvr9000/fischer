
import os
import numpy as np
from typing import Iterable



NON_HOME_SLOT_INDICES = [n for n in range(6)] + [n for n in range(7,13)]



class Mancala:
	def __init__(self):
		self.random_player1_move_decider = lambda g: np.random.choice(g.get_valid_player1_move_space())
		self.decide_player1_move = self.random_player1_move_decider
		self.random_player2_move_decider = lambda g: np.random.choice(g.get_valid_player2_move_space())
		self.decide_player2_move = self.random_player2_move_decider
		self.move_stack = [] # contains relative indices instead of absolute
		self.start_player = None # set to True for first player, False for second
		self.reset()
	def set_player1_move_decider(self, func):
		self.decide_player1_move = func
	def set_player2_move_decider(self, func):
		self.decide_player2_move = func
	def reset(self, slots=None, turn=None):
		if slots is None:
			slots = [6]*6
		self.slots = (slots+[0])*2
		self.score_threshold = sum(self.slots) // 2
		if turn is None:
			self.start_player = np.random.random() < 0.5
		else:
			self.start_player = turn
		self.even_turn = self.start_player

	def get_state(self, from_other_side: bool = False) -> np.ndarray[int]:
		arr = np.zeros((14,7))
		for i,v in enumerate(self.slots):
			for j,d in enumerate('{:07b}'.format(v)):
				arr[i,j] = int(d)
		if from_other_side == self.even_turn:
			return np.roll(arr * 2 - 1, 7, axis=0)
		return arr*2-1
	
	def get_state_non_binary(self, from_other_side: bool = False) -> np.ndarray[int]:
		state = self.slots.copy()
		if from_other_side == self.even_turn:
			state = np.roll(state, 7)
		return state
	
	def is_action_valid(self, action: int) -> bool:
		if action < 0 or action >= 6:
			return False
		if self.even_turn:
			return self.slots[action] > 0
		return self.slots[action+7] > 0

	def step(self, action: int):
		# index action is relative (always from current player's perspective)
		# return whether the game is finished
		if not self.is_action_valid(action):
			print('[WARNING] Mancala.step() was passed an invalid action! Using random action.')
			return self.move(np.random.choice(self.get_valid_player1_move_space() if self.even_turn else self.get_valid_player2_move_space()))
		if not self.even_turn:
			action += 7
		return self.move(action)

	def move(self, i):
		# index i is absolute (always from first player's perspective)
		# return whether the game is finished
		p = self.slots[i]
		if p == 0: return False
		self.move_stack.append(i if self.even_turn else i-7)
		self.slots[i] = 0
		ohs = 13 if self.even_turn else 6
		k = i
		for _ in range(p):
			k += 1
			if k == ohs:
				k += 1
			if k == 14:
				k = 0
			self.slots[k] += 1
		free_turn = False
		if k == (6 if self.even_turn else 13):
			free_turn = True
		elif self.slots[k] == 1 and ((k < 6) if self.even_turn else (k >= 7)):
			self.slots[6 if self.even_turn else 13] += self.slots[k] + self.slots[12-k]
			self.slots[k] = 0
			self.slots[12-k] = 0
		if sum(self.slots[:6]) == 0:
			self.slots = [0] * 6 + [self.slots[6]] + [0] * 6 + [self.slots[13] + sum(self.slots[7:13])]
		elif sum(self.slots[7:13]) == 0:
			self.slots = [0] * 6 + [self.slots[6] + sum(self.slots[:6])] + [0] * 6 + [self.slots[13]]
		self.player1_won = self.slots[6] > self.score_threshold
		self.player2_won = self.slots[13] > self.score_threshold
		if not free_turn:
			self.even_turn = not self.even_turn
		return self.player1_won or self.player2_won or (self.slots[6] == self.score_threshold and self.slots[13] == self.score_threshold)
	
	def actions_for_free_turn(self) -> Iterable[int]:
		if self.even_turn:
			inc = 0
			home = 6
		else:
			inc = 7
			home = 13
		for move in range(6):
			i = inc + move
			if (self.slots[i] + move) % 14 == home:
				yield move
	
	def actions_for_capture(self) -> Iterable[int]:
		if self.even_turn:
			inc = 0
		else:
			inc = 7
		for move in range(6):
			i = inc + move
			if self.slots[i] == 0:
				continue
			if self.slots[i] >= 14:
				continue
			if self.slots[i] == 13:
				yield move
				continue
			j = self.slots[i] + move
			if j > 13 and j - inc < 19 and self.slots[j - 13] == 0:
				yield move
				continue
			if j >= 13:
				continue
			if j - inc > 0 and self.slots[j] == 0:
				yield move
	
	def actions_for_capture_threat(self) -> Iterable[int]:
		'''
		Actions that open slots to create a threat of a capture, or (TODO) create a slot with 13 pieces in it which also indicates a capture threat.
		'''
		if self.even_turn:
			inc = 0
		else:
			inc = 7
		for move in range(6):
			i = inc + move
			if self.slots[i] == 0:
				continue
			for j in range(inc, i):
				if self.slots[j] == i - j:
					yield move
					break

	def __repr__(self):
		return '--------------------------------------\n|    |  ' + '  '.join([f'{s: >2d}' for s in self.slots[12:6:-1]]) + '  |    |\n| ' + f'{self.slots[13]: >2d}' + ' | ------------------------ | ' + f'{self.slots[6]: >2d}' + ' |\n|    |  ' + '  '.join([f'{s: >2d}' for s in self.slots[:6]]) + '  |    |\n--------------------------------------'

	def get_player1_score(self):
		return self.slots[6]
	def get_player2_score(self):
		return self.slots[13]
	def get_player1_lead(self):
		return self.get_player1_score() - self.get_player2_score()
	def get_player2_lead(self):
		return self.get_player2_score() - self.get_player1_score()
	def get_winner(self) -> int:
		# returns 1 if player 1 won, 0 if player 2 won, or -1 if drawn
		l = self.get_player1_lead()
		return 1 if l > 0 else 0 if l < 0 else -1

	def fix_player1_move(self, m):
		return np.random.choice(self.get_valid_player1_move_space()) if self.slots[m] == 0 else m
	def fix_player2_move(self, m):
		return np.random.choice(self.get_valid_player2_move_space()) if self.slots[m+7] == 0 else m
	def get_valid_player1_move_space(self):
		return [n for n in range(6) if self.slots[n] != 0]
	def get_valid_player2_move_space(self):
		return [n for n in range(7,13) if self.slots[n] != 0]
	def get_valid_player1_move_mask(self):
		return [self.slots[i] != 0 for i in range(6)]
	def get_valid_player2_move_mask(self):
		return [self.slots[i] != 0 for i in range(7,13)]
	def get_valid_action_space(self):
		if self.even_turn:
			return self.get_valid_player1_move_space()
		else:
			return [n - 7 for n in self.get_valid_player2_move_space()]
	def get_valid_action_mask(self):
		if self.even_turn:
			return self.get_valid_player1_move_mask()
		else:
			return self.get_valid_player2_move_mask()
	
	def to_bytearray(self):
		num_bytes = int(8*len(self.move_stack)/3.+.5) # 8 bits per byte, one move is 3 bits
		bit_sequence = []
		for i in self.move_stack:
			for b in f'{i:03b}':
				bit_sequence.append(int(b))
		bit_sequence += [0] * ((-len(bit_sequence)) % 8)
		bits = np.reshape(bit_sequence, (-1,8))
		ba = np.dot(bits, [128,64,32,16,8,4,2,1])
		ba = bytearray([num_bytes] + ba.tolist())
		return ba
	
	@classmethod
	def from_bytearray(cls, ba):
		num_bytes = ba[0]&255
		bits = [
			bit
			for b in ba[1:]
			for bit in (
				b&0b10000000!=0,
				b&0b01000000!=0,
				b&0b00100000!=0,
				b&0b00010000!=0,
				b&0b00001000!=0,
				b&0b00000100!=0,
				b&0b00000010!=0,
				b&0b00000001!=0,
			)
		]
		print(num_bytes)
		print(len(bits))
	
	def save(self, fpath):
		with open(fpath, 'ab') as f:
			f.write(self.to_bytearray())

def load_game(fpath):
	with open(fpath, 'rb') as f:
		ba = f.read()
		return Mancala.from_bytearray(ba)
