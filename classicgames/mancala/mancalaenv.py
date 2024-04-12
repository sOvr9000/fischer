
import numpy as np
from .mancala import Mancala, mcts_basic_heuristic as mcts_bh, mcts_basic_move_quality_heuristic as mcts_bmqh
from pygenv import PygEnv, KEYCODES



__all__ = ['MancalaEnv']

class MancalaEnv(PygEnv):
	def __init__(self):
		super().__init__(screen_size = (500, 200))
		self.set_pannable(False)
		self.set_heuristics(None, None, 0)
		self.game = Mancala()
		self.game.setup([4,4,4,4,4,4], 0)
		self.set_bg_color((128,64,0))
		self.eval()
	def set_heuristics(self, heuristic = None, move_quality_heuristic = None, depth = 0):
		self.heuristic = heuristic
		self.move_quality_heuristic = move_quality_heuristic
		self.mcts_depth = depth
	def get_np(self):
		p = 1 - self.mouse_pos_y // 30
		n = self.mouse_pos_x // 30 - 1
		if p == 0:
			return n, p
		return 5-n,p
	def is_valid_np(self, n, p):
		return n >= 0 and n < 6 and (p == 0 or p == 1)
	def left_mouse_button_pressed(self):
		n,p = self.get_np()
		if self.is_valid_np(n,p):
			if p == self.game.turn and self.game.can_move(n,p):
				self.game.move(n, p)
				self.eval()
	def key_pressed(self, key):
		if key == KEYCODES.r:
			self.game.setup(Mancala.random_setup(32))
			self.eval()
	def eval(self):
		if self.heuristic is not None and self.move_quality_heuristic is not None:
			self.eval_values = self.game.mcts(self.mcts_depth, heuristic = self.heuristic, move_quality_heuristic = self.move_quality_heuristic)
			norm = np.exp(np.array(self.eval_values) / np.sum(self.eval_values))
			self.eval_softmax = norm / norm.sum()
		else:
			self.eval_values = None
	def render(self):
		i = 0
		for p in (0,1):
			for n in (0,1,2,3,4,5,6):
				self.slot_labels[i].set_text(str(self.game.pieces[p,n]))
				i += 1
		self.draw_screen_circle((50,120,200) if self.game.turn else (200,120,50), 30*8, 15+30*(1-self.game.turn), 12, 3)
		if self.eval_values is None:
			for i in range(6):
				self.eval_labels[i].set_text('N/A')
				self.eval_labels[i].set_bg_rect(c = (128,128,128))
		else:
			for i in range(6):
				_i = 5-i if self.game.turn else i
				self.eval_labels[i].set_text(f'{self.eval_values[_i]:.1f}')
				if type(self.eval_softmax[_i]) is float:
					self.eval_labels[i].set_bg_rect(c = self.hsv_to_rgb(0.333333 * (1-self.eval_softmax[_i]), 1, 0.5))
	def on_mouse_wheel(self, v):
		n,p = self.get_np()
		if self.is_valid_np(n,p):
			if v < 0:
				if self.game.pieces[p,n] > 0:
					self.game.set_pieces(n,p,self.game.pieces[p,n]-1)
					self.eval()
			else:
				self.game.set_pieces(n,p,self.game.pieces[p,n]+1)
				self.eval()




if __name__ == '__main__':
	env = MancalaEnv()
	env.set_heuristics(mcts_bh, mcts_bmqh, 5)
	env.eval()
	env.run_loop()



