
from pygenv import PygEnv
from .tetris import Tetris




class TetrisEnv(PygEnv):
	def __init__(self, game:Tetris):
		self.game = game
		
	def render(self):
		pass
