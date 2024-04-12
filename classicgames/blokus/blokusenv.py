

from blokus import Blokus
from pygenv import GridEnv, KEYCODE



class BlokusEnv(GridEnv):
	TILE_COLORS = [
		(0,0,0),
		(0,255,0),
		(0,0,255),
		(255,0,0),
		(255,255,0),
	]
	def __init__(self):
		super().__init__(screen_size = (990, 990))
		self.set_default_tile_color(self.TILE_COLORS[0])
		self.set_dimensions(22, 22)
		self.set_scale(45)
		self.set_scrollable(False)
		self.set_pannable(False)
		self.set_grid_lines(False)
		self.center_camera()
		self.game = Blokus()
		self.game.reset()
		self.max_num_moves = -1
	def on_rendering_tile(self, x, y):
		self.set_tile_color(x, y, self.TILE_COLORS[self.game.board[y,x]+1])
	def update(self):
		# pass
		if self.frame % 30 == 0:
			if self.game.done:
				self.game.reset()
			else:
				self.game.move(*self.game.random_move())
				self.max_num_moves = max(self.max_num_moves, len(self.game.valid_moves()))
				print(self.max_num_moves)


if __name__ == '__main__':
	env = BlokusEnv()
	env.run_loop()
