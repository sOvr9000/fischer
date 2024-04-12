

import numpy as np
from pygenv import PygEnv, KEYCODE
from snakegame import SnakeGame




class SnakeEnv(PygEnv):
	SCALE = 32
	def __init__(self):
		self.game = SnakeGame(11)
		super().__init__(screen_size = (self.SCALE*self.game.size, self.SCALE*self.game.size))
		self.control_mode = 'absolute'
		#self.control_mode = 'relative'
		self.control_mode_absolute_turns = np.array([
			[0, -1, 0, 1],
			[1, 0, -1, 0],
			[0, 1, 0, -1],
			[-1, 0, 1, 0]
		])
	def reset(self):
		self.game.reset()
		self.turn = 0
		self.set_bg_color((0,0,0))
	def render(self):
		for bx,by in self.game.snake.body_locations():
			self.draw_screen_rect((0,255,0),self.SCALE*bx,(self.game.size-1-by)*self.SCALE,self.SCALE,self.SCALE,width=0)
		for fx,fy,food_type,tier in self.game.get_foods():
			self.draw_screen_rect((255,0,0),self.SCALE*fx,(self.game.size-1-fy)*self.SCALE,self.SCALE,self.SCALE,width=0)
		hx,hy = self.game.snake.head_location()
		self.draw_screen_rect((255,255,0),self.SCALE*hx,(self.game.size-1-hy)*self.SCALE,self.SCALE,self.SCALE,width=0)
	def update(self):
		if self.frame % 7 == 0 and not self.game.done:
			self.game.turn(self.turn)
			self.game.step()
			self.turn = 0
			if self.game.done:
				self.set_bg_color((25,0,0))
	def key_pressed(self, key):
		if self.control_mode == 'absolute':
			if key == KEYCODE.d:
				self.turn = self.control_mode_absolute_turns[0, self.game.snake.direction]
			elif key == KEYCODE.w:
				self.turn = self.control_mode_absolute_turns[1, self.game.snake.direction]
			elif key == KEYCODE.a:
				self.turn = self.control_mode_absolute_turns[2, self.game.snake.direction]
			elif key == KEYCODE.s:
				self.turn = self.control_mode_absolute_turns[3, self.game.snake.direction]
		elif self.control_mode == 'relative':
			if key == KEYCODE.d:
				self.turn = -1
			elif key == KEYCODE.a:
				self.turn = 1
			elif key == KEYCODE.w:
				self.turn = 0
		if key == KEYCODE.r:
			self.reset()



if __name__ == '__main__':
	env = SnakeEnv()
	env.reset()
	env.run_loop()
