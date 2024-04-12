

import numpy as np



class SnakePlayer:
	DIRECTION_OFFSETS = [(1,0),(0,1),(-1,0),(0,-1)]
	def __init__(self, game):
		self.game = game
	def respawn(self, x, y, length = 4, direction = 0):
		self.length = length
		self.lifetime = -self.length
		self.body = np.array([[y, x] for _ in range(self.length)])
		self.direction = direction
		self.dead = False
	def has_crashed(self):
		hy,hx = self.body[0]
		if hx < 0 or hx >= self.game.size or hy < 0 or hy >= self.game.size:
			return True
		if self.lifetime >= 0:
			for bx,by in self.body_locations():
				if bx == hx and by == hy:
					return True
		return False
	def move(self):
		hy,hx = self.body[0]
		dx,dy = self.DIRECTION_OFFSETS[self.direction]
		self.body = np.roll(self.body, 1, axis=0)
		self.body[0] = hy+dy,hx+dx
		if self.has_crashed():
			self.dead = True
		f = self.game.get_food_at(hx, hy)
		if f is not None:
			self.eat_food(*f)
			self.game.respawn_food(hx, hy)
		self.lifetime += 1
	def turn(self, d): # d = 0 means no turn, otherwise turn right with d = -1 and left with d = 1
		self.direction = (self.direction + d) % 4
	def body_locations(self):
		for y,x in self.body[1:]:
			yield x,y
	def head_location(self):
		y,x = self.body[0]
		return x,y
	def eat_food(self, food_type, tier):
		if food_type == 0:
			self.grow(tier+1)
	def grow(self, amount):
		by,bx = self.body[1]
		self.body = np.vstack((self.body, [[by,bx] for _ in range(amount)]))

