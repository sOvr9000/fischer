
import numpy as np

from snakeplayer import SnakePlayer


class SnakeGame:
	def __init__(self, size = 11):
		self.size = size
		self.food = {}
		self.snake = SnakePlayer(self)
	def reset(self):
		x,y = np.random.randint(self.size), np.random.randint(self.size)
		d = np.random.randint(4)
		if x > self.size // 2 and d == 0:
			d = 2
		elif y > self.size // 2 and d == 1:
			d = 1
		elif x < self.size // 2 and d == 2:
			d = 0
		elif y < self.size // 2 and d == 3:
			d = 3
		self.snake.respawn(
			x,
			y,
			length = 4,
			direction = d
		)
		self.food.clear()
		for _ in range(4):
			self.spawn_food()
		self.done = False
	def step(self):
		self.snake.move()
		if self.snake.dead:
			self.done = True
	def turn(self, d):
		self.snake.turn(d)
	def spawn_food(self, food_type = 0, tier = 0):
		fx = np.random.randint(self.size)
		fy = np.random.randint(self.size)
		K = 0
		while (fx,fy) in self.food or any(fx==bx and fy==by for bx,by in self.snake.body_locations()):
			fx = np.random.randint(self.size)
			fy = np.random.randint(self.size)
			K += 1
			if K >= 10:
				break
		self.food[(fx, fy)] = (food_type, tier)
	def respawn_food(self, x, y, food_type = 0, tier = 0):
		del self.food[(x,y)]
		self.spawn_food(food_type = food_type, tier = tier)
	def get_food_at(self, x, y):
		if (x,y) in self.food:
			return self.food[(x,y)]
		return None
	def get_foods(self):
		for (fx,fy),(food_type,tier) in self.food.items():
			yield fx, fy, food_type, tier
	def __repr__(self):
		grid = [[' ']*self.size for _ in range(self.size)]
		for bx,by in self.snake.body_locations():
			grid[by][bx] = '\u25a0'
		for fx,fy,food_type,tier in self.get_foods():
			grid[fy][fx] = str(tier)
		hx,hy = self.snake.head_location()
		if hx >= 0 and hx < self.size and hy >= 0 and hy < self.size:
			grid[hy][hx] = '\u25a1'
		return '-' * (self.size*2+1) + '\n' + '\n'.join('|' + ' '.join(g) + '|' for g in grid[::-1]) + '\n' + '-' * (self.size*2+1)



if __name__ == '__main__':
	game = SnakeGame()
	game.reset()
	while not game.done:
		print(game)
		inp = input('Move: ')
		d = -1 if inp == ' ' else 1 if inp == '  ' else 0
		game.turn(d)
		game.step()
	print(game)




