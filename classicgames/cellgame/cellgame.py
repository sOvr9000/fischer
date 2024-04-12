

import numpy as np



class CellGame:
	def __init__(self, num_players = 2):
		self.num_players = num_players
		self.players = [Player() for _ in range(num_players)]
	def step(self):
		for cell in self.cells:
			cell.step()


class Player:
	def __init__(self):
		self.owned_cells = []

class Cell:
	def __init__(self, x, y, initial_mass, owner):
		self.x = x
		self.y = y
		self.mass = initial_mass
		self.owner = owner
		self.incoming_mass = []
	def step(self):
		for i in range(len(self.incoming_mass)-1,-1,-1):
			self.incoming_mass[i][0] -= 1
			if self.incoming_mass[i][0] <= 0:
				self.last_received_mass_from = self.incoming_mass[i][1]
				self.add_mass(self.incoming_mass[i][2] * (1-2*int(self.last_received_mass_from == self.owner)))
			del self.incoming_mass[i][0]
	def add_mass(self, amount):
		self.mass += amount
		if self.mass <= 0:
			self.owner = self.last_received_mass_from
			self.mass = -self.mass


