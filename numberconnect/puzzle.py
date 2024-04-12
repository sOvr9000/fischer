




class NumberConnect:
	def __init__(self, array):
		self.array = array
		self.score = 0
	def can_remove(self, x1, y1, x2, y2):
		if x2 < x1:
			x1, x2 = x2, x1
		if y2 < y1:
			y1, y2 = y2, y1
		return \
			(x1 == x2 or y1 == y2) and \
			(x2 > x1 or y2 > y1) and \
			x1 >= 0 and x1 < self.array.shape[1] and \
			x2 >= 0 and x2 < self.array.shape[1] and \
			y1 >= 0 and y1 < self.array.shape[0] and \
			y2 >= 0 and y2 < self.array.shape[0] and \
			self.array[y1,x1] >= 0 and self.array[y2,x2] >= 0 and \
			(
				all(
					self.array[y1,x] == -1
					for x in range(x1+1,x2)
				)
				if y1 == y2 else
				all(
					self.array[y,x1] == -1
					for y in range(y1+1,y2)
				)
			)
	def remove(self, x1, y1, x2, y2):
		if not self.can_remove(x1, y1, x2, y2):
			raise Exception(f'Cannot remove {x1, y1, x2, y2}')
		self.score += self.array[y1,x1]*10 + self.array[y2,x2]
		self.array[y1,x1] = -1
		self.array[y2,x2] = -1
	def valid_actions(self):
		for y in range(self.array.shape[0]):
			for x1 in range(self.array.shape[1]-1):
				for x2 in range(x1+1,self.array.shape[1]):
					if self.can_remove(x1, y, x2, y):
						if self.array[y,x1] > self.array[y,x2]:
							yield x1, y, x2, y
						else:
							yield x2, y, x1, y
		for x in range(self.array.shape[1]):
			for y1 in range(self.array.shape[0]-1):
				for y2 in range(y1+1,self.array.shape[0]):
					if self.can_remove(x, y1, x, y2):
						if self.array[y1,x] > self.array[y2,x]:
							yield x, y1, x, y2
						else:
							yield x, y2, x, y1
	def __repr__(self):
		return '\n'.join(
			' '.join(
				str(v) if v >= 0 else ' '
				for v in r
			)
			for r in self.array
		)


