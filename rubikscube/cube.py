
import numpy as np
from twophase.solver import solve, solveto
from .cubelib import inverted_moves_list, str_move_map, orient_symmetry, str_to_moves

OPPOSITE_SIDES = [1, 0, 3, 2, 5, 4]

FLAT_STR_FACE_INDICES = [(x, y) for y in (1, 2, 3) for x in (1, 2, 3)]
FLAT_STR_INDICES = [
	(4-y, 0, x) for x, y in FLAT_STR_FACE_INDICES
] + [
	(x, y, 4) for x, y in FLAT_STR_FACE_INDICES
] + [
	(0, y, x) for x, y in FLAT_STR_FACE_INDICES
] + [
	(y, 4, x) for x, y in FLAT_STR_FACE_INDICES
] + [
	(4-x, y, 0) for x, y in FLAT_STR_FACE_INDICES
] + [
	(4, y, 4-x) for x, y in FLAT_STR_FACE_INDICES
]
FLAT_STR_FACELETS = ['F', 'B', 'U', 'D', 'L', 'R']

CUBELET_PERMUTATION_MAP = {
	
}


class RubiksCube:
	def __init__(self):
		self.reset()

	def reset(self):
		self.colors = -np.ones((5, 5, 5), dtype=np.int8)
		self.colors[0, 1:-1, 1:-1] = 0  # front
		self.colors[4, 1:-1, 1:-1] = 1  # back
		self.colors[1:-1, 0, 1:-1] = 2  # top
		self.colors[1:-1, 4, 1:-1] = 3  # bottom
		self.colors[1:-1, 1:-1, 0] = 4  # left
		self.colors[1:-1, 1:-1, 4] = 5  # right

	def copy(self):
		cube = RubiksCube()
		cube.colors = np.copy(self.colors)
		return cube

	def rotate(self, face, times):
		assert face >= 0 and face <= 5
		assert face is not None and type(face) is int
		assert times >= 1 and times <= 3
		assert times is not None and type(times) is int
		if face == 1:  # back
			self.colors[3:, :, :] = np.rot90(
				self.colors[3:, :, :], k=times, axes=(1, 2))
		elif face == 0:  # front
			self.colors[:2, :, :] = np.rot90(
				self.colors[:2, :, :], k=times, axes=(1, 2))
		elif face == 3:  # bottom
			self.colors[:, 3:, :] = np.rot90(
				self.colors[:, 3:, :], k=times, axes=(0, 2))
		elif face == 2:  # top
			self.colors[:, :2, :] = np.rot90(
				self.colors[:, :2, :], k=times, axes=(0, 2))
		elif face == 5:  # right
			self.colors[:, :, 3:] = np.rot90(
				self.colors[:, :, 3:], k=times, axes=(0, 1))
		else:  # left
			self.colors[:, :, :2] = np.rot90(
				self.colors[:, :, :2], k=times, axes=(0, 1))

	def center_layout(self):
		return self.colors[[0,4,2,2,2,2],[2,2,0,4,2,2],[2,2,2,2,0,4]].tolist()

	def orient(self, face, times):
		self.colors = np.rot90(self.colors, k=times, axes=(1, 2) if face < 2 else (0, 2) if face < 4 else (0, 1))

	def get_orientation(self):
		return self.colors[0,2,2], self.colors[2,0,2]

	def set_orientation(self, front_face, top_face):
		# An ineffecient algorithm, but it's very easy to understand and implement.
		if self.get_orientation() == (front_face, top_face):
			return
		for _ in self.do_all_orientations():
			if self.get_orientation() == (front_face, top_face):
				return

	def do_all_orientations(self):
		# Upon each yield, the cube has a distinct orientation, preserving the arrangement of facelets on each face with respect to each other.
		# There are 24 distinct orientations.
		# Each complete iteration over this returns the cube to its original orientation prior to iteration.
		yield
		self.orient(0,1)
		yield
		self.orient(2,1)
		yield
		self.orient(0,1)
		yield
		self.orient(2,1)
		yield
		self.orient(0,1)
		yield
		self.orient(0,1)
		yield
		self.orient(2,1)
		yield
		self.orient(0,1)
		self.orient(2,1)
		yield
		self.orient(0,1)
		yield
		self.orient(0,1)
		yield
		self.orient(2,1)
		yield
		self.orient(0,1)
		self.orient(2,1)
		yield
		self.orient(0,1)
		yield
		self.orient(0,2)
		self.orient(2,2)
		yield
		self.orient(0,1)
		self.orient(2,1)
		yield
		self.orient(0,1)
		yield
		self.orient(2,1)
		self.orient(0,1)
		yield
		self.orient(0,1)
		yield
		self.orient(2,1)
		yield
		self.orient(0,1)
		self.orient(2,1)
		yield
		self.orient(0,1)
		yield
		self.orient(0,1)
		self.orient(2,1)
		yield
		self.orient(4,1)
		self.orient(0,2)
		yield
		self.orient(0,2)
		self.orient(2,2)

	def fix_orientation(self):
		# Just reorient such that it's back to "normal"
		self.set_orientation(0,2)

	def scramble(self):
		for f, t in RubiksCube.random_moves(30):
			self.rotate(f, t)

	@classmethod
	def get_scrambling_moves(cls, num_moves=20, max_time=0.5):
		num_moves = min(max(int(num_moves), 0), 20)
		c = RubiksCube()
		c.scramble()
		sol = c.solve_twophase(max_time=max_time)
		return inverted_moves_list(sol)[:num_moves]

	@classmethod
	def random_moves(cls, num_moves=30):
		for _ in range(num_moves):
			yield np.random.randint(0,6), np.random.randint(1,4)

	def move(self, m):
		'''
		m is a string, like F2, B, etc.
		'''
		assert m in str_move_map, f'The move {m} is invalid.  Possible moves: ' + ', '.join(
			str_move_map.keys())
		self.rotate(*str_move_map[m])

	def compact_array(self):
		ringx = [1, 2, 3, 1, 3, 1, 2, 3]
		ringy = [1, 1, 1, 2, 2, 3, 3, 3]
		return np.hstack((
			self.colors[0, ringx, ringy].reshape((4, 2)),
			self.colors[4, ringx, ringy].reshape((4, 2)),
			self.colors[ringx, 0, ringy].reshape((4, 2)),
			self.colors[ringx, 4, ringy].reshape((4, 2)),
			self.colors[ringx, ringy, 0].reshape((4, 2)),
			self.colors[ringx, ringy, 4].reshape((4, 2)),
		))

	def surface_stack(self):
		return np.concatenate((
			self.colors[np.newaxis, 0, 1:-1, 1:-1],
			self.colors[np.newaxis, 4, 1:-1, 1:-1],
			self.colors[np.newaxis, 1:-1, 0, 1:-1],
			self.colors[np.newaxis, 1:-1, 4, 1:-1],
			self.colors[np.newaxis, 1:-1, 1:-1, 0],
			self.colors[np.newaxis, 1:-1, 1:-1, 4]
		), axis=0)

	@classmethod
	def from_surface_stack(cls, surface_stack):
		cube = RubiksCube()
		cube.colors[0, 1:-1, 1:-1] = surface_stack[0]
		cube.colors[4, 1:-1, 1:-1] = surface_stack[1]
		cube.colors[1:-1, 0, 1:-1] = surface_stack[2]
		cube.colors[1:-1, 4, 1:-1] = surface_stack[3]
		cube.colors[1:-1, 1:-1, 0] = surface_stack[4]
		cube.colors[1:-1, 1:-1, 4] = surface_stack[5]
		return cube

	def is_solved(self):
		return (
			(self.colors[0, 1:-1, 1:-1] == 0) *  # front
			(self.colors[4, 1:-1, 1:-1] == 1) *  # back
			(self.colors[1:-1, 0, 1:-1] == 2) *  # top
			(self.colors[1:-1, 4, 1:-1] == 3) *  # bottom
			(self.colors[1:-1, 1:-1, 0] == 4) *  # left
			(self.colors[1:-1, 1:-1, 4] == 5)  # right
		).all()

	def to_definition_string(self):
		# Facelets in order of faces top, right, front, bottom, left, and then back.
		# Refer to Herbert Kociemba's notation defined at https://github.com/hkociemba/RubiksCube-TwophaseSolver/blob/master/enums.py.
		return ''.join(FLAT_STR_FACELETS[self.colors[z, y, x]] for z, y, x in FLAT_STR_INDICES)

	def solve_twophase(self, target_cube:'RubiksCube'=None, target_length=20, max_time=0.5):
		o = self.get_orientation()
		if target_cube is None:
			assert o == (0,2), 'The orientation of the cube is invalid.  Call RubiksCube.fix_orientation() first.'
			self.fix_orientation()
			if max_time < 0:
				solution = solve(self.to_definition_string(), 0, -max_time)
			else:
				solution = solve(self.to_definition_string(), target_length, max_time)
			self.set_orientation(*o)
		else:
			assert o == target_cube.get_orientation(), 'The orientations of the cubes aren\'t equivalent.  Call RubiksCube.fix_orientation() or RubiksCube.set_orientation() first.'
			self.fix_orientation()
			target_cube.fix_orientation()
			if max_time < 0:
				solution = solveto(self.to_definition_string(), target_cube.to_definition_string(), 0, -max_time)
			else:
				solution = solveto(self.to_definition_string(),
							   target_cube.to_definition_string(), target_length, max_time)
			self.set_orientation(*o)
			target_cube.set_orientation(*o)
		if 'error' in solution.lower():
			raise Exception(f'Error finding solution.  Solver response: {solution}')
		return [orient_symmetry(f,t,*o) for f,t in str_to_moves(solution)]
	
	def get_cubelet_permutation(self) -> list[int]:
		'''
		Return the arrangement of the cubelets which is the current permutation of the solved arrangement of cubelets.  The solved permutation is `[0, 1, ..., 19]`.
		'''

	def __repr__(self):
		return self.to_definition_string()

	def __eq__(self, other):
		return np.array_equal(self.colors, other.colors)





if __name__ == '__main__':
	pass
