
import numpy as np



def invert_times(times):
	return 4-times


def invert_move(m):
	return m[0], invert_times(m[1])


def to_one_hot_index(face, times):
	return face*3+times-1


def from_one_hot_index(index):
	return int(index//3), int(index % 3+1)


def symmetry(face, times, r0, r1, r2, r3):
	# r0 is rotation about the axis between midpoints of front and back faces (4 rotations in total)
	# r1 is rotation about the axis between midpoints of the front-right edge and back-left edge (2 rotations in total)
	# r2 is rotation about the axis between the top-front-right vertex and the bottom-back-left vertex (3 rotations in total)
	# r3 is reflection across the plane that is parallel to the front and back faces
	f, t = face, times
	for l in range(r3):
		f, t = r3_sym[(f, t)]
	for i in range(r0):
		f, t = r0_sym[(f, t)]
	for j in range(r1):
		f, t = r1_sym[(f, t)]
	for k in range(r2):
		f, t = r2_sym[(f, t)]
	return f, t


def orient_symmetry(face, times, front_face, top_face):
	# if the cube is oriented such that the front and top faces are the provided ones, then the returned move is the corresponding symmetry of (face,times)
	return symmetry(face, times, *orient_sym[(front_face, top_face)])


def sequence_symmetries(moves, include_antisymmetries=True):
	if include_antisymmetries:
		moves = sequence_symmetries(moves, include_antisymmetries=False)
		moves.extend([
			m[::-1]
			for m in moves
		])
		return moves
	return [
		tuple(
			symmetry(f, t, i, j, k, l)
			for f, t in moves
		)
		for i in range(4)
		for j in range(2)
		for k in range(3)
		for l in range(2)
	]


def moves_are_equivalent(moves1, moves2):
	from .cube import RubiksCube
	cube1 = RubiksCube()
	for f, t in moves1:
		cube1.rotate(f, t)
	cube2 = RubiksCube()
	for f, t in moves2:
		cube2.rotate(f, t)
	return np.array_equal(cube1.colors, cube2.colors)


def inverted_moves_list(moves):
	return tuple(
		invert_move(m)
		for m in reversed(moves)
	)


def str_to_moves(s):
	if ' ' in s:
		return [str_move_map[m] for m in s.split(' ') if m in str_move_map]
	return [
		str_move_map[s[i:i+2]]
		for i in range(0, len(s), 2)
		if s[i:i+2] in str_move_map
	]


def str_to_move(s):
	return str_move_map[s]


def moves_to_str(moves):
	return ' '.join(
		move_str_map[m]
		for m in moves
	)


def move_to_str(move):
	return move_str_map[move]


str_move_map = {
	'F1': (0, 3),
	'F': (0, 3),
	'F2': (0, 2),
	'F3': (0, 1),
	'F\'': (0, 1),
	'B1': (1, 1),
	'B': (1, 1),
	'B2': (1, 2),
	'B3': (1, 3),
	'B\'': (1, 3),
	'U1': (2, 1),
	'U': (2, 1),
	'U2': (2, 2),
	'U3': (2, 3),
	'U\'': (2, 3),
	'D1': (3, 3),
	'D': (3, 3),
	'D2': (3, 2),
	'D3': (3, 1),
	'D\'': (3, 1),
	'L1': (4, 3),
	'L': (4, 3),
	'L2': (4, 2),
	'L3': (4, 1),
	'L\'': (4, 1),
	'R1': (5, 1),
	'R': (5, 1),
	'R2': (5, 2),
	'R3': (5, 3),
	'R\'': (5, 3),
}
move_str_map = {
	str_move_map[s]: s
	for t in [(f, f+'2', f+'\'') for f in 'FBUDLR'] for s in t
}
r0_sym = {
	(0, 1): (0, 1),
	(0, 2): (0, 2),
	(0, 3): (0, 3),
	(1, 1): (1, 1),
	(1, 2): (1, 2),
	(1, 3): (1, 3),

	(2, 1): (4, 3),
	(2, 2): (4, 2),
	(2, 3): (4, 1),
	(3, 1): (5, 3),
	(3, 2): (5, 2),
	(3, 3): (5, 1),

	(4, 1): (3, 1),
	(4, 2): (3, 2),
	(4, 3): (3, 3),
	(5, 1): (2, 1),
	(5, 2): (2, 2),
	(5, 3): (2, 3),
}
r1_sym = {
	(0, 1): (5, 3),
	(0, 2): (5, 2),
	(0, 3): (5, 1),
	(1, 1): (4, 3),
	(1, 2): (4, 2),
	(1, 3): (4, 1),

	(2, 1): (3, 3),
	(2, 2): (3, 2),
	(2, 3): (3, 1),
	(3, 1): (2, 3),
	(3, 2): (2, 2),
	(3, 3): (2, 1),

	(4, 1): (1, 3),
	(4, 2): (1, 2),
	(4, 3): (1, 1),
	(5, 1): (0, 3),
	(5, 2): (0, 2),
	(5, 3): (0, 1),
}
r2_sym = {
	(0, 1): (2, 3),
	(0, 2): (2, 2),
	(0, 3): (2, 1),
	(1, 1): (3, 3),
	(1, 2): (3, 2),
	(1, 3): (3, 1),

	(2, 1): (4, 3),
	(2, 2): (4, 2),
	(2, 3): (4, 1),
	(3, 1): (5, 3),
	(3, 2): (5, 2),
	(3, 3): (5, 1),

	(4, 1): (0, 1),
	(4, 2): (0, 2),
	(4, 3): (0, 3),
	(5, 1): (1, 1),
	(5, 2): (1, 2),
	(5, 3): (1, 3),
}
r3_sym = {
	(0, 1): (1, 1),
	(0, 2): (1, 2),
	(0, 3): (1, 3),
	(1, 1): (0, 1),
	(1, 2): (0, 2),
	(1, 3): (0, 3),

	(2, 1): (2, 3),
	(2, 2): (2, 2),
	(2, 3): (2, 1),
	(3, 1): (3, 3),
	(3, 2): (3, 2),
	(3, 3): (3, 1),

	(4, 1): (4, 3),
	(4, 2): (4, 2),
	(4, 3): (4, 1),
	(5, 1): (5, 3),
	(5, 2): (5, 2),
	(5, 3): (5, 1),
}
orient_sym = {(0, 2): (0, 0, 0, 0), (0, 5): (1, 0, 0, 0), (3, 5): (2, 0, 2, 0), (3, 1): (0, 1, 2, 0), (4, 1): (1, 1, 2, 0), (4, 2): (2, 1, 0, 0), (4, 0): (0, 0, 1, 0), (3, 0): (3, 0, 1, 0), (1, 5): (0, 1, 1, 0), (1, 2): (3, 1, 1, 0), (1, 4): (2, 1, 1, 0), (3, 4): (1, 1, 0, 0), (5, 0): (2, 0, 1, 0), (5, 2): (1, 0, 2, 0), (4, 3): (3, 0, 2, 0), (2, 1): (2, 1, 2, 0), (2, 5): (3, 1, 0, 0), (0, 3): (2, 0, 0, 0), (0, 4): (3, 0, 0, 0), (2, 4): (0, 0, 2, 0), (5, 1): (3, 1, 2, 0), (5, 3): (0, 1, 0, 0), (2, 0): (1, 0, 1, 0), (1, 3): (1, 1, 1, 0)}
