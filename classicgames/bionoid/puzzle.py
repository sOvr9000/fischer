
import numpy as np
from typing import Iterable
from math import comb



class Bionoid:
    def __init__(self, size: tuple[int, int]):
        if not isinstance(size, tuple):
            raise ValueError(f'Board size={size} must be a tuple of two positive, even integers.')
        if len(size) != 2:
            raise ValueError(f'Board size={size} must be a tuple of two positive, even integers.')
        if not isinstance(size[0], (int, np.int_)) or not isinstance(size[1], (int, np.int_)):
            raise ValueError(f'Board size={size} must be a tuple of two positive, even integers.')
        if size[0] % 2 != 0 or size[1] % 2 != 0:
            raise ValueError(f'Board size={size} must be a tuple of two positive, even integers.')
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError(f'Board size={size} must be a tuple of two positive, even integers.')

        # By a direct mathematical result of combinatoric analysis on number of possible rows vs. number of columns, or number of possible columns vs. number of rows, check to see if a puzzle can actually be created with the given size.
        # Take for example, if given the size `(18, 6)`, then it is impossible to create a Bionoid puzzle because there is guaranteed to be at least one pair of rows which are equivalent; there are only 6!/3!^2 = 6*5*4/3/2 = 20 unique rows that can be created with 3 zeros and 3 ones in some arrangement, and that does not exclude the rows that have the illegal triple ones or triple zeros.
        # Omitting some of the illegal rows, we count that there are at least 6 - 2 = 4 unique rows that are illegal, considering only the possible locations of triples within the rows.
        # Hence, for the example of size `(18, 6)`, we check the condition for the smaller number, which is `6`: It's an invalid board size if 6!/3!^2 - (6 - 2) < 18.  In this case, 16 < 18, so that means there does not exist a Bionoid puzzle of size `(18, 6)`.
        m, M = min(size), max(size)
        if comb(m, m // 2) + 2 - m < M:
            raise ValueError(f'Invalid board size={size}.  A board size (a, b) or (b, a) with a <= b must at least satisfy nCr(a, a//2) + 2 - a >= b.  In this case, {comb(m, m // 2) + 2 - m} < {M}.  Too close to pushing this limit might also restrict the ability to generate new valid puzzles.')

        self.board_size = size
        self.half_size = size[0] // 2, size[1] // 2
        self.solution = np.empty(self.board_size, dtype=bool)
        self.board = -np.ones(self.board_size, dtype=int) # -1 means no entry, 0 means 0, 1 means 1
        self.givens_mask = np.empty(self.board_size, dtype=bool) # a mask of the solution array that is to be revealed at the start
        self.pow2_cols = np.power(2, np.arange(self.board_size[0]), dtype=int) # Used to quickly detect linear dependent columns.
        self.pow2_rows = np.power(2, np.arange(self.board_size[1]), dtype=int) # Used to quickly detect linear dependent rows.
    def reset(self, new_puzzle: bool = True):
        '''
        Clear the board.

        If `new_puzzle=True`, then generate a new random puzzle.
        '''
        if new_puzzle:
            self.randomize_solution()
        self.reset_board()
    def reset_board(self):
        '''
        Clear the guesses on the board and leave only the givens.
        '''
        self.board[:] = -1
        self.board[self.givens_mask] = self.solution[self.givens_mask]
    def _is_solution_valid(self) -> bool:
        '''
        Return whether the current solution to the board is valid.
        '''
        # Test for when 3 ones in a vertical arrangement exists anywhere.
        if np.any(np.logical_and(self.solution[:-2], np.logical_and(self.solution[1:-1], self.solution[2:]))):
            return False
        # Test for when 3 ones in a horizontal arrangement exists anywhere.
        if np.any(np.logical_and(self.solution[:, :-2], np.logical_and(self.solution[:, 1:-1], self.solution[:, 2:]))):
            return False
        # Test for when 3 zeros in a vertical arrangement exists anywhere.
        c = np.logical_not(self.solution)
        if np.any(np.logical_and(c[:-2], np.logical_and(c[1:-1], c[2:]))):
            return False
        # Test for when 3 zeros in a horizontal arrangement exists anywhere.
        if np.any(np.logical_and(c[:, :-2], np.logical_and(c[:, 1:-1], c[:, 2:]))):
            return False
        # Test for the total count of ones and zeros in each column.
        if np.any(np.sum(self.solution, axis=0, dtype=int) != self.half_size[0]):
            return False
        # Test for the total count of ones and zeros in each row.
        if np.any(np.sum(self.solution, axis=1, dtype=int) != self.half_size[1]):
            return False
        return True
    def _solution_squares_of_three_or_more(self) -> Iterable[tuple[int, int]]:
        '''
        Return an iterable of coordinates `(i, j)` such that the square at `(i, j)` in the solution is part of the reason why the solution is invalid.
        '''
        indices = np.array([
            [i, j]
                for j in range(self.board_size[1])
            for i in range(self.board_size[0])
        ])
        np.random.shuffle(indices)
        for i, j in indices:
            do_yield = False
            if i == 0:
                if self.solution[i, j] == self.solution[i+1, j] == self.solution[i+2, j]:
                    do_yield = True
            elif i == self.board_size[0] - 1:
                if self.solution[i, j] == self.solution[i-1, j] == self.solution[i-2, j]:
                    do_yield = True
            else:
                if self.solution[i, j] == self.solution[i-1, j] == self.solution[i+1, j]:
                    do_yield = True
            if not do_yield:
                if j == 0:
                    if self.solution[i, j] == self.solution[i, j+1] == self.solution[i, j+2]:
                        do_yield = True
                elif j == self.board_size[1] - 1:
                    if self.solution[i, j] == self.solution[i, j-1] == self.solution[i, j-2]:
                        do_yield = True
                else:
                    if self.solution[i, j] == self.solution[i, j-1] == self.solution[i, j+1]:
                        do_yield = True
            if do_yield:
                yield i, j
    def _find_illegal_solution_squares(self) -> list[tuple[int, int]]:
        '''
        Return a list of coordinates `(i, j)` to indicate where there's at least one value that makes the current solution invalid.
        '''
        return list(set(self._solution_squares_of_three_or_more()))
    def _equivalent_solution_rows(self) -> Iterable[tuple[int, int]]:
        '''
        Return an iterable of integer pairs that correspond to the indices of the rows which are equivalent to each other.
        '''
        decoded = self._solution_row_encodings() # A vector of ints such that equivalent integers means equivalent rows.
        for i in range(self.board_size[0] - 1):
            for j in range(i + 1, self.board_size[0]):
                if decoded[i] == decoded[j]:
                    yield i, j
    def _equivalent_solution_columns(self) -> Iterable[tuple[int, int]]:
        '''
        Return an iterable of integer pairs that correspond to the indices of the columns which are equivalent to each other.
        '''
        decoded = self._solution_column_encodings() # A vector of ints such that equivalent integers means equivalent columns.
        for i in range(self.board_size[1] - 1):
            for j in range(i + 1, self.board_size[1]):
                if decoded[i] == decoded[j]:
                    yield i, j
    def _toggle_solution_corners(self, i, j, i2, j2):
        '''
        Toggle the values at `(i, j)`, `(i2, j)`, `(i2, j2)`, and `(i, j2)`.

        This is a self-inverting operation, so it can be used to quickly and easily traverse search trees for valid puzzle solutions.
        '''
        self.solution[i, j] = not self.solution[i, j]
        self.solution[i2, j] = not self.solution[i2, j]
        self.solution[i2, j2] = not self.solution[i2, j2]
        self.solution[i, j2] = not self.solution[i, j2]
    def _solution_has_repeated_vectors(self) -> bool:
        return any(self._equivalent_solution_columns()) or any(self._equivalent_solution_rows())
    def _solution_row_encodings(self) -> np.ndarray[int]:
        '''
        Return the single-integer encoding of each row.

        These can be used to quickly determine relationships between rows -- such as equivalence or binary complementarity -- by using bitwise operations.
        '''
        return np.dot(self.solution, self.pow2_rows)
    def _solution_column_encodings(self) -> np.ndarray[int]:
        '''
        Return the single-integer encoding of each column.

        These can be used to quickly determine relationships between columns -- such as equivalence or binary complementarity -- by using bitwise operations.
        '''
        return np.dot(self.pow2_cols, self.solution)
    def randomize_solution(self):
        '''
        Generate a new valid solution
        '''
        # Create rows with correct sums.
        row = np.zeros(self.board_size[1], dtype=bool)
        row[:self.half_size[1]] = True
        while True:
            for i in range(self.board_size[0]):
                np.random.shuffle(row)
                self.solution[i] = row
            # Ensure that each column also has the correct sum.
            # This works by randomly swapping values between columns such that row sums are preserved until the column sums are okay.
            s = np.sum(self.solution, axis=0, dtype=int) # the current column sums, to be modified as values get swapped
            for j in range(self.board_size[1]):
                if s[j] == self.half_size[0]:
                    continue
                j2s = np.arange(j + 1, self.board_size[1])
                np.random.shuffle(j2s)
                for j2 in j2s:
                    if s[j2] != self.half_size[0] and (s[j] > self.half_size[0]) ^ (s[j2] > self.half_size[0]):
                        k = abs(s[j] - self.half_size[0])
                        for i in np.random.choice(self.board_size[0], size=self.board_size[0], replace=False):
                            if (s[j] < self.half_size[0]) ^ self.solution[i, j] and self.solution[i, j] ^ self.solution[i, j2]:
                                t = self.solution[i, j]
                                self.solution[i, j] = self.solution[i, j2]
                                self.solution[i, j2] = t
                                k -= 1
                                if s[j] > self.half_size[0]:
                                    s[j] -= 1
                                    s[j2] += 1
                                else:
                                    s[j] += 1
                                    s[j2] -= 1
                                if k <= 0:
                                    break
                        if s[j] == self.half_size[0]:
                            break
            # Now ensure that the "no more than two in a row" rule is satisfied.
            # This works by randomly toggling the values of rectangularly aligned squares such that the row and column sums are preserved until the "no more than two in a row" and "linearly independent" rules are satisfied.
            # print(repr(self))
            # print(f'linearly dependent rows: {list(self._equivalent_solution_rows())}')
            # print(f'linearly dependent columns: {list(self._equivalent_solution_columns())}')
            indices = []
            for steps in range(1024):
                iterated = False
                b = False
                for i, j in self._solution_squares_of_three_or_more():
                    iterated = True
                    v = self.solution[i, j]
                    for i2, j2 in indices:
                        if i2 != i and j2 != j:
                            if self.solution[i2, j2] == v:
                                if self.solution[i2, j] ^ v and self.solution[i, j2] ^ v:
                                    b = True
                                    break
                    if b:
                        break
                    indices.append((i, j))
                if len(indices) == 0:
                    # if iterated:
                    #     print('problems exist but no index pair found')
                    break
                indices.clear()
                # Rotate the vertices (i, j), (i2, j), (i2, j2), (i, j2), which is simply a NOT on each.
                self._toggle_solution_corners(i, j, i2, j2)
                # print('\n'*3)
                # print(i, j, i2, j2)
                # print(repr(self))
                # input()
            if iterated:
                pass
                # print('problems exist but no index pair found')
                # print(f'linearly dependent rows: {list(self._equivalent_solution_rows())}')
                # print(f'linearly dependent columns: {list(self._equivalent_solution_columns())}')
            else:
                # print(f'fixed in {steps} steps')
                break
        self.givens_mask[:] = False
    def __str__(self) -> str:
        return '\n'.join(
            ' '.join(
                '.'
                if v == -1 else
                str(v)
                for v in row
            )
            for row in self.board
        )
    def __repr__(self) -> str:
        return f'Solution:\n{self.solution.astype(int)}\nBoard:\n{str(self)}'
    def solution_to_bytes(self) -> bytes:
        return self.solution.tobytes()



puzzle = Bionoid((14, 14))
puzzle.reset()
print(repr(puzzle))



# If two rows/columns are equivalent, then their encodings are the same number.
# How to check if two rows/columns differ by only two values:
# - let x = XOR of encodings
# - while x is even:
# -     x >>= 1
# - if x is odd and x > 1, then return False
# - x -= 1
# - while x is even:
# -     x >>= 1
# - return x == 1


# 24x24
# [[1 0 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0]
#  [1 1 0 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 1 0 0 1]
#  [0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 0 1 1 0 1 0]
#  [1 0 1 0 0 1 0 1 0 0 1 0 1 0 1 1 0 1 0 1 0 1 1 0]
#  [0 1 0 1 1 0 1 0 1 0 1 0 0 1 1 0 1 0 1 0 1 0 0 1]
#  [1 0 1 0 1 0 1 1 0 1 0 1 1 0 0 1 0 0 1 1 0 0 1 0]
#  [0 1 0 1 0 1 0 0 1 1 0 1 0 1 0 0 1 1 0 0 1 1 0 1]
#  [0 1 1 0 1 0 1 1 0 0 1 0 1 0 1 0 1 0 1 0 0 1 0 1]
#  [1 0 1 1 0 1 0 0 1 1 0 1 0 1 0 1 0 1 0 1 0 0 1 0]
#  [0 1 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 0 1 0 0 1]
#  [0 0 1 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 0 1 1 0]
#  [1 1 0 0 1 0 1 0 0 1 0 0 1 0 1 0 1 0 1 0 1 1 0 1]
#  [0 1 0 1 0 1 0 1 1 0 0 1 0 1 0 1 0 1 0 1 0 0 1 1]
#  [1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 1 0 1 0 1 1 0]
#  [0 1 1 0 1 0 0 1 0 1 0 1 0 1 0 1 1 0 1 0 1 1 0 0]
#  [1 0 0 1 0 1 1 0 1 0 1 0 1 0 1 0 1 0 0 1 0 0 1 1]
#  [0 1 1 0 1 0 0 1 0 1 0 1 0 1 0 1 0 1 1 0 1 0 0 1]
#  [0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 1 1 0 1 0 1 1 0]
#  [1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 0 0 1 0 1 0 0 1]
#  [0 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 1 0]
#  [1 0 1 0 1 0 1 0 1 0 1 1 0 1 0 1 0 0 1 0 1 0 0 1]
#  [0 1 0 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 0 1 0 0 1 0]
#  [1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 1 1 0 1 1 0 0]
#  [1 0 0 1 0 1 0 1 1 0 1 1 0 1 1 0 1 0 1 0 1 1 0 1]]