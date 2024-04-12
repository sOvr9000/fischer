
import numpy as np
from .solver import solve
from .su import convert_to_seed_puzzle



def read_puzzles(file, start_puzzle: int = 0, end_puzzle: int = -1) -> tuple[np.ndarray, np.ndarray]:
    '''
    Read Sudoku puzzles from a file.

    The first row of each puzzle is `[1, 2, 3, 4, 5, 6, 7, 8, 9]` so that the puzzle serves as a seed for generating more by symmetry.
    '''
    if end_puzzle == -1:
        end_puzzle = np.inf
    puzzles = []
    with open(file, 'rb') as f:
        f.seek(start_puzzle * 47)
        while True:
            # Each puzzle is 47 bytes long.
            _bytes = f.read(47)
            if not _bytes:
                break
            # Decode the puzzle, bit-by-bit.
            puzzle = np.zeros((9, 9), dtype=np.int8)
            puzzle[0] = np.arange(1, 10)
            for i, _byte in enumerate(_bytes):
                for j in range(8):
                    loc = 8 * i + j
                    # The first row is [1, 2, 3, 4, 5, 6, 7, 8, 9].
                    if loc < 9:
                        puzzle[0, loc] = loc + 1
                        if _byte >> j & 1:
                            # Negatives are given values.
                            puzzle[0, loc] = -puzzle[0, loc]
                        continue
                    if loc >= 369:
                        # 7 remaining bits are padding.
                        break
                    loc -= 9
                    p = loc % 5
                    col = loc // 5 % 9
                    row = loc // 45 + 1
                    bit = _byte >> j & 1
                    # if row == 1:
                    #     print(row, col, p, bit)
                    if p == 4 and bit:
                        puzzle[row, col] = -puzzle[row, col]
                    else:
                        puzzle[row, col] += bit << p
            puzzles.append(puzzle)
    return get_puzzle_and_solution(np.array(puzzles))


def write_puzzles(file, puzzles: np.ndarray, solutions: np.ndarray):
    '''
    Write Sudoku puzzles to a file.

    Each puzzle is expected to be an array of shape `(9, 9)` with values in the range `[-9, 9]`.  If a value is negative, it is a given value.
    '''
    if isinstance(puzzles, list):
        puzzles = np.array(puzzles)
    if isinstance(solutions, list):
        solutions = np.array(solutions)
    puzzles = format_puzzle_and_solution(puzzles, solutions)
    with open(file, 'ab') as f:
        for puzzle in puzzles:
            seed = convert_to_seed_puzzle(np.abs(puzzle))
            bits = []
            for i in range(9):
                bits.append(1 if puzzle[0, i] < 0 else 0)
                # print(bits[-1], end='')
            # print()
            for i in range(72):
                row = i // 9 + 1
                col = i % 9
                v = seed[row, col]
                for j in range(4):
                    bits.append(v >> j & 1)
                    # print(bits[-1], end='')
                bits.append(1 if puzzle[row, col] < 0 else 0)
                # print()
            # print(bits)
            # print(len(bits))
            for i in range(47):
                _byte = 0
                for j in range(8):
                    loc = 8 * i + j
                    if loc >= 369:
                        break
                    _byte |= bits[loc] << j
                f.write(bytes([_byte]))

def format_puzzle_and_solution(puzzle: np.ndarray, solution: np.ndarray) -> np.ndarray:
    '''
    Convert a puzzle and its solution to a combined array with negative values to indicate where the given values are.
    '''
    return np.where(puzzle == 0, solution, -puzzle)

def get_puzzle_and_solution(puzzle: np.ndarray) -> np.ndarray:
    '''
    Split a combined array into a puzzle and its solution.
    '''
    solution = np.abs(puzzle)
    return np.where(puzzle < 0, solution, 0), solution

if __name__ == '__main__':
    board = np.array([
        [5,3,0, 0,7,0, 0,0,0],
        [0,0,0, 0,9,5, 0,0,0],
        [0,9,8, 0,0,0, 0,6,0],

        [8,0,0, 0,6,0, 0,0,0],
        [4,0,0, 8,0,3, 0,0,1],
        [7,0,0, 0,2,0, 0,0,0],

        [0,0,0, 0,0,0, 2,8,0],
        [0,0,0, 4,1,9, 0,0,5],
        [0,0,0, 0,0,0, 0,7,0],
    ])
    print(board)
    puzzle = board.copy()
    steps = solve(board)
    board = convert_to_seed_puzzle(board)
    board = np.where(puzzle == 0, board, -board)
    print(board)
    write_puzzles('puzzles.sudk', [board])
    puzzles = read_puzzles('puzzles.sudk')
    print(puzzles)
    print(np.array_equal(puzzles[0], board))

