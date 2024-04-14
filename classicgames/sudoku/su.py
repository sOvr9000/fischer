
import numpy as np
from .solver import solve


def number_swap(puzzle: np.ndarray, permutation: np.ndarray, inverted: bool = False) -> np.ndarray:
    '''
    Swap two numbers in a Sudoku puzzle.
    
    For example:
    ```
    number_swap(puzzle, np.array([1, 6, 7, 8, 9, 4, 5, 2, 3]))
    ```
    swaps 1 with 1, 2 with 6, 3 with 7, and so on.

    If `inverted` is `True`, the permutation is inverted before being applied.
    '''
    if inverted:
        permutation = np.argsort(permutation) + 1
    return np.where(puzzle == 0, 0, permutation[puzzle - 1])



def convert_to_seed_puzzle(puzzle: np.ndarray) -> np.ndarray:
    '''
    Convert a Sudoku puzzle to a seed puzzle, where the first row is `[1, 2, 3, 4, 5, 6, 7, 8, 9]`.
    '''
    return number_swap(puzzle, puzzle[0], inverted=True)


def is_seed_puzzle(puzzle: np.ndarray) -> bool:
    '''
    Return whether a Sudoku puzzle is a seed.
    '''
    return np.array_equal(puzzle[0], np.arange(1, 10))

def puzzle_str(puzzle: np.ndarray) -> str:
    '''
    Return a string representation of a Sudoku puzzle.
    '''
    return '\n'.join(' '.join(str(num) if num != 0 else '-' for num in row) for row in puzzle)



permutations3 = np.array([
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
])

def row_swap(puzzles: np.ndarray, perm_indices: tuple[int, int, int]) -> np.ndarray:
    '''
    Swap the rows of puzzles.
    '''
    return np.concatenate([puzzles[:, permutations3[perm_indices[0]]], puzzles[:, permutations3[perm_indices[1]] + 3], puzzles[:, permutations3[perm_indices[2]] + 6]], axis=1)

def column_swap(puzzles: np.ndarray, perm_indices: tuple[int, int, int]) -> np.ndarray:
    '''
    Swap the columns of puzzles.
    '''
    return np.concatenate([puzzles[:, :, permutations3[perm_indices[0]]], puzzles[:, :, permutations3[perm_indices[1]] + 3], puzzles[:, :, permutations3[perm_indices[2]] + 6]], axis=2)

def box_row_swap(puzzles: np.ndarray, perm_index: int) -> np.ndarray:
    '''
    Swap the box rows of puzzles.
    '''
    perm = permutations3[perm_index] * 3
    return np.concatenate([puzzles[:, perm[0]:perm[0]+3], puzzles[:, perm[1]:perm[1]+3], puzzles[:, perm[2]:perm[2]+3]], axis=1)

def box_column_swap(puzzles: np.ndarray, perm_index: int) -> np.ndarray:
    '''
    Swap the box columns of puzzles.
    '''
    perm = permutations3[perm_index] * 3
    return np.concatenate([puzzles[:, :, perm[0]:perm[0]+3], puzzles[:, :, perm[1]:perm[1]+3], puzzles[:, :, perm[2]:perm[2]+3]], axis=2)

def transpose(puzzles: np.ndarray) -> np.ndarray:
    '''
    Transpose puzzles.
    '''
    return np.transpose(puzzles, (0, 2, 1))

def random_symmetry(puzzles: np.ndarray) -> np.ndarray:
    '''
    Get a random symmetry of a series of Sudoku puzzles.
    '''
    perm = np.arange(1, 10)
    np.random.shuffle(perm)
    puzzles = number_swap(puzzles, perm)
    puzzles = row_swap(puzzles, np.random.randint(0, 6, 3))
    puzzles = column_swap(puzzles, np.random.randint(0, 6, 3))
    puzzles = box_row_swap(puzzles, np.random.randint(0, 6))
    puzzles = box_column_swap(puzzles, np.random.randint(0, 6))
    if np.random.randint(0, 2):
        puzzles = transpose(puzzles)
    return puzzles


if __name__ == '__main__':
    from solver import solve

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
    print('Original board:')
    print(puzzle_str(board))
    print()
    perm = np.array([1, 6, 7, 8, 9, 4, 5, 2, 3])
    print('Number swapped:')
    print(f'perm = {perm}')
    s = number_swap(board, perm)
    print(s)
    print()
    print('Number swapped (inverted perm):')
    s = number_swap(s, perm, inverted=True)
    print(puzzle_str(s)) # This is the original board
    print()

    print('Solved board:')
    steps = solve(board)
    print(puzzle_str(board))
    print()

    print('Seed board:')
    seed = convert_to_seed_puzzle(board)
    print(puzzle_str(seed))
    print()



