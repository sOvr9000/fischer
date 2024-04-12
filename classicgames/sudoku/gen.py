

import numpy as np
from .solver import has_single_solution, is_valid, reduced_forms, solve
from .sudoku_io import write_puzzles
from .su import puzzle_str
from typing import Iterable
from multiprocessing import Pool



def generate_puzzle(initial_givens: int = 28, reduce: bool = False, max_solver_steps: int = 512) -> np.ndarray:
    '''
    Generate a Sudoku puzzle with a unique solution.

    If `reduce` is `True`, the puzzle is reduced to its minimal form.
    '''
    board = np.zeros((9, 9), dtype=np.int8)
    while True:
        board[:] = 0
        for _ in range(initial_givens):
            while True:
                row, col, num = np.random.randint(9, size=3)
                if is_valid(board, row, col, num + 1):
                    break
            board[row, col] = num + 1
        # print(board)
        if has_single_solution(board, max_steps=max_solver_steps):
            break
        # Puzzle has multiple solutions or is unsolvable
        # print('bad puzzle: multiple solutions or unsolvable')
    if reduce:
        num_givens = np.sum(board != 0)
        print(f'generate_puzzle: found puzzle with {num_givens} givens')
        board = next(reduced_forms(board))
        removed_givens = num_givens - np.sum(board != 0)
        print(f'generate_puzzle: removed {removed_givens} givens (now {np.sum(board != 0)} givens)')
        # board = min(reduced_forms(board, max_steps=max_solver_steps), key=lambda b: np.sum(b != 0))
        # print(f'num givens: {np.sum(board != 0)}')
    return board

def generate_puzzles(n: int, reduce: bool = False, max_solver_steps: int = 512) -> Iterable[np.ndarray]:
    '''
    Generate `n` Sudoku puzzles.
    '''
    for _ in range(n):
        yield generate_puzzle(initial_givens=np.random.randint(32, 40), reduce=reduce, max_solver_steps=max_solver_steps)

def _generate_puzzles_mp(n: int, reduce: bool = False, max_solver_steps: int = 512) -> np.ndarray:
    '''
    Generate `n` Sudoku puzzles.
    '''
    return np.array([generate_puzzle(initial_givens=np.random.randint(32, 40), reduce=reduce, max_solver_steps=max_solver_steps) for _ in range(n)])

def generate_puzzles_mp(block_size: int = 16, processes: int = 12, reduce: bool = False, max_solver_steps: int = 512) -> np.ndarray:
    '''
    Generate Sudoku puzzles using multiprocessing.
    '''
    with Pool(processes) as pool:
        puzzles = pool.starmap(_generate_puzzles_mp, [(block_size, reduce, max_solver_steps)] * processes)
    return np.concatenate(puzzles)


if __name__ == '__main__':
    block_size = 1
    processes = 15
    reduce = True
    max_steps = 128
    while True:
        for i, puzzle in enumerate(generate_puzzles_mp(block_size=block_size, processes=processes, reduce=reduce, max_solver_steps=max_steps)):
            print(f'Puzzle {i + 1}:')
            sol = puzzle.copy()
            solve(sol, max_steps=max_steps)
            write_puzzles('puzzles_reduced.sudk' if reduce else 'puzzles.sudk', [puzzle], [sol])
            print(puzzle_str(puzzle))
            print()
