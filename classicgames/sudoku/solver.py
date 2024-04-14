
from typing import Iterable
import numpy as np
from numba import jit


@jit()
def is_valid(board: np.ndarray, row: int, col: int, num: int) -> bool:
    # Check if `num` can be placed at (row, col).
    for i in range(9):
        if board[row, i] == num:
            return False
        if board[i, col] == num:
            return False
        if board[row - row % 3 + i // 3, col - col % 3 + i % 3] == num:
            return False
    return True

def is_solved(board: np.ndarray) -> bool:
    return \
        np.all(np.sum(board, axis=1) == 45) \
        and np.all(np.sum(board, axis=0) == 45) \
        and np.all(np.prod(board, axis=1) == 362880) \
        and np.all(np.prod(board, axis=0) == 362880)


def hidden_singles(candidates: np.ndarray) -> bool:
    # Employ technique "Hidden Singles": If a candidate exists in only one square of a row, box, or column, then the other candidates in that square may be removed.
    # Return whether the candidates were changed.
    old_candidates = candidates.copy()
    for i in range(9):
        # rows
        s = np.sum(candidates[i, :, :], axis=0)
        # The relevant indices (each corresponding to a number candidate) of `s` are where it's equal to 1.
        cd = np.where(s == 1)[0]
        for c in cd:
            col = np.argmax(candidates[i, :, c])
            # progress |= np.any(candidates[i, col, :c]) or np.any(candidates[i, col, c+1:])
            candidates[i, col, :] = False
            candidates[i, col, c] = True
        # columns
        s = np.sum(candidates[:, i, :], axis=0)
        cd = np.where(s == 1)[0]
        for c in cd:
            row = np.argmax(candidates[:, i, c])
            # progress |= np.any(candidates[row, i, :c]) or np.any(candidates[row, i, c+1:])
            candidates[row, i, :] = False
            candidates[row, i, c] = True
        # boxes
        box_row_start = i // 3 * 3
        box_col_start = i % 3 * 3
        flat_box = candidates[box_row_start:box_row_start+3, box_col_start:box_col_start+3, :].reshape(9, 9)
        s = np.sum(flat_box, axis=0)
        cd = np.where(s == 1)[0]
        for c in cd:
            row, col = np.unravel_index(np.argmax(flat_box[:, c]), (3, 3))
            row += box_row_start
            col += box_col_start
            candidates[row, col, :] = False
            candidates[row, col, c] = True
    return not np.array_equal(candidates, old_candidates)

def naked_pairs(candidates: np.ndarray) -> bool:
    # Employ technique "Naked Pairs"
    # Return whether the candidates were changed.
    old_candidates = candidates.copy()
    for k in range(81):
        i = k // 9
        j = k % 9
        # rows
        if np.sum(candidates[i, j, :]) == 2:
            for k in range(j + 1, 9):
                if np.array_equal(candidates[i, j, :], candidates[i, k, :]):
                    pair_values = np.where(candidates[i, j, :] == 1)[0]
                    for l in range(9):
                        if l != j and l != k:
                            candidates[i, l, pair_values[0]] = False
                            candidates[i, l, pair_values[1]] = False
        # columns
        if np.sum(candidates[j, i, :]) == 2:
            for k in range(j + 1, 9):
                if np.array_equal(candidates[j, i, :], candidates[k, i, :]):
                    pair_values = np.where(candidates[j, i, :] == 1)[0]
                    for l in range(9):
                        if l != j and l != k:
                            candidates[l, i, pair_values[0]] = False
                            candidates[l, i, pair_values[1]] = False
        # boxes
        box_row_start = i // 3 * 3
        box_col_start = i % 3 * 3
        flat_box = candidates[box_row_start:box_row_start+3, box_col_start:box_col_start+3, :].reshape(9, 9)
        if np.sum(flat_box[j, :]) == 2:
            for k in range(j + 1, 9):
                if np.array_equal(flat_box[j, :], flat_box[k, :]):
                    pair_values = np.where(flat_box[j, :] == 1)[0]
                    for l in range(9):
                        if l != j and l != k:
                            _row, _col = np.unravel_index(l, (3, 3))
                            _row += box_row_start
                            _col += box_col_start
                            candidates[_row, _col, pair_values[0]] = False
                            candidates[_row, _col, pair_values[1]] = False
    return not np.array_equal(candidates, old_candidates)

def hidden_pairs(candidates: np.ndarray) -> bool:
    # Employ technique "Hidden Pairs"
    # Return whether the candidates were changed.
    old_candidates = candidates.copy()
    for i in range(9):
        # rows
        for j in range(9):
            if np.sum(candidates[i, j, :]) == 2:
                for k in range(j + 1, 9):
                    if np.array_equal(candidates[i, j, :], candidates[i, k, :]):
                        for l in range(9):
                            if l != j and l != k:
                                candidates[i, l, :] = candidates[i, l, :] & candidates[i, j, :]
        # columns
        for j in range(9):
            if np.sum(candidates[j, i, :]) == 2:
                for k in range(j + 1, 9):
                    if np.array_equal(candidates[j, i, :], candidates[k, i, :]):
                        for l in range(9):
                            if l != j and l != k:
                                candidates[l, i, :] = candidates[l, i, :] & candidates[j, i, :]
        # boxes
        box_row_start = i // 3 * 3
        box_col_start = i % 3 * 3
        flat_box = candidates[box_row_start:box_row_start+3, box_col_start:box_col_start+3, :].reshape(9, 9)
        for j in range(9):
            if np.sum(flat_box[j, :]) == 2:
                for k in range(j + 1, 9):
                    if np.array_equal(flat_box[j, :], flat_box[k, :]):
                        for l in range(9):
                            if l != j and l != k:
                                _row, _col = np.unravel_index(l, (3, 3))
                                _row += box_row_start
                                _col += box_col_start
                                candidates[_row, _col, :] = candidates[_row, _col, :] & flat_box[j, :]
    return not np.array_equal(candidates, old_candidates)

def x_wing(candidates: np.ndarray) -> bool:
    # Employ technique "X-Wing"
    # Return whether the candidates were changed.
    old_candidates = candidates.copy()
    for val in range(9):
        # rows
        for i in range(9):
            if np.sum(candidates[i, :, val]) == 2:
                for j in range(i + 1, 9):
                    if np.array_equal(candidates[i, :, val], candidates[j, :, val]):
                        for k in range(9):
                            if k != i and k != j:
                                candidates[k, :, val] = False
        # columns
        for i in range(9):
            if np.sum(candidates[:, i, val]) == 2:
                for j in range(i + 1, 9):
                    if np.array_equal(candidates[:, i, val], candidates[:, j, val]):
                        for k in range(9):
                            if k != i and k != j:
                                candidates[:, k, val] = False
    return not np.array_equal(candidates, old_candidates)

def prune_candidates(candidates: np.ndarray) -> None:
    if hidden_singles(candidates):
        # print('hidden singles succeeded')
        return
    if naked_pairs(candidates):
        # print('naked pairs succeeded')
        return
    if hidden_pairs(candidates):
        # print('hidden pairs succeeded')
        return
    if x_wing(candidates):
        # print('x-wing succeeded')
        return
    # print('no pruning succeeded')


def update_candidates_global(board: np.ndarray, candidates: np.ndarray, prune: bool) -> None:
    for i in range(81):
        row = i // 9
        col = i % 9
        if board[row, col] != 0:
            continue
        for num in range(9):
            candidates[row, col, num] = is_valid(board, row, col, num + 1)
    if prune:
        prune_candidates(candidates)








def solve(board: np.ndarray, max_steps: int = 512, backdoor_cell: tuple[int, int] = None) -> int:
    stack = [] # stack of (index, num), where index = row * 9 + col
    candidates = np.zeros((9, 9, 9), dtype=np.bool_) # candidates[row, col, num] = True if num is a candidate for cell (row, col)

    # Initialize candidates.
    update_candidates_global(board, candidates, True)

    row = -1
    col = -1
    index = 0
    num = -1

    if backdoor_cell is not None:
        row, col = backdoor_cell
        num = 0

    steps = 0
    while True:
        steps += 1
        if steps > max_steps:
            return -1
        update_candidates_global(board, candidates, False)
        if row == -1:
            # Find the next empty cell which has the fewest candidates.
            num_candidates = np.where(board == 0, np.sum(candidates, axis=2), 10)
            index = np.argmin(num_candidates)
            row = index // 9
            col = index % 9
            if num_candidates[row, col] > 1:
                prune_candidates(candidates)
                num_candidates = np.where(board == 0, np.sum(candidates, axis=2), 10)
                index = np.argmin(num_candidates)
                row = index // 9
                col = index % 9
            if num_candidates[row, col] == 0:
                # At least one of the remaining empty cells has no possible candidates.  Backtrack.
                # _row = row
                # _col = col
                if len(stack) == 0:
                    # If we've tried all numbers and none of them are valid, the puzzle is unsolvable.
                    # print(board)
                    # print('unsolvable')
                    # print(f'steps: {steps}')
                    return -1
                index, num = stack.pop()
                row = index // 9
                col = index % 9
                board[row, col] = 0
                candidates[row, col, :num] = False
                continue
            num = 0
            if board[row, col] != 0:
                # If we've reached the end of the board, we've found a solution
                # print(f'solved in {steps} steps')
                return steps
        # Try filling in this cell with each number from 1 to 9 until we find a valid number.
        # (Leaving off from the last number we tried.)
        for num in range(num + 1, 10):
            if candidates[row, col, num - 1]:
                board[row, col] = num
                stack.append((index, num))
                # Update candidates where necessary by checking the row, column, and box that contains the cell that just got filled in.
                row = -1 # Find the next empty cell.
                break
        else:
            if len(stack) == 0:
                # If we've tried all numbers and none of them are valid, the puzzle is unsolvable.
                # print(board)
                # print('unsolvable')
                # print(f'steps: {steps}')
                return -1
            # If we've tried all numbers and none of them are valid, backtrack.
            index, num = stack.pop()
            row = index // 9
            col = index % 9
            board[row, col] = 0




def has_single_solution(board: np.ndarray, max_steps: int = 512) -> bool:
    '''
    Return whether a Sudoku puzzle has a unique solution.
    '''
    sol1 = board.copy()
    steps = solve(sol1, max_steps=max_steps)
    if steps == -1:
        # print(f'has_single_solution: unsolvable puzzle')
        return False
    sol2 = board.copy()[::-1, ::-1]
    steps += solve(sol2, max_steps=max_steps)
    # print(f'has_single_solution: {steps} total steps')
    sol2 = sol2[::-1, ::-1]
    return np.array_equal(sol1, sol2)


def removable_givens(board: np.ndarray, max_steps: int = 512) -> Iterable[tuple[int, int]]:
    # Find the givens that can be removed while still having a unique solution.
    for row in range(9):
        for col in range(9):
            if board[row, col] == 0:
                continue
            val = board[row, col]
            board[row, col] = 0
            if has_single_solution(board, max_steps=max_steps):
                yield row, col
            board[row, col] = val

def reduced_forms(board: np.ndarray, max_steps: int = 512) -> Iterable[np.ndarray]:
    # Try to remove numbers from the puzzle while keeping it solvable and having only one solution.
    givens = list(removable_givens(board, max_steps=max_steps))
    if len(givens) == 0:
        yield board.copy()
    for row, col in givens:
        v = board[row, col]
        board[row, col] = 0
        yield from reduced_forms(board, max_steps=max_steps)
        board[row, col] = v



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

    steps = solve(board)
    print(board)
    print(f'solved: {is_solved(board)}')
    if steps == -1:
        print('unsolvable puzzle')
    else:
        print(f'solved in {steps} steps')

