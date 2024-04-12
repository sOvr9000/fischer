

import numpy as np

from typing import Any, Iterator



class LinkMDiffN:
    '''
    In a grid of size `size`, create links between adjacent numbers.
    
    A link between adjacent numbers `a` and `b` can only be created if `abs(a - b) <= max_diff`.  A chain/group of linked numbers must have exactly `chain_length` numbers in it (`chain_length - 1` links).
    
    A grid cell cannot have more than two links to/from it (links are undirected).

    Goal: Link every grid cell to at least one other grid cell, and have all chains consist of exactly `chain_length` grid cells.
    '''
    def __init__(self, size: tuple[int, int] = (6, 6), chain_length: int = 5, max_diff: int = 2):
        self.size = size
        self.chain_length = chain_length
        self.max_diff = max_diff
        # The grid array has shape (size[0], size[1]), and represents the numbers in each grid cell.
        self.grid = np.empty(self.size, dtype=int)
        # The links array has shape (size[0], size[1], 2), where the third dimension represents links from a cell going upward (index 0) or rightward (index 1).
        self.links = np.empty((*self.size, 2), dtype=bool)
    def randomize_grid(self, max_n: int = None):
        '''
        Set the grid to random numbers from `1` to `max_n`, and remove all currently existing links.

        There is not guaranteed to be a complete solution to the new puzzle, but there will always be at least one best partial solution which links the most cells.
        '''
        if max_n is None:
            max_n = 2 * self.max_diff
        self.grid[:] = np.random.randint(1, max_n + 1, size=self.size)
        self.links[:] = False
    def get_adjacent_cell(self, i: int, j: int, direction: int) -> tuple[int, int]:
        '''
        Return the coordinates of the cell adjacent to the cell at `(i, j)` in the specified direction.
        '''
        if direction == 0:
            return i + 1, j
        if direction == 1:
            return i, j + 1
        if direction == 2:
            return i - 1, j
        if direction == 3:
            return i, j - 1
        raise ValueError(f'Invalid direction: {direction}')
    def is_index_valid(self, i: int, j: int) -> bool:
        '''
        Return whether the cell at `(i, j)` is within the grid.
        '''
        return 0 <= i < self.size[0] and 0 <= j < self.size[1]
    def get_adjacent_cells(self, i: int, j: int) -> Iterator[tuple[int, int, int]]:
        '''
        Return an iterator over the coordinates of all cells adjacent to the cell at `(i, j)`.

        The returned tuples are of the form `(i2, j2, d)`, where `(i2, j2)` are the coordinates of the adjacent cell, and `d` is the direction from the cell at `(i, j)` to the cell at `(i2, j2)`.
        '''
        for direction in range(4):
            i2, j2 = self.get_adjacent_cell(i, j, direction)
            if self.is_index_valid(i2, j2):
                yield i2, j2, direction
    def get_direction(self, i: int, j: int, i2: int, j2: int) -> int:
        '''
        Return the direction from the cell at `(i, j)` to the cell at `(i2, j2)`.

        Return `-1` if the cells are not adjacent.
        '''
        if abs(i - i2) + abs(j - j2) != 1:
            return -1
        if i2 > i:
            return 0
        if j2 > j:
            return 1
        if i2 < i:
            return 2
        return 3
    def get_total_links_to_cell(self, i: int, j: int) -> int:
        '''
        Return the total number of links to the cell at `(i, j)`.
        '''
        s = self.links[i, j].sum()
        s += self.links[i - 1, j, 0] if i > 0 else 0
        s += self.links[i, j - 1, 1] if j > 0 else 0
        return s
    def cells_are_linked(self, i: int, j: int, i2: int, j2: int) -> bool:
        '''
        Return whether the cells at `(i, j)` and `(i2, j2)` are linked.
        '''
        d = self.get_direction(i, j, i2, j2)
        if d == -1:
            return False
        if d >= 2:
            return self.cells_are_linked(i2, j2, i, j)
        return self.links[i, j, d]
    def cells_in_chain(self, i: int, j: int) -> Iterator[tuple[int, int]]:
        '''
        Return an iterator over the coordinates of all cells in the chain that the cell at `(i, j)` is a part of.
        '''
        visited = [i*self.size[1] + j]
        yield from self._dfs_cells_in_chain(i, j, visited)
    def _dfs_cells_in_chain(self, i: int, j: int, visited: list[int]) -> Iterator[tuple[int, int]]:
        visited.append(i*self.size[1] + j)
        yield i, j
        for i2, j2, _ in self.get_adjacent_cells(i, j):
            if self.cells_are_linked(i, j, i2, j2) and i2*self.size[1] + j2 not in visited:
                yield from self._dfs_cells_in_chain(i2, j2, visited)
    def can_add_link(self, i: int, j: int, i2: int, j2: int) -> bool:
        '''
        Return whether a link can be added between the two grid cells.
        '''
        # Check that the two cells are adjacent.
        if abs(i - i2) + abs(j - j2) != 1:
            return False
        # Check that the two cells have numbers that are close enough to each other.
        if abs(self.grid[i, j] - self.grid[i2, j2]) > self.max_diff:
            return False
        # Reassign argument values such that i2 > i or j2 > j.
        # This puts the problem in the context of the upward or rightward direction so that indexing the links array is easier.
        if i2 < i or j2 < j:
            i, j, i2, j2 = i2, j2, i, j
        # Check that the two cells are not already linked.
        if self.links[i, j, j2 - j]:
            return False
        # Check that the two cells do not have more than two links.
        if self.get_total_links_to_cell(i, j) >= 2 or self.get_total_links_to_cell(i2, j2) >= 2:
            return False
        # Check that the two cells do not belong to the same chain.
        chain = list(self.cells_in_chain(i, j))
        if any(self.cells_are_linked(i2, j2, *cell) for cell in chain):
            return False
        # Check that the resulting chain will not be too long.
        chain2 = list(self.cells_in_chain(i2, j2))
        if len(chain) + len(chain2) >= self.chain_length:
            return False
        return True
    def add_link(self, i: int, j: int, i2: int, j2: int) -> None:
        '''
        Add a link between the two grid cells.
        '''
        d = self.get_direction(i, j, i2, j2)
        if d >= 2:
            self.add_link(i2, j2, i, j)
            return
        self.links[i, j, d] = True
    def remove_link(self, i: int, j: int, i2: int, j2: int) -> None:
        '''
        Remove the link between the two grid cells.
        '''
        d = self.get_direction(i, j, i2, j2)
        if d >= 2:
            self.remove_link(i2, j2, i, j)
            return
        self.links[i, j, d] = False
    def get_solutions(self, partial: bool = True) -> Iterator[np.ndarray[bool]]:
        '''
        Perform a depth-first search to generate complete solutions to the puzzle, or include partial (incomplete) solutions if `partial = True`.
        '''
        yield from self._dfs_get_solutions(partial)
    def _dfs_get_solutions(self, partial: bool) -> Iterator[np.ndarray[bool]]:
        # If partial solutions are allowed, yield the current partial solution.
        if partial:
            yield self.links.copy()
        # Find each cell that is linked to at most one cell.
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self.get_total_links_to_cell(i, j) <= 1:
                    # Try to link the cell at (i, j) to an adjacent cell.
                    for i2, j2, _ in self.get_adjacent_cells(i, j):
                        if self.can_add_link(i, j, i2, j2):
                            self.add_link(i, j, i2, j2)
                            yield from self._dfs_get_solutions(partial)
                            self.remove_link(i, j, i2, j2)
    def get_best_partial_solution(self) -> Iterator[np.ndarray[bool]]:
        '''
        Iterate over `LinkMDiffN.get_solutions(partial=True)` and yield the best partial solution found so far, where the best partial solution is the one that links the most cells.
        '''
        best_solution = None
        best_solution_links = 0
        for solution in self.get_solutions(partial=True):
            links = solution.sum()
            if links > best_solution_links:
                best_solution = solution
                best_solution_links = links
                yield best_solution
    def get_score(self, solution: np.ndarray[bool]) -> int:
        '''
        Return the score of the solution, which is calculated as the number of cells linked.
        '''
        return solution.sum()
    def __str__(self) -> str:
        s = ''
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                s += f'{self.grid[i, j]}'
                if j < self.size[1] - 1:
                    s += ' '
            s += '\n'
        return s
