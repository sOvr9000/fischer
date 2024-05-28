
import numpy as np
from random import shuffle
from typing import Iterable



class EventHandler:
    def on_cell_added(self, cell: 'Cell'):
        pass
    def on_cell_removed(self, cell: 'Cell'):
        pass

class CellMovementBehavior:
    def choose_direction(self, cell: 'Cell') -> int:
        raise NotImplementedError

class CellMovementBehaviorRandom(CellMovementBehavior):
    def choose_direction(self, cell: 'Cell') -> int:
        return np.random.randint(4)

class CellCollisionBehavior:
    def on_collision(self, cell: 'Cell', other_cell: 'Cell'):
        '''
        Called when `cell` tries to step onto the same position as `other_cell`.
        '''
        raise NotImplementedError

class CellCollisionBehaviorAttack(CellCollisionBehavior):
    def on_collision(self, cell: 'Cell', other_cell: 'Cell'):
        '''TODO'''

random_cell_movement_behavior = CellMovementBehaviorRandom()
attack_cell_collision_behavior = CellCollisionBehaviorAttack()



class Cell:
    def __init__(self, idx: int, world: 'CellLife', x: int, y: int, movement_behavior: CellMovementBehavior = None, collision_behavior: CellCollisionBehavior = None):
        self.idx = idx
        self.world = world
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self._energy = np.uint8(255)
        if movement_behavior is None:
            movement_behavior = random_cell_movement_behavior
        self.movement_behavior = movement_behavior
        if collision_behavior is None:
            collision_behavior = attack_cell_collision_behavior
        self.collision_behavior = collision_behavior
    @property
    def energy(self) -> np.uint8:
        return self._energy
    @energy.setter
    def energy(self, energy: int):
        if energy < 0:
            energy = 0
        elif energy > 255:
            energy = 255
        self._energy = np.uint8(energy)
    def get_last_move_direction(self) -> int:
        '''
        Return the direction of the last move of the cell.  If the cell did not move in the last step, return -1.
        '''
        if self.x > self.prev_x:
            return 0
        elif self.x < self.prev_x:
            return 2
        elif self.y > self.prev_y:
            return 1
        elif self.y < self.prev_y:
            return 3
        else:
            return -1
    def move(self, d: int):
        self.prev_x = self.x
        self.prev_y = self.y
        if d == -1:
            return
        if self.get_last_move_direction() == (d+2)%4:
            return
        dx, dy = 0, 0
        if d == 0:
            dx = 1
        elif d == 1:
            dy = 1
        elif d == 2:
            dx = -1
        elif d == 3:
            dy = -1
        nx, ny = self.x + dx, self.y + dy
        if not self.world.is_within_bounds(nx, ny):
            return
        other_cell = self.world.get_cell(nx, ny)
        if other_cell is not None:
            self.collision_behavior.on_collision(self, other_cell)
            return
        self.world.grid[self.y, self.x] = -1
        self.x, self.y = nx, ny
        self.world.grid[self.y, self.x] = self.idx
    def update(self):
        d = self.movement_behavior.choose_direction(self)
        self.move(d)
        if self.energy == 0:
            self.world.remove_cell(self)
    def __str__(self):
        return f'Cell({self.x}, {self.y}, {self.get_last_move_direction()}, {self.energy})'



class CellLife:
    def __init__(self, size: tuple[int, int], event_handler: EventHandler = None):
        self.cells = []
        self.grid = -np.ones(size, dtype=int) # map of grid positions to indices of `CellLife.cells`, defaulting to -1 when there is no cell
        if event_handler is None:
            event_handler = EventHandler()
        self.event_handler = event_handler
    def get_cell(self, x: int, y: int) -> Cell:
        if not self.is_within_bounds(x, y):
            return None
        idx = self.grid[y, x]
        if idx == -1:
            return None
        return self.cells[idx]
    @property
    def size(self) -> tuple[int, int]:
        return self.grid.shape
    def is_within_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]
    def add_cell(self, x: int, y: int) -> Cell:
        if not self.is_within_bounds(x, y):
            return None
        if self.grid[y, x] != -1:
            return None
        idx = len(self.cells)
        cell = Cell(idx, self, x, y)
        self.cells.append(cell)
        self.grid[y, x] = idx
        self.event_handler.on_cell_added(cell)
        return cell
    def remove_cell(self, cell: Cell):
        if self.is_within_bounds(cell.x, cell.y):
            self.grid[cell.y, cell.x] = -1
        for other_cell in self.cells[cell.idx+1:]:
            other_cell.idx -= 1
            self.grid[other_cell.y, other_cell.x] = other_cell.idx
        if cell in self.cells:
            self.cells.remove(cell)
            self.event_handler.on_cell_removed(cell)
    def update(self):
        cell_indices = np.arange(len(self.cells))
        np.random.shuffle(cell_indices)
        for idx in cell_indices:
            self.cells[idx].update()
    def open_positions(self) -> Iterable[tuple[int, int]]:
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x] == -1:
                    yield x, y
    def neighborhood_positions(self, x: int, y: int, r: int) -> Iterable[tuple[int, int]]:
        for ny in range(y-r, y+r+1):
            if ny < 0:
                continue
            if ny >= self.grid.shape[0]:
                break
            for nx in range(x-r, x+r+1):
                if nx == x and ny == y:
                    continue
                if nx < 0:
                    continue
                if nx >= self.grid.shape[1]:
                    break
                yield nx, ny
    def neighborhood_cells(self, x: int, y: int, r: int) -> Iterable[Cell]:
        for nx, ny in self.neighborhood_positions(x, y, r):
            cell = self.get_cell(nx, ny)
            if cell is not None:
                yield cell


