
import numpy as np
from random import shuffle
from typing import Iterable



class EventHandler:
    def on_cell_added(self, cell: 'Cell'):
        pass
    def on_cell_removed(self, cell: 'Cell'):
        pass
    def on_cell_eaten(self, eater: 'Cell', eaten: 'Cell'):
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
        # If the cells are facing each other, they both attack each other as though the attacked cell defends itself first.
        if cell.get_last_move_direction() == (other_cell.get_last_move_direction() + 2) % 4:
            e = cell.energy
            cell.energy -= other_cell.genes[1] # defense attribute
            if not cell.is_alive:
                other_cell.energy += e * 3 // 4
                cell.world.event_handler.on_cell_eaten(other_cell, cell)
        if other_cell.is_alive:
            e = other_cell.energy
            other_cell.energy -= cell.genes[0] # offense attribute
            if not other_cell.is_alive:
                cell.energy += e * 3 // 4
                cell.world.event_handler.on_cell_eaten(cell, other_cell)

class CellCollisionBehaviorBreed(CellCollisionBehavior):
    def on_collision(self, cell: 'Cell', other_cell: 'Cell'):
        if cell.get_last_move_direction() != (other_cell.get_last_move_direction() + 2) % 4:
            return
        if gene_similarity(cell.genes, other_cell.genes) < 0.618:
            return
        if cell.energy < 128 or other_cell.energy < 128:
            return
        nx, ny = cell.world.get_random_nearby_open_position(cell.x, cell.y)
        if nx is None:
            return
        print('cells breeding')
        genes = crossover_genes(cell.genes, other_cell.genes)
        cell.energy -= 32
        other_cell.energy -= 8
        cell.world.add_cell(nx, ny, genes=genes)

class CellCollisionBehaviorComposite(CellCollisionBehavior):
    def __init__(self, behaviors: list):
        self.behaviors = behaviors
    def on_collision(self, cell: 'Cell', other_cell: 'Cell'):
        for behavior in self.behaviors:
            behavior.on_collision(cell, other_cell)

random_cell_movement_behavior = CellMovementBehaviorRandom()
attack_cell_collision_behavior = CellCollisionBehaviorAttack()
breed_cell_collision_behavior = CellCollisionBehaviorBreed()
composite_cell_collision_behavior = CellCollisionBehaviorComposite([
    attack_cell_collision_behavior,
    breed_cell_collision_behavior
])

def mutate_genes(genes: bytearray, mutation_rate: float = 0.01) -> bytearray:
    '''
    Randomly flip bits in the genes.  The mutation rate is the probability for each bit to be flipped.  For example, if the mutation rate is 0.01, then on average 1% of the bits will be flipped.
    '''
    genes = bytearray(genes)
    r = np.random.rand(len(genes) * 8)
    for k, g in enumerate(genes):
        x = 1
        for i in range(8):
            if r[k * 8 + i] < mutation_rate:
                g ^= x
            x <<= 1
        genes[k] = g
    return genes

def crossover_genes(genes1: bytearray, genes2: bytearray, mutation_rate: float = 0.01) -> bytearray:
    '''
    Randomly select bits from either `genes1` or `genes2` to create a new set of genes, and return a mutation of the new set of genes.
    '''
    genes = bytearray(len(genes1))
    r = np.random.rand(len(genes1) * 8)
    for k in range(len(genes1)):
        x = 1
        for i in range(8):
            if r[k * 8 + i] < 0.5:
                genes[k] |= genes1[k] & x
            else:
                genes[k] |= genes2[k] & x
            x <<= 1
    return genes

def get_gene_bits(genes: bytearray, from_idx: int = 0, to_idx: int = 1024) -> Iterable[int]:
    for k, g in enumerate(genes):
        if k * 8 + 8 <= from_idx:
            continue
        if k * 8 >= to_idx:
            break
        x = 1
        for i in range(8):
            if k * 8 + i < from_idx:
                continue
            if k * 8 + i >= to_idx:
                break
            yield (g & x) >> i
            x <<= 1

def gene_similarity(genes1: bytearray, genes2: bytearray) -> float:
    '''
    Return the similarity between two sets of genes as a value between 0 and 1, where 0 means the genes are completely different and 1 means the genes are identical.
    '''
    s = 0
    for b1, b2 in zip(get_gene_bits(genes1), get_gene_bits(genes2)):
        s += b1 == b2
    return s * .125 / len(genes1)



class Cell:
    def __init__(self, idx: int, world: 'CellLife', x: int, y: int, movement_behavior: CellMovementBehavior = None, collision_behavior: CellCollisionBehavior = None, genes: bytearray = None):
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
        if genes is None:
            genes = bytearray([np.random.randint(0, 256) for _ in range(128)])
        self.genes = genes
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
    @property
    def is_alive(self) -> bool:
        return self.energy > 0
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
        if other_cell is not None and other_cell.is_alive:
            self.collision_behavior.on_collision(self, other_cell)
            return
        self.world.grid[self.y, self.x] = -1
        self.x, self.y = nx, ny
        self.world.grid[self.y, self.x] = self.idx
        self.energy -= 1
    def update(self):
        if not self.is_alive:
            return
        d = self.movement_behavior.choose_direction(self)
        self.move(d)
        self.energy -= (self.genes[0] + self.genes[1]) // 128
    def __str__(self):
        return f'Cell({self.x}, {self.y}, {self.get_last_move_direction()}, {self.energy})'



class CellLife:
    def __init__(self, size: tuple[int, int], event_handler: EventHandler = None, cells_respawn: bool = True):
        self.cells = []
        self.grid = -np.ones(size, dtype=int) # map of grid positions to indices of `CellLife.cells`, defaulting to -1 when there is no cell
        if event_handler is None:
            event_handler = EventHandler()
        self.event_handler = event_handler
        self.cells_respawn = cells_respawn
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
    def add_cell(self, x: int, y: int, genes: bytearray = None) -> Cell:
        if not self.is_within_bounds(x, y):
            return None
        if self.grid[y, x] != -1:
            return None
        idx = len(self.cells)
        cell = Cell(idx, self, x, y, genes=genes)
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
            if self.cells_respawn:
                c = self.add_cell(*self.get_random_open_position(), genes=mutate_genes(cell.genes, mutation_rate=.25))
    def update(self):
        cell_indices = np.arange(len(self.cells))
        np.random.shuffle(cell_indices)
        for idx in cell_indices:
            self.cells[idx].update()
        for idx in range(len(self.cells)-1, -1, -1):
            if not self.cells[idx].is_alive:
                self.remove_cell(self.cells[idx])
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
    def get_random_open_position(self) -> tuple[int, int]:
        positions = list(self.open_positions())
        if not positions:
            return None, None
        return positions[np.random.randint(len(positions))]
    def get_random_nearby_open_position(self, x: int, y: int, r: int = 1) -> tuple[int, int]:
        positions = list(filter(lambda p: abs(x - p[0]) < r and abs(y - p[1]) < r, self.open_positions()))
        if not positions:
            return None, None
        return positions[np.random.randint(len(positions))]
