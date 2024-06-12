
import numpy as np
from graph_tools import Graph
from typing import Iterable

from .behaviors import PlayerBehavior



class CellGame:
    '''
    An abstract, non-spatial representation of a game inspired by Eufloria.
    '''
    def __init__(self, num_players: int = 2, starter_mass: int = 32, starter_growth_rate: int = 2, starter_max_mass: int = 256):
        # The format of the player_data array is undefined for now, but it will contain special information about each player, etc.
        self.player_data = np.zeros((num_players, 0))

        # The format of the cell_data array is as follows:
        # [owner, current_mass, grow_rate, max_mass]
        # If owner is 0, the cell is unowned.  Otherwise, it is one more than the index of the corresponding player in player_data.
        self.cell_data = np.zeros((0, 4), dtype=np.uint16)

        # The cell_graph list contains a list for each cell, which contains the indices of the cells that are connected to it.
        self.cell_graph = Graph(directed=True)

        # The format of the projectiles array is as follows:
        # [owner, source_cell, target_cell, mass, distance]
        self.projectiles = np.zeros((0, 5), dtype=np.uint16)

        self.owned_cells: list[int] = []

        self.total_steps = 0

        self.starter_mass = starter_mass
        self.starter_growth_rate = starter_growth_rate
        self.starter_max_mass = starter_max_mass
    
    @property
    def num_players(self) -> int:
        '''
        Return the number of players in the game.
        '''
        return self.player_data.shape[0]

    @property
    def num_cells(self) -> int:
        '''
        Return the number of cells in the game.
        '''
        return self.cell_data.shape[0]
    
    @property
    def num_projectiles(self) -> int:
        '''
        Return the number of projectiles in the game.
        '''
        return self.projectiles.shape[0]

    def update(self):
        '''
        Update the game state.  Advance the simulation by one time step.
        '''
        self.update_cells()
        self.update_projectiles()
        self.total_steps += 1

    def update_cells(self):
        '''
        Update the cells in the game.
        '''
        # Add growth rate to current mass.
        self.cell_data[self.owned_cells, 1] += self.cell_data[self.owned_cells, 2]
        # Limit mass to max mass.
        self.cell_data[self.owned_cells, 1] = np.minimum(self.cell_data[self.owned_cells, 1], self.cell_data[self.owned_cells, 3])
    
    def update_projectiles(self):
        '''
        Update the projectiles in the game.
        '''
        # Move projectiles toward target cell.
        self.projectiles[:, 4] -= 1
        # Check for collisions.  Note that the distance for each projectile is a uint16, so it will wrap around if it goes negative.
        collisions = np.where(self.projectiles[:, 4] == 0)[0]
        for i in collisions:
            projectile_owner = self.projectiles[i, 0]
            projectile_mass = self.projectiles[i, 3]
            target_cell = self.projectiles[i, 2]
            target_cell_owner = self.cell_data[self.projectiles[i, 2], 0]
            if projectile_owner == target_cell_owner:
                # If the target cell is owned by the projectile owner, add the mass of the projectile to the target cell.
                # Note that uint8 overflow/negatives must be avoided.
                self.cell_data[target_cell, 1] += min(65535 - projectile_mass, projectile_mass)
            else:
                # If the target cell is unowned, take the mass of the projectile away from the target cell.
                # Note that uint8 overflow/negatives must be avoided.
                new_mass = self.cell_data[target_cell, 1] - min(self.cell_data[target_cell, 1], projectile_mass)
                # If the target cell would have negative mass, change the owner to the projectile owner and set the mass to the difference between the projectile mass and the target cell mass.
                if projectile_mass > self.cell_data[target_cell, 1]:
                    self.cell_data[target_cell, 1] = projectile_mass - self.cell_data[target_cell, 1]
                    self.set_cell_owner(target_cell, projectile_owner)
                    # If the target cell was unowned, add it to the list of owned cells.
                    if target_cell not in self.owned_cells:
                        self.owned_cells.append(target_cell)
                else:
                    self.cell_data[target_cell, 1] = new_mass
        # Limit mass to max mass.
        self.cell_data[self.projectiles[collisions, 2], 1] = np.minimum(self.cell_data[self.projectiles[collisions, 2], 1], self.cell_data[self.projectiles[collisions, 2], 3])
        # Remove projectiles that have reached their target.
        self.remove_projectiles(collisions)

    def add_cell(self, max_mass: int, growth_rate: int, owner: int = 0) -> int:
        '''
        Add a cell to the game with `max_mass` maximum mass, `growth_rate` growth rate, and player `owner - 1` as the owner.

        Use `owner = 0` for an unowned cell.

        Return the index of the new cell.
        '''
        if self.cell_data.shape[0] >= 65536:
            raise ValueError('The maximum number of cells (65536) has been reached.')
        index = self.cell_data.shape[0]
        self.cell_data = np.vstack((self.cell_data, np.array([owner, max_mass // 8, growth_rate, max_mass], dtype=np.uint16)))
        # Update the list of owned cells.
        if owner > 0:
            if index not in self.owned_cells:
                self.owned_cells.append(index)
        self.cell_graph.add_vertex(index)
        return index

    def add_cells(self, max_mass: np.ndarray, growth_rates: np.ndarray, owners: np.ndarray = None) -> np.ndarray:
        '''
        Add multiple cells to the game with `max_mass` maximum mass, `growth_rates` growth rates, and players `owners - 1` as the owners.

        Use `owners = None` for unowned cells.

        Return the indices of the new cells.
        '''
        if max_mass.dtype != np.uint16:
            raise ValueError('max_mass must be of type np.uint16.')
        if growth_rates.dtype != np.uint16:
            raise ValueError('growth_rates must be of type np.uint16.')
        if owners is not None:
            if owners.dtype != np.uint16:
                raise ValueError('owners must be of type np.uint16.')
        else:
            owners = np.zeros((max_mass.shape[0], 1), dtype=np.uint16)
        if max_mass.shape[0] != growth_rates.shape[0]:
            raise ValueError('max_mass and growth_rates must have the same length.')
        if len(max_mass.shape) == 1:
            max_mass = max_mass.T
        if len(growth_rates.shape) == 1:
            growth_rates = growth_rates.T
        if len(owners.shape) == 1:
            owners = owners.T
        if len(max_mass.shape) > 2:
            raise ValueError('max_mass must be a 1D or 2D array.')
        if len(growth_rates.shape) > 2:
            raise ValueError('growth_rates must be a 1D or 2D array.')
        if len(owners.shape) > 2:
            raise ValueError('owners must be a 1D or 2D array.')
        if max_mass.shape[0] != growth_rates.shape[0] or max_mass.shape[0] != owners.shape[0]:
            raise ValueError('max_mass, growth_rates, and owners must have the same length.')
        if self.cell_data.shape[0] + max_mass.shape[0] > 65536:
            raise ValueError('Cannot add multiple cells as the maximum number of cells (65536) would be exceeded.')
        indices = np.arange(self.cell_data.shape[0], self.cell_data.shape[0] + max_mass.shape[0])
        self.cell_data = np.vstack((self.cell_data, np.hstack((owners, max_mass // 8, growth_rates, max_mass))))
        # Update the list of owned cells.
        self.owned_cells += indices[owners > 0].tolist()
        self.cell_graph.add_vertices(*indices)
        return indices

    def set_cell_owner(self, cell: int, owner: int):
        '''
        Set the owner of the cell with index `cell` to player `owner - 1`.
        '''
        if owner == 0:
            if cell in self.owned_cells:
                self.owned_cells.remove(cell)
        else:
            if cell not in self.owned_cells:
                self.owned_cells.append(cell)
        if self.cell_data[cell, 0] == 0 and owner > 0:
            if self.total_steps == 0:
                # If the cell was unowned and is now owned, and if no time steps have been simulated, set the cell as a "starter" cell.  (This balances early gameplay.)
                self.cell_data[cell, 1] = self.starter_mass
                self.cell_data[cell, 2] = self.starter_growth_rate
                self.cell_data[cell, 3] = self.starter_max_mass
        self.cell_data[cell, 0] = owner

    def is_cell_connected(self, from_cell: int, to_cell: int) -> bool:
        '''
        Returns True if there is a connection from `from_cell` to `to_cell`, and False otherwise.
        '''
        return self.cell_graph.has_edge(from_cell, to_cell)
    
    def connect_cell(self, from_cell: int, to_cell: int, projectile_travel_time: int):
        '''
        Connect cell `from_cell` to cell `to_cell` with an edge of weight `projectile_travel_time`, which determines how many steps that a projectile fired from `from_cell` will take to reach `to_cell`.
        '''
        if from_cell == to_cell:
            raise ValueError('from_cell and to_cell must be different.')
        if self.cell_graph.has_edge(from_cell, to_cell):
            return
        self.cell_graph.add_edge(from_cell, to_cell)
        self.cell_graph.set_edge_weight(from_cell, to_cell, projectile_travel_time)
    
    def disconnect_cell(self, from_cell: int, to_cell: int):
        '''
        Disconnect cell `from_cell` from cell `to_cell`.
        '''
        if from_cell == to_cell:
            raise ValueError('from_cell and to_cell must be different.')
        if not self.cell_graph.has_edge(from_cell, to_cell):
            return
        self.cell_graph.remove_edge(from_cell, to_cell)

    def update_cell_projectile_travel_time(self, from_cell: int, to_cell: int, projectile_travel_time: int):
        '''
        Update the travel time of a projectile from `from_cell` to `to_cell` to `projectile_travel_time`.
        '''
        if from_cell == to_cell:
            raise ValueError('from_cell and to_cell must be different.')
        if not self.cell_graph.has_edge(from_cell, to_cell):
            raise ValueError('There is no edge from the source cell to the target cell.')
        self.cell_graph.set_edge_weight(from_cell, to_cell, projectile_travel_time)

    def get_projectile_travel_time(self, source_cell: int, target_cell: int) -> int:
        '''
        Returns the time (number of simulation steps) it takes for a projectile to travel from `source_cell` to `target_cell`.

        It is based on the cell graph's edge weight, and the returned value is never less than 1.
        '''
        return max(1, int(.5 + self.cell_graph.get_edge_weight(source_cell, target_cell)))

    def can_add_projectile(self, source_cell: int, target_cell: int, mass: int) -> bool:
        '''
        Returns True if a projectile with `mass` mass can be launched from `source_cell` to `target_cell`, and False otherwise.
        '''
        if not self.is_cell_connected(source_cell, target_cell):
            return False
        if self.cell_data[source_cell, 1] < mass:
            return False
        return True

    def add_projectile(self, source_cell: int, target_cell: int, mass: int):
        '''
        Add a projectile with `mass` mass, `source_cell` source cell, and `target_cell` target cell.

        The owner of the projectile is the owner of the source cell.
        '''
        if not self.is_cell_connected(source_cell, target_cell):
            raise ValueError('There is no edge from the source cell to the target cell.')
        if self.cell_data[source_cell, 1] < mass:
            raise ValueError('The source cell does not have enough mass to launch the projectile.')
        self.projectiles = np.vstack((self.projectiles, np.array([self.cell_data[source_cell, 0], source_cell, target_cell, mass, self.get_projectile_travel_time(source_cell, target_cell)], dtype=np.uint16)))
        # Subtract the mass of the projectile from the source cell.
        self.cell_data[source_cell, 1] -= mass

    def remove_projectiles(self, indices: np.ndarray):
        '''
        Remove the projectiles at indices `indices`.
        '''
        if len(indices) == 0:
            return
        self.projectiles = np.delete(self.projectiles, indices, axis=0)

    def is_cell_graph_playable(self) -> bool:
        '''
        Return True if there exists a path from each cell to every other cell, and False otherwise.
        '''
        # TODO: Do not use Dijkstra's algorithm to check for connectivity because it is very slow on large graphs, when instead we can use a faster algorithm such as DFS.
        for i in range(self.cell_data.shape[0]):
            _, prev = self.cell_graph.dijkstra(i)
            for j in range(self.cell_data.shape[0]):
                if i == j:
                    continue
                if not prev[j]:
                    return False
        return True

    def get_path(self, from_cell: int, to_cell: int) -> list[int]:
        '''
        Return a list of cells representing the fastest path (or the first one found) from `from_cell` to `to_cell`, using fewest time steps.

        If there is no path, return an empty list.

        The returned list starts with `from_cell` and ends with `to_cell`.
        '''
        _, prev = self.cell_graph.dijkstra(from_cell)
        path = []
        current = to_cell
        while current != from_cell:
            path.append(current)
            candidates = prev[current]
            if len(candidates) == 0:
                return []
            current = candidates[0]
            if current in path:
                return []
        path.append(from_cell)
        return path

    def get_targetable_cells(self, from_cell: int) -> list[int]:
        '''
        Return a list of cells that can be targeted by a projectile launched from `source_cell`.
        '''
        return [to_cell for _, to_cell in self.cell_graph.out_edges(from_cell)]
    
    def get_targeted_by_cells(self, to_cell: int) -> list[int]:
        '''
        Return a list of cells that can target `to_cell`.
        '''
        return [from_cell for from_cell, _ in self.cell_graph.in_edges(to_cell)]
    
    def economic_value_heuristic(self, cell: int) -> float:
        '''
        Return a heuristic value representing the economic value of the cell's traits.

        The heuristic value is the product of the cell's growth rate and the square root of the mass capacity.
        '''
        return self.cell_data[cell, 3] ** .5 * self.cell_data[cell, 2]
    
    def military_value_heuristic(self, cell: int) -> float:
        '''
        Return a heuristic value representing the military value of the cell's traits.

        The heuristic value is `T / A ** 1.5` where `T` is the number of cells that this cell can target and `A` is the number of cells that can target this cell.

        The value of `A` should never be zero in a game where `CellGame.is_cell_graph_playable()` returns True.
        '''
        return len(self.get_targetable_cells(cell)) / max(1, len(self.get_targeted_by_cells(cell))) ** 1.5

    # def capture_cost_heuristic(self, from_cell: int, to_cell: int) -> float:
    #     '''
    #     Return a heuristic value representing the cost of capturing a cell.

    #     The heuristic value is the sum of the masses of the cells in the shortest path from `from_cell` to `to_cell`.
    #     '''
    #     path = self.get_path(from_cell, to_cell)
    #     if len(path) == 0:
    #         return np.inf
    #     return sum(self.cell_data[cell, 1] for cell in path)
    
    def all_cells_of_owner(self, owner: int) -> np.ndarray:
        '''
        Return a list of all cells owned by player `owner - 1`.
        '''
        return np.where(self.cell_data[:, 0] == owner)[0]
    
    def all_cells_of_player(self, player: int) -> np.ndarray:
        '''
        Return a list of all cells owned by player `player`.
        '''
        return self.all_cells_of_owner(player + 1)
    
    def get_cell_mass(self, cell: int) -> np.uint16:
        '''
        Return the mass of the cell.
        '''
        return self.cell_data[cell, 1]
    
    def get_cell_owner(self, cell: int) -> np.uint16:
        '''
        Return the owner of the cell.
        '''
        return self.cell_data[cell, 0]
    
    def get_cell_growth_rate(self, cell: int) -> np.uint16:
        '''
        Return the growth rate of the cell.
        '''
        return self.cell_data[cell, 2]
    
    def get_cell_max_mass(self, cell: int) -> np.uint16:
        '''
        Return the maximum mass of the cell.
        '''
        return self.cell_data[cell, 3]
    
    def run_player_behavior(self, behavior: PlayerBehavior, player: int):
        '''
        Run the behavior of player `player` in the game.
        '''
        action, args = behavior.choose_action(self, player)
        if action == 'pass':
            return
        elif action == 'send':
            if self.can_add_projectile(*args):
                self.add_projectile(*args)
        else:
            raise ValueError(f'Invalid action: {action}')

    # def get_attacking_cells_of_owner(self, owner: int) -> np.ndarray:
    #     '''
    #     Return a list of all cells owned by player `owner - 1` that can attack cells that are either unowned or owned by other players.
    #     '''
    #     return np.array([cell for cell in self.all_cells_of_owner(owner) if len(self.get_targetable_cells(cell)) > 0])

    def __str__(self):
        # Neatly format each cell's data.
        cell_data_str = '\t CELL | OWNER  MASS  GROW   MAX [ CON1  CON2 ...]\n'
        cell_data_str += '\t------|------------------------------------------\n'
        for i in range(self.cell_data.shape[0]):
            cell_data_str += f'\t{i:5d} | ' + ' '.join(
                f'{self.cell_data[i, j]:5d}'
                for j in range(self.cell_data.shape[1])
            )
            if self.cell_graph.out_degree(i) > 0:
                cell_data_str += ' [' + ' '.join(
                    f'{j:5d}'
                    for _, j in self.cell_graph.out_edges(i)
                ) + ']\n'
            else:
                cell_data_str += '\n'
        # Neatly format each projectile's data.
        projectile_str = '\t PROJ | OWNER  FROM    TO  MASS  DIST\n'
        projectile_str += '\t------|------------------------------\n'
        for i in range(self.projectiles.shape[0]):
            projectile_str += f'\t{i:5d} | ' + ' '.join(
                f'{self.projectiles[i, j]:5d}'
                for j in range(self.projectiles.shape[1])
            ) + '\n'
        return f'Cells:\n{cell_data_str}Projectiles:\n{projectile_str}'

