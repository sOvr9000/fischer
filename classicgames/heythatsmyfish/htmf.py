
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from typing import Generator, Union



__all__ = ['HeyThatsMyFish', 'random_tile_layout', 'TilePrototype', 'standard_tile_set']



adjacency_offsets = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 0),
    (0, -1),
    (1, -1),
]

adjacency_offsets_transposed = [
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
]



def random_tile_layout(tile_set: list['TilePrototype']) -> dict[tuple[int, int], int]:
    positions = [(0, 0)]
    open_positions = []
    for _ in range(len(tile_set) - 1):
        for i in range(6):
            x, y = positions[-1]
            if y % 2 == 0:
                dx, dy = adjacency_offsets[i]
            else:
                dx, dy = adjacency_offsets_transposed[i]
            nx, ny = x + dx, y+ dy
            if (nx, ny) not in open_positions and (nx, ny) not in positions:
                open_positions.append((nx, ny))
        i = np.random.randint(len(open_positions))
        nx, ny = open_positions[i]
        positions.append((nx, ny))
        del open_positions[i]
    indices = np.arange(len(tile_set))
    np.random.shuffle(indices)
    return dict(zip(positions, indices))



@dataclass
class TilePrototype:
    def __init__(self, num_fish: int):
        self.num_fish = num_fish



standard_tile_set = [TilePrototype(num_fish=3)] * 10 + [TilePrototype(num_fish=2)] * 20 + [TilePrototype(num_fish=1)] * 30

class HeyThatsMyFish:
    def __init__(self, num_players: int = 4, num_penguins_per_player: int = 2, num_moves_per_turn: int = 1, max_fish_in_starting_tile: int = 1, tile_set: list[TilePrototype] = None, tile_layout: dict[tuple[int, int], int] = None):
        self.num_players = num_players
        self.num_penguins_per_player = num_penguins_per_player
        self.num_moves_per_turn = num_moves_per_turn
        self.max_fish_in_starting_tile = max_fish_in_starting_tile
        if tile_set is None:
            tile_set = deepcopy(standard_tile_set)
        if tile_layout is None:
            if tile_set == standard_tile_set:
                tile_set = deepcopy(standard_tile_set)
                tile_layout: dict[tuple[int, int], int] = {}
                av = np.arange(len(tile_set))
                np.random.shuffle(av)
                for i, (x, y) in enumerate((_x, _y) for _y in range(8) for _x in range(_y%2, 8)):
                    tile_layout[(x, y)] = av[i]
            else:
                tile_layout = random_tile_layout(tile_set)
        self.tile_set = tile_set
        self.tile_layout = tile_layout
        self.turn = 0
        self.sub_turn = 0
        self.game_phase = 0
        self.game_over = False
        self.penguins = np.ones((self.num_players, self.num_penguins_per_player, 2), dtype=int) * 999999
        self.caught_fish = np.zeros(self.num_players, dtype=int)
        self.tile_layout_bounds = (
            (
                min(x for x, _ in self.tile_layout.keys()),
                min(y for _, y in self.tile_layout.keys()),
            ),
            (
                max(x for x, _ in self.tile_layout.keys()),
                max(y for _, y in self.tile_layout.keys()),
            ),
        )
    def swap_tiles(self, x1: int, y1: int, x2: int, y2: int):
        self.tile_layout[(x1, y1)], self.tile_layout[(x2, y2)] = self.tile_layout[(x2, y2)], self.tile_layout[(x1, y1)]
    def move_tile(self, from_x: int, from_y: int, to_x: int, to_y: int):
        self.tile_layout[(to_x, to_y)] = self.tile_layout[(from_x, from_y)]
        del self.tile_layout[(from_x, from_y)]
    def valid_moves(self) -> Generator[Union[tuple[int, int], tuple[int, int, int, int]], None, None]:
        if self.game_phase == 0:
            # Place penguins phase
            for (x, y), prot_index in self.tile_layout.items():
                prot = self.tile_set[prot_index]
                if prot.num_fish <= self.max_fish_in_starting_tile:
                    yield x, y
        else:
            # Move penguins phase
            for penguin_id in range(self.num_penguins_per_player):
                px, py = self.penguins[self.turn, penguin_id]
                for i in range(6):
                    x, y = px, py
                    for _ in range(1, 1024): # loop will likely break before finishing, unless the board is very large and the game is not near its end (for loops are faster than while loops)
                        dy = adjacency_offsets[i][1]
                        dx = (adjacency_offsets_transposed if y % 2 == 1 else adjacency_offsets)[i][0]
                        y += dy
                        x += dx
                        prot_index = self.tile_layout.get((x, y), -1)
                        if prot_index == -1:
                            break
                        yield px, py, x, y
    def place_penguin(self, x: int, y: int):
        for i in range(self.num_penguins_per_player):
            if self.penguins[self.turn, i, 0] == 999999:
                break
        else:
            return
        self.penguins[self.turn, i] = x, y
        del self.tile_layout[(x, y)]
        self.turn = (self.turn + 1) % self.num_players
        if not np.any(self.penguins == 999999):
            self.game_phase = 1
    def play_move(self, from_x: int, from_y: int, to_x: int, to_y: int):
        for penguin_id in range(self.num_penguins_per_player):
            if from_x == self.penguins[self.turn, penguin_id, 0] and from_y == self.penguins[self.turn, penguin_id, 1]:
                break
        else:
            return
        self.penguins[self.turn, penguin_id] = to_x, to_y
        prot_index = self.tile_layout[(to_x, to_y)]
        prot = self.tile_set[prot_index]
        self.caught_fish[self.turn] += prot.num_fish
        del self.tile_layout[(to_x, to_y)]
        self.sub_turn = (self.sub_turn + 1) % self.num_moves_per_turn
        if self.sub_turn == 0 or len(list(self.valid_moves())) == 0:
            for _ in range(self.num_players):
                self.turn = (self.turn + 1) % self.num_players
                if len(list(self.valid_moves())) > 0:
                    break
            else:
                self.game_over = True
    def __str__(self) -> str:
        s = [[' '] * (self.tile_layout_bounds[1][0] - self.tile_layout_bounds[0][0] + 1) for _ in range(self.tile_layout_bounds[1][1] - self.tile_layout_bounds[0][1] + 1)]
        for (x, y), prot_index in self.tile_layout.items():
            prot = self.tile_set[prot_index]
            s[y - self.tile_layout_bounds[0][1]][x - self.tile_layout_bounds[0][0]] = str(prot.num_fish)
        for player_id, penguins in enumerate(self.penguins):
            for penguin_id, (x, y) in enumerate(penguins):
                if x == 999999: continue
                s[y - self.tile_layout_bounds[0][1]][x - self.tile_layout_bounds[0][0]] = chr(65 + player_id) if penguin_id % 2 == 0 else chr(97 + player_id)
        return '\n\n'.join(
            ('  ' if i % 2 == 1 else '') + '   '.join(row)
            for i, row in enumerate(s[::-1])
        )
