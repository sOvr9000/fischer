
# Simple application of the genetic algorithm to this game.

import numpy as np
from fischer.classicgames.heythatsmyfish import HeyThatsMyFish, TilePrototype



class Agent:
    def __init__(self, dna: np.ndarray = None):
        if dna is None:
            dna = np.random.normal(0, 1, (60, 5))
        self.dna = dna

class GeneticSim:
    def __init__(self, population_size: int, num_games: int):
        self.population_size = population_size
        self.population = [Agent() for _ in range(self.population_size)]
        self.num_games = num_games
        self.matches: dict[tuple, HeyThatsMyFish] = {}
        self.tile_prots = [TilePrototype(num_fish=n) for n in range(1, 9)]
        self.player_indices = np.arange(self.population_size)
    def generate_matches(self):
        self.matches.clear()
        for _ in range(self.num_games):
            num_players = np.random.randint(2, 9)
            players = tuple(np.random.choice(self.player_indices, num_players, replace=False))
            num_penguins_per_player = np.random.randint(1, 5)
            num_moves_per_turn = np.random.randint(1, 4)
            scale = num_players * num_penguins_per_player * num_moves_per_turn + 16
            tile_set = [self.tile_prots[0]] * np.random.randint(scale, scale * 2) + [self.tile_prots[1]] * np.random.randint(scale)
            for _ in range(np.random.randint(scale // 2)):
                tile_set.append(np.random.choice(self.tile_prots[2:]))
            game = HeyThatsMyFish(num_players=num_players, num_penguins_per_player=num_penguins_per_player, num_moves_per_turn=num_moves_per_turn, max_fish_in_starting_tile=1, tile_set=tile_set)
            self.matches[players] = game
    def play_matches(self):
        pass



if __name__ == '__main__':
    sim = GeneticSim(256, 1024)
    sim.generate_matches()
    for players, match in sim.matches.items():
        print(players)
        print(match)
        input()


