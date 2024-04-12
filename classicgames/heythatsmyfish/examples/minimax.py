
import numpy as np
from fischer.classicgames.heythatsmyfish import HeyThatsMyFish, HeyThatsMyFishMM, TilePrototype
from easyAI import Human_Player, AI_Player, Negamax



def main():
    game = HeyThatsMyFish(
        num_players=5,
        num_penguins_per_player=1,
        num_moves_per_turn=2,
        max_fish_in_starting_tile=2,
        tile_set=[TilePrototype(num_fish=1)] * 12 + [TilePrototype(num_fish=2)] * 6 + [TilePrototype(num_fish=4)] * 3 + [TilePrototype(num_fish=8)] * 3,
        # leave tile_layout=None for a randomly generated tile layout with uniform distribution over all possible tile layouts with the given tile_set
    )
    ai = Negamax(4)
    mm = HeyThatsMyFishMM([Human_Player(), AI_Player(ai)], game)
    history = mm.play()



if __name__ == '__main__':
    main()
