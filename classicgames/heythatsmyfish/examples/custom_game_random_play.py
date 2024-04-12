
import numpy as np
from fischer.classicgames.heythatsmyfish import HeyThatsMyFish, TilePrototype



def main():
    game = HeyThatsMyFish(
        num_players=5,
        num_penguins_per_player=1,
        num_moves_per_turn=2,
        max_fish_in_starting_tile=2,
        tile_set=[TilePrototype(num_fish=1)] * 12 + [TilePrototype(num_fish=2)] * 6 + [TilePrototype(num_fish=4)] * 3 + [TilePrototype(num_fish=8)] * 3,
        # leave tile_layout=None for a randomly generated tile layout with uniform distribution over all possible tile layouts with the given tile_set
    )
    print(game)
    print(game.tile_layout)
    input('Press any key to continue...')
    while not game.game_over:
        moves = list(game.valid_moves())
        move = moves[np.random.randint(len(moves))]
        if game.game_phase == 0:
            game.place_penguin(*move)
        else:
            game.play_move(*move)
        print(move)
        print(game.penguins)
        print(game)
        input('Press any key to continue...')
    print('Game over. Final scores:')
    print(game.caught_fish)



if __name__ == '__main__':
    main()
