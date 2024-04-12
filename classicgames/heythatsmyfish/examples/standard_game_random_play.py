
import numpy as np
from fischer.classicgames.heythatsmyfish import HeyThatsMyFish



def main():
    game = HeyThatsMyFish()
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
