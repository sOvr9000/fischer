
import numpy as np
from .htmf import HeyThatsMyFish
from easyAI import AI_Player, Human_Player, TwoPlayerGame
from typing import Union



__all__ = ['HeyThatsMyFishMM']

class HeyThatsMyFishMM(TwoPlayerGame):
    def __init__(self, players: list[Union[Human_Player, AI_Player]], game: HeyThatsMyFish):
        self.players = players
        self.current_player = 1
        self.game = game
    def possible_moves(self) -> list:
        return list(self.game.valid_moves())
    def make_move(self, move: Union[tuple[int, int], tuple[int, int, int, int]]):
        if self.game.game_phase == 0:
            self.game.place_penguin(*move)
        else:
            self.game.play_move(*move)
    def is_over(self) -> bool:
        return self.game.game_over
    def win(self) -> bool:
        return self.game.caught_fish[self.game.turn] > max(s for i, s in enumerate(self.game.caught_fish) if i != self.game.turn)
    def show(self):
        print(self.game)
    def scoring(self):
        return 999 if self.win() else self.game.caught_fish[self.game.turn] - max(s for i, s in enumerate(self.game.caught_fish) if i != self.game.turn)
    def update_player_turn(self):
        self.current_player = 2 if self.turn != 0 else 1

