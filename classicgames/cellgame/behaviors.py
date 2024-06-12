
import numpy as np



class PlayerBehavior:
    def choose_action(self, game: 'CellGame', as_player: int) -> tuple[str, tuple]:
        '''
        Return a tuple of the form `(action, args)` where action is a string that is either 'pass' or 'send'.

        If action is 'send', then args is a tuple of the form `(from_cell, to_cell, mass)` where `from_cell`, `to_cell`, and `mass` are integers (which are converted to uint16s).
        '''
        raise NotImplementedError()



class RandomPlayerBehavior(PlayerBehavior):
    def choose_action(self, game: 'CellGame', as_player: int) -> tuple[str, tuple]:
        if np.random.randint(2):
            return 'pass', ()
        else:
            owned_cells = game.all_cells_of_player(as_player)
            possible_sends = [
                (cell1, cell2)
                for cell1 in owned_cells
                for cell2 in game.get_targetable_cells(cell1)
            ]
            if not possible_sends:
                return 'pass', ()
            cell1, cell2 = possible_sends[np.random.randint(len(possible_sends))]
            mass = 1 + np.random.randint(game.get_cell_mass(cell1))
            return 'send', (cell1, cell2, mass)



# class HeuristicPlayerBehavior(PlayerBehavior):
#     def choose_action(self, game: 'CellGame', as_player: int) -> tuple[str, tuple]:
#         targetable_cells = game.get_targetable_cells_by_player(as_player)
        
#         return 'send', (cell1, cell2, mass)
