
import numpy as np
from typing import Union



class DuelGame:
    '''
    Models a two-player game where each player takes turns playing moves.

    With `DuelGame.skip_turn()`, it is possible skip the opponent's turn to allow players to play multiple moves in a row before letting the other player play another move.
    For example, in "Dots and Lines", players get a free turn by enclosing a 1x1 box and claiming a point for themselves.
    For another example, in the Kalah variant of "Mancala", a player gets a free turn by distributing the last stone/pebble/piece from their chosen pit/bowl/slot in their Mancala/home slot.

    You must override `DuelGame.reset()`, `DuelGame.step()`, `DuelGame.get_state()`, `DuelGame.get_static_reward()`.
    
    For further control, you may override `DuelGame.is_terminal()`, `DuelGame.get_action_mask()`, `DuelGame.get_valid_actions()`, and `DuelGame.get_heuristic_action()`.

        If the environment actions are multidimensional, then override `get_valid_actions()` but not `get_action_mask()`.  Otherwise, override `get_action_mask()` but not `get_valid_actions()`.

    If using `tournament()` from `fischer.dql.duelenv.tournament`, then `DuelGame.get_result()` must also be implemented.
    '''
    def __init__(self):
        self.sim_index: int = -1
    def reset(self):
        '''
        Reset the environment.
        '''
        raise NotImplementedError
    def step(self, action: Union[int, np.ndarray]):
        '''
        Take one step in the game given action `action`.

        If a player gets to move again before letting the other player move, then ensure that you call `DuelGame.skip_turn()` at the end of this function.

        This function is called for either player, regardless of which player is the training agent.
        
        That said, one step in the environment for the training agent consists of both the training agent's most recent action and all of the responses from its opponent as a single transition from one state to another, which is used for training during experience replay.
        '''
        raise NotImplementedError
    def get_state(self) -> np.ndarray[float]:
        '''
        Return the current relative state of the environment with values normalized so that the model is capable of interpreting the state.

        The returned state must be relative such that it is "flipped" if the current player to move is the second player.
        '''
        raise NotImplementedError
    def get_action_mask(self) -> np.ndarray[bool]:
        '''
        Return a boolean mask over the action space that describes which actions are legal in the current state of the environment, from the perspective of the current player to move.

        By default, this function returns `None`, which means all actions are assumed to be legal.

        This needs to be overridden when the model has single-dimensional outputs.  In this case, `DuelGame.get_valid_actions()` does not need to be overridden.
        '''
        return None
    def get_valid_actions(self) -> list[tuple]:
        '''
        Return a list of the possible actions which are each a tuple of integers.

        This only needs to be overridden when the model has multidimensional outputs.  In this case, `DuelGame.get_action_mask()` does not need to be overridden.
        '''
        raise NotImplementedError
    def get_heuristic_action(self) -> Union[int, np.ndarray]:
        '''
        Return the action that is likely to be best based on human knowledge of the environment, using quick and basic calculations.

        If returned None, then either the agent's prediction is used or a random action is selected, depending on the state of the simulation.
        '''
        return None
    def get_static_reward(self) -> float:
        '''
        Return an absolute evaluation of the current state, where positive values correspond to the first player having an advantage and negative values correspond to the second player having an advantage.

        The first differences of the values returned by this function are used as the reward function in the DQL algorithm.
        '''
        raise NotImplementedError
    def is_terminal(self) -> bool:
        '''
        Return whether the game is in a terminal state.

        By default, this function returns `False`.
        '''
        return False
    def get_result(self) -> int:
        '''
        Return the winner of the game.  If the first player won, then return `1`.  If the second player won, then return `0`.  If drawn, then return `-1`.
        '''
        raise NotImplementedError
    def toggle_turn(self):
        self.set_turn(not self.get_turn())
    def set_turn(self, turn: bool):
        self.player1_turn = turn
    def get_turn(self) -> bool:
        '''
        Return whether it's the first player's turn.
        '''
        return self.player1_turn
    def _reset(self):
        self.total_steps = 0
        self.player1_turn = True # default to True, but reset() can override this
        self.accumulated_reward = 0
        self.reset()
    def _step(self, action: Union[int, np.ndarray]):
        self.step(action)
        self.total_steps += 1
    def _get_result(self) -> int:
        assert self.is_terminal()
        return self.get_result()
    def _get_valid_actions(self) -> list[tuple]:
        return self.get_valid_actions()
