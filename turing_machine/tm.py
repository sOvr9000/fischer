
import numpy as np


class TuringMachine:
    '''
    A Turing machine implementation without restrictions on the length of the tape or the number of symbols and states.

    The tape is populated with zeros by default, representing the blank symbol.

    The halting state is -1.
    '''

    TAPE_ORIGIN_INDICATOR = '|'
    HEAD_INDICATOR_LEFT = '*'
    HEAD_INDICATOR_RIGHT = ' '

    def __init__(self):
        self.state = None
        self.head = 0
        self.tape = []
        self.transition_table = np.zeros((0, 0, 3), dtype=int)
        self.origin = 0 # used to keep track of how far the tape has been shifted as the head moves to the left
    def set_tape(self, tape: list[int], head_position: int = 0):
        self.tape = tape
        self.head = head_position
    def set_initial_state(self, state: int):
        self.state = state
    def set_transition_rule(self, state: int, symbol: int, new_state: int, new_symbol: int, move_forward: bool):
        if state >= self.transition_table.shape[0] or symbol >= self.transition_table.shape[1]:
            new_table = np.zeros((max(state+1, self.transition_table.shape[0]), max(symbol+1, self.transition_table.shape[1]), 3), dtype=int)
            new_table[:self.transition_table.shape[0], :self.transition_table.shape[1]] = self.transition_table
            self.transition_table = new_table
        self.transition_table[(state, symbol)] = new_state, new_symbol, int(move_forward) * 2 - 1
    def set_transition_table(self, table: np.ndarray):
        if table.dtype is not np.int_:
            raise ValueError('Transition table must be of integer type.')
        if len(table.shape) != 3:
            raise ValueError('Transition table must have 3 dimensions.')
        if table.shape[2] != 3:
            raise ValueError('Transition table must have size 3 on the third axis.')
        if np.any(table[:, :, 2] != -1) and np.any(table[:, :, 2] != 1):
            raise ValueError('The third element of each transition must be -1 or 1.')
        self.transition_table = table
    def step(self):
        if self.state == -1:
            return
        new_state, new_symbol, move = self.transition_table[self.state, self.tape[self.head]]
        self.tape[self.head] = new_symbol
        self.state = new_state
        self.head += move
        if self.head < 0:
            self.head = 0
            self.tape.insert(0, 0)
            self.origin += 1
        if self.head >= len(self.tape):
            self.tape.append(0)
    @property
    def halted(self) -> bool:
        return self.state == -1
    def __str__(self):
        s = f'State: {self.state if self.state != -1 else "-1 (halted)" if self.state == -1 else "None (unset)"}\n'
        for i, symbol in enumerate(self.tape):
            if i == self.origin:
                s += TuringMachine.TAPE_ORIGIN_INDICATOR
            if i == self.head:
                s += f' {TuringMachine.HEAD_INDICATOR_LEFT}{symbol}{TuringMachine.HEAD_INDICATOR_RIGHT} '
            else:
                s += f'  {symbol}  '
        return s


