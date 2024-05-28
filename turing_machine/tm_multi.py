# multi-tape Turing machine

import numpy as np



class TuringMachineMultiTape:
    '''
    A Turing machine implementation without restrictions on the length of the tape or the number of symbols and states.

    The tape is populated with zeros by default, representing the blank symbol.

    The halting state is -1.
    '''

    TAPE_ORIGIN_INDICATOR = '|'
    HEAD_INDICATOR_LEFT = '*'
    HEAD_INDICATOR_RIGHT = ' '

    def __init__(self, num_heads: int):
        self.state = None
        self.heads = [0] * num_heads
        self.tapes = [[] for _ in range(num_heads)]
        table_shape = tuple([0] * num_heads + [0, 1 + num_heads * 2])
        self.transition_table = np.zeros(table_shape, dtype=int)
        self.origins = [0] * num_heads # used to keep track of how far the tape has been shifted as the head moves to the left
    def set_tapes(self, tapes: list[list[int]], head_positions: list[int] = None):
        if head_positions is None:
            head_positions = [0] * len(tapes)
        self.tapes = tapes
        self.heads = head_positions
    def set_initial_state(self, state: int):
        self.state = state
    def set_transition_rule(self, state: int, symbols: tuple, new_state: int, new_symbols: tuple, moves: tuple):
        sy = max(symbols)
        if state >= self.transition_table.shape[0] or np.any(sy >= self.transition_table.shape[1:]):
            new_table = np.zeros((max(state+1, self.transition_table.shape[0]), max(sy+1, self.transition_table.shape[1]), 3), dtype=int)
            new_table[:self.transition_table.shape[0], :self.transition_table.shape[1]] = self.transition_table
            self.transition_table = new_table
        self.transition_table[(state, *symbols)] = new_state, *new_symbols, *moves
    def set_transition_table(self, table: np.ndarray):
        if len(table.shape) != 2 + len(self.heads):
            raise ValueError(f'Transition table must have {2 + len(self.heads)} dimensions.')
        if table.shape[-1] != 1 + 2 * len(self.heads):
            raise ValueError(f'Transition table must have size {1 + 2 * len(self.heads)} on the final axis.')
        self.transition_table = table
    def step(self):
        if self.state == -1:
            return
        indices = tuple([self.state, *map(lambda h: self.tapes[h][self.heads[h]], range(len(self.heads)))])
        r = self.transition_table[indices]
        new_state, *new_symbols = r
        moves = new_symbols[len(new_symbols) // 2:]
        new_symbols = new_symbols[:len(new_symbols) // 2]
        for k, (h, s, m) in enumerate(zip(self.heads, new_symbols, moves)):
            t = self.tapes[k]
            t[h] = s
            self.heads[k] += m
            if self.heads[k] < 0:
                self.heads[k] = 0
                t.insert(0, 0)
                self.origins[k] += 1
            if self.heads[k] >= len(t):
                t.append(0)
        self.state = new_state
    @property
    def halted(self) -> bool:
        return self.state == -1
    def __str__(self):
        s = f'State: {self.state if self.state != -1 else "-1 (halted)" if self.state == -1 else "None (unset)"}'
        for j, tape in enumerate(self.tapes):
            s += f'\nTape #{j+1}\n'
            for i, symbol in enumerate(tape):
                if i == self.origins[j]:
                    s += TuringMachineMultiTape.TAPE_ORIGIN_INDICATOR
                if i == self.heads[j]:
                    s += f' {TuringMachineMultiTape.HEAD_INDICATOR_LEFT}{symbol}{TuringMachineMultiTape.HEAD_INDICATOR_RIGHT} '
                else:
                    s += f'  {symbol}  '
        return s


