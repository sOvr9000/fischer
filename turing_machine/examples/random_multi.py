
import numpy as np
from fischer.turing_machine import TuringMachineMultiTape



tm = TuringMachineMultiTape(4)

print(tm.transition_table.shape)

num_states = 3
num_symbols = 4

tm.set_tapes([np.random.randint(0, num_symbols, size=np.random.randint(4, 9)).tolist() for _ in range(len(tm.heads))])
tm.set_initial_state(0)

new_table = np.empty((num_states, *tuple([num_symbols] * len(tm.heads)), tm.transition_table.shape[-1]), dtype=int)
print(new_table.shape)

new_table[:, :, :, :, :, 0] = np.random.randint(0, num_states, size=new_table.shape[:-1]) # set new states
new_table[:, :, :, :, :, 1:5] = np.random.randint(0, num_symbols, size=new_table.shape[:-1] + (4,)) # set new symbols
new_table[:, :, :, :, :, 5:] = np.random.randint(-1, 2, size=new_table.shape[:-1] + (4,)) # set head movements
new_table[:, :, :, :, :, 0] = np.where(np.random.rand(*new_table.shape[:-1]) < .005, -1, new_table[:, :, :, :, :, 0]) # set some of the new state values to -1
# print(new_table)

tm.set_transition_table(new_table)

while not tm.halted:
    tm.step()
    print(tm)


