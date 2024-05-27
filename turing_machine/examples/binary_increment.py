
from fischer.turing_machine import TuringMachine


def binary_increment(x: int) -> int:
    '''
    Increment a number by 1 using Turing machine logic on binary digits.
    '''
    tm = TuringMachine()
    tape = [int(d)+1 for d in bin(x)[2:]]
    tm.set_tape(tape, len(tape) - 1)
    tm.set_initial_state(0)
    tm.set_transition_rule(0, 2, 0, 1, False)
    tm.set_transition_rule(0, 1, -1, 2, False)
    tm.set_transition_rule(0, 0, -1, 2, True)
    while not tm.halted:
        tm.step()
        print(tm)
    digits = [d - 1 for d in tm.tape if d != 0]
    return int(''.join(map(str, digits)), 2)


print(binary_increment(0)) # 1
print(binary_increment(63)) # 64
print(binary_increment(95437100874267743)) # 95437100874267744

