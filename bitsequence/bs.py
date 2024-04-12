
from typing import Generator
from math import log2, ceil

def get_bit_sequence(x: float, length: int = 56) -> Generator[None, None, int]:
    '''
    Get the significant sequence of binary digits in the binary digit expansion of `x`, starting with the bit corresponding to the highest power of two that is less than `x`.
    '''
    if x == 0:
        for _ in range(length):
            yield 0
        return
    u = x * 2 ** -ceil(log2(x))
    if u == 1:
        yield 1
        for _ in range(length - 1):
            yield 0
    else:
        for _ in range(length):
            u *= 2
            yield int(u%2)
