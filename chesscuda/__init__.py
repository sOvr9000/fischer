

'''

CUDA powered minimax chess engine.  Very fast engine which plays moderately strong moves.

(WIP)

'''


# pieces:
# 0 = empty
# 1-6 = white pawn, knight, bishop, rook, queen, king
# 7-12 = black pawn, knight, bishop, rook, queen, king



from numba import cuda
import numpy as np


bitboards = {}
bitboards['standard'] = (
    sum(int(16 ** n * p) for n, p in enumerate([4, 2, 3, 5, 6, 3, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1] + [0] * 16)),
    sum(int(16 ** n * p) for n, p in enumerate([0] * 16 + [7, 7, 7, 7, 7, 7, 7, 7, 10, 8, 9, 11, 12, 9, 8, 10]))
)

piece_symbols = [' ', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

def piece_is_empty(p: np.int_) -> np.bool_:
    return p == 0

def piece_is_pawn(p: np.int_) -> np.bool_:
    return p == 1 or p == 7

def piece_is_knight(p: np.int_) -> np.bool_:
    return p == 2 or p == 8

def piece_is_bishop(p: np.int_) -> np.bool_:
    return p == 3 or p == 9

def piece_is_rook(p: np.int_) -> np.bool_:
    return p == 4 or p == 10

def piece_is_queen(p: np.int_) -> np.bool_:
    return p == 5 or p == 11

def piece_is_king(p: np.int_) -> np.bool_:
    return p == 6 or p == 12

def get_piece_at(bb0: np.int_, bb1: np.int_, s: np.int_) -> np.int_:
    if s >= 32:
        s -= 32
        bb0 = bb1
    s <<= 2
    return (bb0&(15<<s))>>s

def set_piece_at(bb0: np.int_, bb1: np.int_, s: np.int_, p: np.int_) -> None:
    if s >= 32:
        s -= 32
        s <<= 2
        l = 15<<s
        bb1 = (bb1 ^ l) | (p * l)
    else:
        s <<= 2
        l = 15<<s
        bb0 = (bb0 ^ l) | (p * l)
    return bb0, bb1


def move_piece(bb0: np.int_, bb1: np.int_, move: np.int_) -> tuple[np.int_, np.int_]:
    s0 = move>>6
    s1 = move&63
    p = get_piece_at(bb0, bb1, s0)
    if not piece_is_empty(p):
        bb0, bb1 = set_piece_at(bb0, bb1, s0, 0)
        bb0, bb1 = set_piece_at(bb0, bb1, s1, p)
    return bb0, bb1


def get_piece_symbol(p: int) -> str:
    return piece_symbols[p]

def print_board(bb0: int, bb1: int) -> None:
    for n in range(64):
        print(get_piece_symbol(get_piece_at(bb0, bb1, 63-n)), end='\n' if n%8==7 else ' ')

def main():
    bb0, bb1 = bitboards['standard']
    print(f'{bb0:x}')
    print(f'{bb1:x}')
    print_board(bb0, bb1)
    bb0, bb1 = move_piece(bb0, bb1, 796)
    print_board(bb0, bb1)

if __name__ == '__main__':
    main()





