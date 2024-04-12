


import numpy as np


# Define board for algorithm 1.
board1 = np.array([
    [1, 2, 3, 4, 5, 6],
    [4, 5, 6, 7, 8, 9],
], dtype=int)

# Define board for algorithm 2.
board2 = [
    0x00_06_05_04_03_02_01,
    0x00_09_08_07_06_05_04,
]
board2_slots_per_player = 7
board2_max_int = (1 << (8 * board2_slots_per_player + 1)) - 1
board2_home = 0xff << (8 * board2_slots_per_player - 1)



print(board1)
print(board2)
print(board2_max_int)



def get_pieces_in_slot(board2: list[int], player: int, slot: int) -> int:
    if player < 0 or player >= len(board2):
        raise ValueError('player index is out of bounds')
    if slot < 0 or slot >= board2_slots_per_player:
        raise ValueError('slot index is out of bounds')
    return (board2[player] & (0xff << (8 * slot))) >> (8 * slot)

def board2_to_str(board2: list[int]) -> str:
    return '\n'.join(
        ' '.join(
            str(get_pieces_in_slot(board2, player, slot))
            for slot in range(board2_slots_per_player)
        )
        for player in range(len(board2))
    )



print(get_pieces_in_slot(board2, 0, 3))
print(board2_to_str(board2))



def distribute_pieces(board2: list[int], player: int, slot: int):
    if player < 0 or player >= len(board2):
        raise ValueError('player index is out of bounds')
    if slot < 0 or slot >= board2_slots_per_player:
        raise ValueError('slot index is out of bounds')
    p = 0xff << (8 * slot)
    n = board2[player] & p
    board2[player] ^= n
    n >>= 8 * slot
    i = 1 << (8 * (slot + 1))
    for _ in range(n // board2_slots_per_player + 1):
        for _ in range(board2_slots_per_player - slot):
            board2[player] += i
            i <<= 8
            if i >= board2_max_int:
                i = 1
            n -= 1
            if n <= 0:
                break
        if n <= 0:
            break
        player = (player + 1) % len(board2)

def can_distribute_pieces(board2: list[int], player: int, slot: int) -> bool:
    if player < 0 or player >= len(board2):
        raise ValueError('player index is out of bounds')
    if slot < 0 or slot >= board2_slots_per_player:
        raise ValueError('slot index is out of bounds')
    if slot == board2_slots_per_player - 1:
        return False
    return get_pieces_in_slot(board2, player, slot) > 0



distribute_pieces(board2, 0, 2)
print(board2_to_str(board2))


