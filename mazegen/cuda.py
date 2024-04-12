

import numpy as np
from numba import cuda, int8
from random import randint



@cuda.jit
def _cuda_kernel_maze_gen(maze):
    i = cuda.grid(1)
    if i >= maze.shape[0]: return
    directions = cuda.const.array_like(DIRECTIONS)
    cur_maze = cuda.local.array(MAZE_SIZE, int8)
    for y in range(MAZE_SIZE[0]):
        for x in range(MAZE_SIZE[1]):
            if cur_maze[y, x] != 0: continue
            d = randint(0,3)
            cur_x = x
            cur_y = y
            while True:
                if d == 0:
                    cur_maze[cur_y, cur_x] = directions[1, 0, 0, 0]
                elif d == 1:
                    cur_maze[cur_y, cur_x] = directions[0, 1, 0, 0]
                elif d == 2:
                    cur_maze[cur_y, cur_x] = directions[0, 0, 1, 0]
                elif d == 3:
                    cur_maze[cur_y, cur_x] = directions[0, 0, 0, 1]



MAZE_SIZE = 128, 128

DIRECTIONS = np.zeros((2, 2, 2, 2), np.int8)
for _e in range(2):
    for _n in range(2):
        for _w in range(2):
            for _s in range(2):
                DIRECTIONS[_e, _n, _w, _s] = _e + _n * 2 + _w * 4 + _s * 8

def maze_gen(N: int = 1024):
    maze = np.zeros((N, *MAZE_SIZE))
    _maze = cuda.to_device(maze)
    _cuda_kernel_maze_gen[N // 32, 32](_maze)
    maze = _maze.copy_to_host()
    return maze


def main():
    maze_gen()

if __name__ == '__main__':
    main()




