
import os
from fischer.factoriobalancers import BeltGrid



def main():
    grid = BeltGrid((4, 10), max_inputs=4, max_outputs=4, max_splitters=4, max_turtles=1)
    turtle = grid.turtles[grid.add_turtle(0, 0, 0)]

    # turtle.set_position(5, 0, 0)
    # turtle.set_position(6, 0, 0)
    # turtle.set_position(8, 0, 0)
    # turtle.set_position(9, 0, 0)
    # turtle.reset(clear_path_on_grid=True)
    
    os.system('cls')
    print(f'Current path: {turtle.path}')
    print(grid)
    print('Valid moves:\n' + '\n'.join(
        f'{x} {y} {d}'
        for x, y, d in turtle.possible_steps(max_underground_length=8)
    ))
    print(grid.turtle_mask)

    while True:
        inp = input('move: ')
        if inp == 'b':
            turtle.backtrack()
        else:
            split = inp.split(' ')
            x, y, d = map(int, split)
            if not turtle.set_position(x, y, d):
                print(f'Invalid step: {turtle.x} {turtle.y} {turtle.direction} -> {x} {y} {d}')
                continue
        os.system('cls')
        print(f'Current path: {turtle.path}')
        print(grid)
        print('Valid moves:\n' + '\n'.join(
            f'{x} {y} {d}'
            for x, y, d in turtle.possible_steps(max_underground_length=8)
        ))


if __name__ == '__main__':
    main()


