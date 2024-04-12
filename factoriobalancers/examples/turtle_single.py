


from fischer.factoriobalancers import BeltGrid


def main():
    grid = BeltGrid((4, 7))
    grid.set_splitter(0, 0, 0)
    turtle = grid.turtles[grid.add_turtle(0, 0, 0)]
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)

    '''
❏ ❏ ❏ ❏ ❏ ❏ ❏
❏ ❏ ❏ ❏ ❏ ❏ ❏
⇨ ❏ ❏ ❏ ❏ ❏ ❏
⇨ ❏ ❏ ❏ ❏ ❏ ❏
[(1, 0, 0)]
    '''

    print()
    turtle.set_position(*steps[0])
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)

    '''
❏ ❏ ❏ ❏ ❏ ❏ ❏
❏ ❏ ❏ ❏ ❏ ❏ ❏
⇨ ❏ ❏ ❏ ❏ ❏ ❏
⇨ → ❏ ❏ ❏ ❏ ❏
[(1, 1, 1), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0)]
    '''

    print()
    turtle.set_position(*steps[0])
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)

    '''
❏ ❏ ❏ ❏ ❏ ❏ ❏
❏ ❏ ❏ ❏ ❏ ❏ ❏
⇨ ↑ ❏ ❏ ❏ ❏ ❏
⇨ ↑ ❏ ❏ ❏ ❏ ❏
[(2, 1, 0), (1, 2, 1)]
    '''

    print()
    turtle.set_position(*steps[1])
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)

    '''
❏ ❏ ❏ ❏ ❏ ❏ ❏
❏ ↑ ❏ ❏ ❏ ❏ ❏
⇨ ↑ ❏ ❏ ❏ ❏ ❏
⇨ ↑ ❏ ❏ ❏ ❏ ❏
[(2, 2, 0), (0, 2, 2), (1, 3, 1)]
    '''

    print()
    turtle.set_position(*steps[0])
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)

    '''
❏ ❏ ❏ ❏ ❏ ❏ ❏
❏ → → ❏ ❏ ❏ ❏
⇨ ↑ ❏ ❏ ❏ ❏ ❏
⇨ ↑ ❏ ❏ ❏ ❏ ❏
[(2, 1, 3), (2, 3, 1), (3, 2, 0), (4, 2, 0), (5, 2, 0)]
    '''

    print()
    turtle.set_position(*steps[3])
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)

    '''
❏ ❏ ❏ ❏ ❏ ❏ ❏
❏ → ⇢ ❏ ⇢ ❏ ❏
⇨ ↑ ❏ ❏ ❏ ❏ ❏
⇨ ↑ ❏ ❏ ❏ ❏ ❏
[(5, 2, 0)]
    '''

    print()
    turtle.set_position(*steps[0])
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)

    '''
❏ ❏ ❏ ❏ ❏ ❏ ❏
❏ → ⇢ ❏ ⇢ → ❏
⇨ ↑ ❏ ❏ ❏ ❏ ❏
⇨ ↑ ❏ ❏ ❏ ❏ ❏
[(5, 1, 3), (5, 3, 1), (6, 2, 0)]
    '''

    print()
    turtle.set_position(*steps[0])
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)

    '''
❏ ❏ ❏ ❏ ❏ ❏ ❏
❏ → ⇢ ❏ ⇢ ↓ ❏
⇨ ↑ ❏ ❏ ❏ ↓ ❏
⇨ ↑ ❏ ❏ ❏ ❏ ❏
[(4, 1, 2), (6, 1, 0), (5, 0, 3)]
    '''

    print()
    turtle.set_position(*steps[2])
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)

    '''
❏ ❏ ❏ ❏ ❏ ❏ ❏
❏ → ⇢ ❏ ⇢ ↓ ❏
⇨ ↑ ❏ ❏ ❏ ↓ ❏
⇨ ↑ ❏ ❏ ❏ ↓ ❏
[(4, 0, 2), (6, 0, 0)]
    '''



if __name__ == '__main__':
    main()


