


from fischer.factoriobalancers import BeltGrid


def main():
    grid = BeltGrid((4, 9))

    # Set up four splitters for 4-4 balancer.
    for x, y, d in (
        (1, 0, 0),
        (1, 2, 0),
        (2, 1, 0),
        (5, 1, 0),
    ):
        grid.set_splitter(x, y, d)
    print(grid)

    # Connect four input belts to first two splitters.
    turtle = grid.turtles[grid.add_turtle(0, 0, 0)]
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[1])
    print()
    print(grid)

    turtle = grid.turtles[grid.add_turtle(0, 1, 0)]
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[1])
    print()
    print(grid)

    turtle = grid.turtles[grid.add_turtle(0, 2, 0)]
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[1])
    print()
    print(grid)

    turtle = grid.turtles[grid.add_turtle(0, 3, 0)]
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)

    # Connect first two splitters to one splitter.
    turtle = grid.turtles[grid.add_turtle(1, 1, 0)]
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)

    turtle = grid.turtles[grid.add_turtle(1, 2, 0)]
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)

    # Connect first two splitters to last splitter.
    turtle = grid.turtles[grid.add_turtle(1, 0, 0)]
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[1])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)

    turtle = grid.turtles[grid.add_turtle(1, 3, 0)]
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[1])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[1])
    print()
    print(grid)

    # Create undergrounds from the middle splitter.
    turtle = grid.turtles[grid.add_turtle(2, 1, 0)]
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[2])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)

    turtle = grid.turtles[grid.add_turtle(2, 2, 0)]
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[1])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)

    # Connect last splitter to outputs. (The outputs are not technically defined by the BeltGrid, but this would bring all the "output" belts together, so it's more elegant.)
    turtle = grid.turtles[grid.add_turtle(5, 1, 0)]
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[1])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)

    turtle = grid.turtles[grid.add_turtle(5, 2, 0)]
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)
    steps = list(turtle.possible_steps(8))
    print(steps)
    turtle.set_position(*steps[0])
    print()
    print(grid)



if __name__ == '__main__':
    main()





