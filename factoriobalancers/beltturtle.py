
import numpy as np
from typing import Generator
from fischer.math.common import l1_distance, lerp_vector
from .beltgraph import try_derive_graph, BeltGraph
import colorama



def get_splitter_center(x: int, y: int, d: int) -> tuple[float, float]:
    '''
    Get the center position of a splitter at `(x, y)` with direction `d`.
    '''
    return lerp_vector(x, y, *get_splitter_other_position(x, y, d), .5)

def step_back(x: int, y: int, d: int) -> tuple[int, int, int]:
    '''
    Return the offset position of `(x, y)` in the opposite direction of `d` along with the flipped `d` as a tuple of the form `(new_x, new_y, new_d)`.
    '''
    _d = (d + 2) % 4
    return *offset_position(x, y, _d), _d

def get_direction(x: int, y: int, prev_x: int, prev_y: int) -> int:
    '''
    Given a previously visited cell and a currently visiting cell, return the direction of the previously taken step.

    In other words, get the direction from `(prev_x, prev_y)` to `(x, y)`.
    '''
    if x == prev_x:
        if y < prev_y:
            return 3
        if y > prev_y:
            return 1
    elif y == prev_y:
        if x < prev_x:
            return 2
        if x > prev_x:
            return 0
    if x == prev_x and y == prev_y:
        return -1
    raise Exception(f'Could not obtain a direction from ({x}, {y}) to ({prev_x}, {prev_y}).')

def directions_are_opposite(d0: int, d1: int) -> bool:
    '''
    Return whether the directions are opposite of each other.
    '''
    return abs(d0 - d1) == 2

adjacency_offsets: list[tuple[int, int]] = [(1, 0), (0, 1), (-1, 0), (0, -1)]

def get_splitter_other_position(x: int, y: int, d: int) -> tuple[int, int]:
    '''
    Given a splitter at `(x, y)` with direction `d`, return the adjacent position that the splitter also occupies.
    '''
    d2 = d % 2
    return x + d2, y + 1 - d2

def get_splitter_other_position_inverse(x: int, y: int, d: int) -> tuple[int, int]:
    '''
    Return the root position of a splitter given its other occupied position.

    It is an identity that `get_splitter_other_position(*get_splitter_other_position_inverse(x, y, d), d) == (x, y)`
    '''
    if d % 2 == 0:
        return x, y - 1
    return x - 1, y

def get_splitter_input_positions(x: int, y: int, d: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if d == 0:
        return (x - 1, y), (x - 1, y + 1)
    if d == 2:
        return (x + 1, y), (x + 1, y + 1)
    if d == 1:
        return (x, y - 1), (x + 1, y - 1)
    if d == 3:
        return (x, y + 1), (x + 1, y + 1)

def get_splitter_output_positions(x: int, y: int, d: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return get_splitter_input_positions(x, y, (d + 2) % 4)

def get_adjacency_offset(d: int) -> tuple[int, int]:
    assert is_valid_direction(d)
    return adjacency_offsets[d]

def offset_position(x: int, y: int, d: int) -> tuple[int, int]:
    '''
    Return `(x, y)` offset by one step in the direction `d`.
    '''
    assert is_valid_position(x, y)
    assert is_valid_direction(d)
    if d == 0:
        return x + 1, y
    if d == 1:
        return x, y + 1
    if d == 2:
        return x - 1, y
    return x, y - 1

def is_valid_direction(d: int) -> bool:
    '''
    Return whether the given direction `d` is either `0`, `1`, `2`, or `3`.
    '''
    return isinstance(d, (int, np.int_, np.int64)) and d >= 0 and d < 4

def is_valid_position(x: int, y: int) -> bool:
    '''
    Return whether the given position `(x, y)` consists of integers.
    '''
    return isinstance(x, (int, np.int_, np.int64)) and isinstance(y, (int, np.int_, np.int64))

def is_valid_turn(turn: int) -> bool:
    '''
    Return whether `turn` is `-1`, `0`, or `1`.
    '''
    return isinstance(turn, (int, np.int_, np.int64)) and turn >= -1 and turn <= 1

def is_valid_max_underground_length(length: int) -> bool:
    '''
    Return whether length is a nonnegative integer.
    '''
    return isinstance(length, (int, np.int_, np.int64)) and length >= 0



class BeltGrid:
    def __init__(self, size: tuple[int, int], max_inputs: int, max_outputs: int, max_splitters: int, max_turtles: int, debug_info: bool = False):
        self.height, self.width = size
        self.max_inputs = max_inputs
        self.max_outputs = max_outputs
        self.max_splitters = max_splitters
        self.max_turtles = max_turtles

        # -1: empty, 0-3: belt, 4-7: underground, 8-11: splitter
        self.grid = -np.ones(size, dtype=int)

        # False: entrance, True: exit
        self.underground_types = np.zeros(size, dtype=bool)

        self.turtles: list['BeltTurtle'] = []
        self.splitters: list[tuple[int, int, int]] = []
        self.inputs = []
        self.outputs = []
        self.turtle_mask = np.zeros(size, dtype=bool)

        self.debug_info = debug_info
    def is_turtle(self, x: int, y: int) -> bool:
        assert is_valid_position(x, y)
        if not self.is_within_bounds(x, y):
            return False
        return self.turtle_mask[y, x]
    def set_direction(self, x: int, y: int, d: int):
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        assert self.grid[y, x] < 8
        if self.grid[y, x] >= 4:
            d += 4
        self.grid[y, x] = d
    def can_set_input(self, x: int, y: int, d: int) -> bool:
        '''
        Return whether an input with direction `d` can be placed at `(x, y)`.
        '''
        if len(self.inputs) >= self.max_inputs:
            return False
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        # Check if the input is facing the edge.
        if not self.is_within_bounds(*offset_position(x, y, d)):
            return False
        if not self.is_open(x, y, ignore_turtles=False):
            return False
        # Check ahead of the input position to see if it is open, ignoring turtles.
        if not self.is_open(*offset_position(x, y, d), ignore_turtles=True):
            return False
        # Check if the input position is behind any currently existing outputs.
        if self.blocks_output(x, y, -1): # -1 used to force the function to check all directions
            return False
        # Check if the output position is ahead of any currently existing inputs.
        if self.blocks_input(x, y, -1): # -1 used to force the function to check all directions
            return False
        return True
    def set_input(self, x: int, y: int, d: int) -> int:
        '''
        Set `(x, y)` to be an input to the belt grid with direction `d`.
        '''
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        assert self.is_open(x, y, ignore_turtles=False)
        self.inputs.append((x, y, d))
        self.grid[y, x] = d
        return len(self.inputs) - 1
    def can_set_output(self, x: int, y: int, d: int) -> bool:
        '''
        Return whether an output with direction `d` can be placed at `(x, y)`.
        '''
        if len(self.outputs) >= self.max_outputs:
            return False
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        # Check if the output is facing away from the edge.
        if not self.is_within_bounds(*offset_position(x, y, (d + 2) % 4)):
            return False
        if not self.is_open(x, y, ignore_turtles=False):
            return False
        # Check behind the output position to see if it is open, not ignoring turtles.
        if not self.is_open(*offset_position(x, y, (d + 2) % 4), ignore_turtles=False):
            return False
        # Check if the output position is ahead of any currently existing inputs.
        if self.blocks_input(x, y, -1): # -1 used to force the function to check all directions
            return False
        # Check if the output position is behind any currently existing outputs.
        if self.blocks_output(x, y, -1): # -1 used to force the function to check all directions
            return False
        return True
    def set_output(self, x: int, y: int, d: int) -> int:
        '''
        Set `(x, y)` to be an output to the belt grid with direction `d`.
        '''
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        assert self.is_open(x, y, ignore_turtles=False)
        self.outputs.append((x, y, d))
        self.grid[y, x] = d
        return len(self.outputs) - 1
    def blocks_output(self, x: int, y: int, d: int) -> bool:
        '''
        Return whether an entity at `(x, y)` facing `d` blocks any outputs.

        Returns False even if the intended usage of the space at `(x, y)` is to place an entrance of an underground belt in the same direction as the nearby output (and the tunnel going under the output), which would be blocking an output if the other cases would return True.  That means potentially one extra condition should be checked before placing an underground belt near an output.  It is a rare case, so it is likely okay to ignore it.
        '''
        assert is_valid_position(x, y)
        for ox, oy, od in self.outputs:
            if d == od:
                # This is where that extra condition should be checked, but is not possible given the limited information about the inteded usage of this position.
                continue
            _x, _y = offset_position(ox, oy, (od + 2) % 4)
            if _x == x and _y == y:
                return True
        return False
    def blocks_input(self, x: int, y: int, d: int) -> bool:
        '''
        Return whether an entity at `(x, y)` facing `d` blocks any inputs.

        Returns False even if the intended usage of the space at `(x, y)` is to place an exit of an underground belt in the same direction as the nearby input (and the tunnel going under the input), which would be blocking an input if the other cases would return True.  That means potentially one extra condition should be checked before placing an underground belt near an input.  It is a rare case, so it is likely okay to ignore it.
        '''
        assert is_valid_position(x, y)
        for ox, oy, od in self.inputs:
            if d == od:
                # This is where that extra condition should be checked, but is not possible given the limited information about the inteded usage of this position.
                continue
            _x, _y = offset_position(ox, oy, od)
            if _x == x and _y == y:
                return True
        return False
    def try_set_direction(self, x: int, y: int, d: int) -> None:
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        if self.grid[y, x] < 8:
            if self.grid[y, x] >= 4:
                if self.grid[y, x] % 2 == d % 2: # allow reversing of undergrounds but not turning
                    d += 4
                else:
                    return
            self.grid[y, x] = d
    def is_within_bounds(self, x: int, y: int) -> bool:
        '''
        Return whether `(x, y)` is within the bounds of the grid.
        '''
        assert is_valid_position(x, y)
        return x >= 0 and y >= 0 and x < self.grid.shape[1] and y < self.grid.shape[0]
    def is_open(self, x: int, y: int, ignore_turtles: bool = False) -> bool:
        '''
        Return whether the given position is open in the grid.
        '''
        assert is_valid_position(x, y)
        if not self.is_within_bounds(x, y):
            return False
        if self.grid[y, x] != -1:
            return False
        if not ignore_turtles:
            if self.is_turtle(x, y):
                return False
        return True
    def is_passable(self, x: int, y: int, d: int) -> bool:
        '''
        Return whether a belt can connect to `(x, y)` in direction `d`.  This is a specific case for splitters that can look like obstructions in a belt path but are still allowed for belts to point toward / cross.

        If not a splitter, return `False` when `(x, y)` is not open, i.e. return `False` when `BeltGrid.is_open(x, y) == False`.
        '''
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        if not self.is_within_bounds(x, y):
            return False
        if self.grid[y, x] >= 8:
            return self.grid[y, x] - 8 == d
        if not self.is_open(x, y):
            return (x, y, d) in self.outputs
        return True
    def underground_exits(self, x: int, y: int, d: int, max_length: int) -> Generator[tuple[int, int], None, None]:
        '''
        Iterate over the possible exit points of an underground belt that starts on `(x, y)` in direction `d`, in order of increasing distance from `(x, y)`.
        '''
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        assert is_valid_max_underground_length(max_length)
        dx, dy = get_adjacency_offset(d)
        ux, uy = x + 2 * dx, y + 2 * dy
        for _ in range(max_length):
            ux += dx
            uy += dy
            if not self.is_within_bounds(ux, uy):
                return
            if self.is_passable(ux, uy, d) and self.is_open(ux - dx, uy - dy):
                yield ux - dx, uy - dy
    def can_set_splitter(self, x: int, y: int, d: int, ignore_turtles: bool = True) -> bool:
        '''
        Return whether a splitter can be positioned at `(x, y)` in direction `d`.
        '''
        if len(self.splitters) >= self.max_splitters:
            return False
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        if not self.is_open(x, y, ignore_turtles=ignore_turtles) or not self.is_open(*get_splitter_other_position(x, y, d), ignore_turtles=ignore_turtles):
            # print(f'{x, y, d}: blocked')
            return False
        if any(not self.is_within_bounds(_x, _y) for _x, _y in list(get_splitter_input_positions(x, y, d)) + list(get_splitter_output_positions(x, y, d))):
            # print(f'{x, y, d}: out of bounds')
            return False
        # Check if the new splitter would be directly in front of a single other splitter, which accomplishes nothing and should be avoided in any scenario.
        s0, s1 = (self.get_splitter_index_from_position(_x, _y) for _x, _y in get_splitter_input_positions(x, y, d))
        if s0 != -1 and s0 == s1:
            return False
        # Check if the new splitter would be directly behind a single other splitter, which accomplishes nothing and should be avoided in any scenario.
        s0, s1 = (self.get_splitter_index_from_position(_x, _y) for _x, _y in get_splitter_output_positions(x, y, d))
        if s0 != -1 and s0 == s1:
            return False
        # Check if any output positions of the new splitter are an input belt.
        (x1, y1), (x2, y2) = get_splitter_output_positions(x, y, d)
        if self.is_input(x1, y1) or self.is_input(x2, y2):
            return False
        # Check if there's at least one passable output position.
        if not self.is_open(x1, y1, ignore_turtles=ignore_turtles) and not self.is_open(x2, y2, ignore_turtles=ignore_turtles):
            return False
        # Check if the new splitter would cause a sideloaded belt at any output.
        if self.is_sideloaded(x1, y1) or self.is_sideloaded(x2, y2):
            return False
        # Check if there's at least one passable input position (that isn't an output).
        (x1, y1), (x2, y2) = get_splitter_input_positions(x, y, d)
        if not (self.is_passable(x1, y1, d) and not self.is_output(x, y)) and not (self.is_passable(x2, y2, d) and not self.is_output(x2, y2)):
            return False
        # Check if any of the splitter positions block an output or input.
        x2, y2 = get_splitter_other_position(x, y, d)
        if self.blocks_output(x, y, d) or self.blocks_output(x2, y2, d) or self.blocks_input(x, y, d) or self.blocks_input(x2, y2, d):
            return False
        return True
    def set_splitter(self, x: int, y: int, d: int) -> int:
        '''
        Position a splitter at `(x, y)` with direction `d`, and return its index in `BeltGrid.splitters`.
        '''
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        self.grid[y, x] = d + 8
        self.splitters.append((x, y, d))
        x, y = get_splitter_other_position(x, y, d)
        self.grid[y, x] = d + 8
        return len(self.splitters) - 1
    def get_splitter(self, index: int) -> tuple[int, int, int]:
        '''
        Return a splitter's position and direction as a 3-tuple of the form `(x, y, d)`.
        '''
        assert index >= 0 and index < len(self.splitters)
        return self.splitters[index]
    def is_sideloaded(self, x: int, y: int) -> bool:
        '''
        Return whether the belt at `(x, y)` is sideloaded (where two or more belts, splitters, or undergrounds are pointed toward it)
        '''
        assert is_valid_position(x, y)
        if not self.is_within_bounds(x, y):
            return False
        if self.grid[y, x] < 0:
            return False
        if self.grid[y, x] >= 8:
            return False
        if self.grid[y, x] >= 4:
            # Check if the underground belt is sideloaded, which is only possible when a belt, splitter, or underground is pointed toward it from the side.
            d = self.grid[y, x] - 4
            x1, y1 = offset_position(x, y, (d + 1) % 4)
            if self.is_within_bounds(x1, y1) and self.grid[y1, x1] % 4 == (d + 3) % 4:
                return True
            x2, y2 = offset_position(x, y, (d + 3) % 4)
            if self.is_within_bounds(x2, y2) and self.grid[y2, x2] % 4 == (d + 1) % 4:
                return True
        else:
            # Check if the belt has two or more ingoing belts, undergrounds, or splitters.
            d = self.grid[y, x]
            if sum(
                self.is_within_bounds(x + dx, y + dy) and self.grid[y + dy, x + dx] >= 0 and self.grid[y + dy, x + dx] % 4 == d
                for dx, dy in adjacency_offsets
            ) >= 2:
                return True
        return False
    def is_splitter_connectable_forward(self, x: int, y: int) -> bool:
        '''
        Return whether the splitter at this position can be connected forward with a belt or underground belt.

        This usually marks a good position to place a turtle.
        '''
        assert is_valid_position(x, y)
        assert self.is_within_bounds(x, y)
        if not self.is_splitter(x, y):
            return False
        sd = self.grid[y, x] - 8
        x2, y2 = offset_position(x, y, sd)
        if not self.is_open(x2, y2):
            return False
        return True
    def is_splitter_connectable_backward(self, x: int, y: int) -> bool:
        '''
        Return whether the splitter at this position can be connected backward with a belt or underground belt.

        This usually marks a good position to place a turtle if the turtle traverses in the opposite direction of the belt.
        '''
        assert is_valid_position(x, y)
        assert self.is_within_bounds(x, y)
        if not self.is_splitter(x, y):
            return False
        sd = self.grid[y, x] - 8
        x2, y2 = offset_position(x, y, (sd + 2) % 4)
        if not self.is_open(x2, y2):
            return False
        return True
    def can_add_turtle(self, x: int, y: int, d: int, require_splitter: bool = True) -> bool:
        '''
        Return whether a turtle can be added at `(x, y)` with direction `d`.

        If `require_splitter = True`, then the turtle can only be added if the position is a splitter that is connectable forward or if it's an input belt.
        '''
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        if len(self.turtles) >= self.max_turtles:
            return False
        if self.is_turtle(x, y):
            return False
        if require_splitter:
            if self.is_splitter(x, y):
                if not self.is_splitter_connectable_forward(x, y):
                    return False
            elif not self.is_input(x, y):
                return False
            # At this point, the position is either an input belt or a splitter that is connectable forward.
            if d != self.grid[y, x] % 4:
                return False
            if self.is_input(x, y) and not self.is_open(*offset_position(x, y, d)):
                return False
            return True
        if not self.is_passable(x, y, d):
            return False
        return True
    def add_turtle(self, x: int, y: int, d: int, ignore: bool = False, set_direction_immediately: bool = False) -> int:
        '''
        Add a `BeltTurtle` to the grid, and return its index in `BeltGrid.turtles`.
        '''
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        self.turtles.append(BeltTurtle(self, x, y, direction=d))
        if set_direction_immediately:
            self.set_direction(x, y, d)
        if not ignore:
            self.turtle_mask[y, x] = True
        return len(self.turtles) - 1
    def remove_turtle(self, turtle: 'BeltTurtle') -> None:
        '''
        Remove a turtle from the grid.
        '''
        self.turtle_mask[turtle.y, turtle.x] = False
        self.turtles.remove(turtle)
    def get_turtle_at(self, x: int, y: int) -> 'BeltTurtle':
        '''
        Return the turtle at `(x, y)`.
        '''
        assert is_valid_position(x, y)
        for turtle in self.turtles:
            if turtle.x == x and turtle.y == y:
                return turtle
    def _post_process_turtle_step(self, turtle: 'BeltTurtle') -> None:
        '''
        Ensure that if the turtle steps onto a splitter, it is removed from the grid because it means a belt connection has been made between splitters.
        '''
        x, y = turtle.x, turtle.y
        if self.is_splitter(x, y) or self.is_output(x, y):
            self.remove_turtle(turtle)
    def to_blueprint_data(self) -> dict:
        '''
        Return a dict which represents a blueprint in Factorio.
        '''
        # Construct blueprint data.
        def splitter_pos(x: int, y: int) -> tuple[float, float]:
            x, y, d = self.get_splitter(self.get_splitter_index_from_position(x, y))
            x2, y2 = get_splitter_other_position(x, y, d)
            x, y = .5 * (x + x2), .5 * (y + y2)
            return {'x': x+.5, 'y': y+.5}
        def offset_pos(x: int, y: int) -> dict[str, float]:
            return {'x': x+.5, 'y': y+.5}
        i = -1
        entities: list[dict] = [
            {
                'entity_number': (i := i + 1),
                'name': 'express-transport-belt' if self.grid[y, x] < 4 else 'express-underground-belt',
                'position': offset_pos(x, y),
                'direction': int(self.grid[y, x] % 4 * 2), # Cast to int to avoid numpy int32, which is not JSON serializable.
            } | ({ # This is the dictionary operator for calling update(), as in `a.update(b)` <=> `a |= b`.
                'type': 'output' if self.is_underground_exit(x, y) else 'input'
            } if self.grid[y, x] >= 4 else {})
            if self.grid[y, x] < 8 else
            {'name': 'express-splitter', 'position': splitter_pos(x, y), 'direction': int((self.grid[y, x] - 8) * 2)}
            for y in range(self.height)
            for x in range(self.width)
            if self.grid[y, x] >= 0
        ]
        bp_data = {'blueprint': {'entities': entities}}
        return bp_data
    def get_graph(self) -> 'BeltGraph':
        '''
        Return a `BeltGraph` which abstractly represents the connections between splitters that the belts define within the `BeltGrid`.

        Utilizes `try_derive_graph()` from `beltgraph`.

        If `try_derive_graph()` fails, return an empty `BeltGraph` of zero inputs, outputs, and edges.  This typically happens when belts are sideloaded, which is when a belt has two or more belts (or splitters or undergrounds) pointed toward it.
        '''
        graph = BeltGraph()
        try_derive_graph(self.to_blueprint_data(), graph, verbose=self.debug_info)
        return graph
    def get_splitter_index_from_position(self, x: int, y: int) -> int:
        '''
        Return the index of a splitter, input, or output that occupies `(x, y)`.
        '''
        assert is_valid_position(x, y)
        if not self.is_within_bounds(x, y):
            return -1
        for i, (sx, sy, sd) in enumerate(self.splitters):
            if sx == x and sy == y:
                return i
            other_x, other_y = get_splitter_other_position(sx, sy, sd)
            if other_x == x and other_y == y:
                return i
        for i, (sx, sy, sd) in enumerate(self.inputs):
            if sx == x and sy == y:
                return i
        for i, (sx, sy, sd) in enumerate(self.outputs):
            if sx == x and sy == y:
                return i
        return -1
    def is_underground(self, x: int, y: int) -> bool:
        '''
        Return whether an underground belt starts or ends at `(x, y)`.
        '''
        assert is_valid_position(x, y)
        if not self.is_within_bounds(x, y):
            return False
        return self.grid[y, x] >= 4 and self.grid[y, x] < 8
    def is_input(self, x: int, y: int) -> bool:
        '''
        Return whether an input occupies `(x, y)`.
        '''
        for _x, _y, _ in self.inputs:
            if _x == x and _y == y:
                return True
        return False
    def is_output(self, x: int, y: int) -> bool:
        '''
        Return whether an output occupies `(x, y)`.
        '''
        for _x, _y, _ in self.outputs:
            if _x == x and _y == y:
                return True
        return False
    def is_splitter(self, x: int, y: int) -> bool:
        '''
        Return whether a splitter occupies `(x, y)`.
        '''
        return self.grid[y, x] >= 8
    def is_underground_exit(self, x: int, y: int) -> bool:
        '''
        Return whether the underground belt at `(x, y)` is an exit.
        '''
        if not self.is_underground(x, y):
            return False
        return self.underground_types[y, x]
    def reset(self):
        '''
        Clear the grid, but keep turtles and send them back to their initial positions.
        '''
        self.grid[:] = -1
        self.splitters.clear()
        self.inputs.clear()
        self.outputs.clear()
        self.turtle_mask[:] = False
        self.underground_types[:] = False
        for turtle in self.turtles:
            turtle.reset()
    def full_reset(self):
        '''
        Clear the grid and remove all turtles.
        '''
        self.reset()
        self.turtles.clear()
    BELT_CHARS = '\u2192', '\u2191', '\u2190', '\u2193'
    UNDERGROUND_CHARS = '\u21e2', '\u21e1', '\u21e0', '\u21e3'
    SPLITTER_CHARS = '\u21e8', '\u21e7', '\u21e6', '\u21e9'
    EMPTY_CHAR = '\u274f'
    INPUT_COLOR = colorama.Fore.GREEN
    OUTPUT_COLOR = colorama.Fore.RED
    SPLITTER_COLOR = colorama.Fore.CYAN
    TURTLE_COLOR = colorama.Fore.MAGENTA
    def __str__(self) -> str:
        def wrap_color(s: str, x: int, y: int) -> str:
            if self.is_input(x, y):
                return self.INPUT_COLOR + s + colorama.Fore.RESET
            if self.is_output(x, y):
                return self.OUTPUT_COLOR + s + colorama.Fore.RESET
            return s
        return '\n'.join(
            ' '.join(
                wrap_color(self.BELT_CHARS[v], x, y)
                if v >= 0 and v < 4 else
                self.UNDERGROUND_CHARS[v - 4]
                if v >= 4 and v < 8 else
                self.SPLITTER_COLOR + self.SPLITTER_CHARS[v - 8] + colorama.Fore.RESET
                if v >= 8 else
                self.EMPTY_CHAR
                for x, v in enumerate(r)
            )
            for y, r in reversed(list(enumerate(self.grid)))
        ) + '\n' + 'Turtle mask:\n' + '\n'.join(
            ' '.join(
                (
                    self.TURTLE_COLOR + self.BELT_CHARS[self.get_turtle_at(x, y).direction] + colorama.Fore.RESET
                ) if self.turtle_mask[y, x] else self.EMPTY_CHAR
                for x in range(self.width)
            )
            for y in range(self.height - 1, -1, -1)
        )

class BeltTurtle:
    def __init__(self, grid: BeltGrid, x: int, y: int, direction: int = 0):
        self.grid = grid
        self.path: list[tuple[int, int, int]] = []
        self.set_start_position(x, y, direction)
    def reset(self, clear_path_on_grid: bool = True):
        '''
        Clear the current path and reset the position to its starting position as defined in the constructor of `BeltTurtle`.
        '''
        self.x = self.start_x
        self.y = self.start_y
        self.direction = self.start_direction
        if clear_path_on_grid:
            for x, y, _ in self.path:
                self.grid.grid[y, x] = -1
        self.path.clear()
        self.path.append((self.x, self.y, self.direction))
    def can_step(self, turn: int) -> bool:
        '''
        Return whether the turtle can turn in a certain direction.
        '''
        assert is_valid_turn(turn)
        new_d = (self.direction + turn) % 4
        new_x, new_y = offset_position(self.x, self.y, new_d)
        if not self.grid.is_passable(new_x, new_y, new_d):
            return turn == 0 and any(self.grid.get_underground_exits(self.x, self.y, new_d))
        return True
    def set_start_position(self, x: int, y: int, d: int):
        '''
        Set the start position and direction of the turtle.
        '''
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        self.start_x = x
        self.start_y = y
        self.start_direction = d
        self.reset(clear_path_on_grid=False)
    def set_position(self, x: int, y: int, d: int) -> bool:
        '''
        Return whether the turtle successfully stepped.
        '''
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        assert x != self.x or y != self.y
        if not self.grid.is_passable(x, y, d) and not (self.grid.is_output(x, y) and d == self.grid.grid[y, x]):
            return False
        is_underground = l1_distance(x, y, self.x, self.y) > 1
        self.grid.turtle_mask[self.y, self.x] = False
        v = d + 4 * int(is_underground)
        if is_underground:
            # Set the current position as an entrace.
            self.grid.underground_types[self.y, self.x] = False
            # Set the new position as an exit.
            self.grid.underground_types[y, x] = True
        if self.grid.grid[self.y, self.x] < 4:
            self.grid.grid[self.y, self.x] = v
        if self.grid.grid[y, x] < 4:
            self.grid.grid[y, x] = v
        self.x = x
        self.y = y
        self.grid.turtle_mask[self.y, self.x] = True
        self.direction = d
        self.path.append((self.x, self.y, self.direction))
        self.grid._post_process_turtle_step(self)
        return True
    def possible_steps(self, max_underground_length: int) -> Generator[tuple[int, int, int], None, None]:
        '''
        Iterate over all possible positions and directions as tuples of the form `(x, y, d)` which the turtle can enter from its current position and direction.

        The maximum number of position-direction tuples iterated is equal to `3 + max_underground_length`.
        '''
        assert is_valid_max_underground_length(max_underground_length)
        turns = -1, 1, 0 # not (-1, 0, 1) solely because (-1, 1, 0) groups the non-underground steps at the beginning of the iterator, potentially simplifying other logic
        def is_step_valid(x: int, y: int, d: int) -> bool:
            if x == self.x and y == self.y:
                return False
            if not self.grid.is_within_bounds(x, y):
                return False
            if self.grid.is_turtle(x, y):
                return False
            if self.grid.is_input(x, y):
                return False
            if self.grid.is_output(x, y):
                if d != self.grid.grid[y, x]:
                    return False
            if 0 <= self.grid.grid[y, x] < 4 and not (self.grid.is_output(x, y) and self.grid.grid[y, x] == d):
                return False
            if not self.grid.is_open(x, y) and self.grid.grid[y, x] % 4 != d and self.grid.grid[y, x] >= 4:
                return False
            if self.grid.is_underground_exit(x, y):
                return False
            return True
        if self.grid.grid[self.y, self.x] >= 8:
            if self.grid.grid[self.y, self.x] - 8 == self.direction:
                x, y = offset_position(self.x, self.y, self.grid.grid[self.y, self.x] - 8)
                d = self.grid.grid[self.y, self.x] - 8
                if is_step_valid(x, y, d):
                    yield x, y, d
            return
        if self.grid.grid[self.y, self.x] >= 4 and self.grid.grid[self.y, self.x] < 8:
            turns = 0,
        for turn in turns:
            new_d = (self.direction + turn) % 4
            new_x, new_y = offset_position(self.x, self.y, new_d)
            if is_step_valid(new_x, new_y, new_d):
                yield new_x, new_y, new_d
            if turn == 0 and self.grid.grid[self.y, self.x] < 4 and not self.grid.is_input(self.x, self.y):
                for x, y in self.grid.underground_exits(self.x, self.y, self.direction, max_underground_length):
                    d = self.direction
                    if not is_step_valid(x, y, d):
                        continue
                    yield x, y, d


