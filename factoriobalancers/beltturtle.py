
import os
import numpy as np
from typing import Generator
from fischer.math.common import l1_distance, lerp_vector, l2_distance
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

def get_splitter_positions(x: int, y: int, d: int) -> tuple[tuple[int, int], tuple[int, int]]:
    '''
    Return the positions that the splitter occupies.
    '''
    return (x, y), get_splitter_other_position(x, y, d)

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
        if self.grid[y, x] % 4 == d:
            return
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
        # Check ahead of the input position to see if it is open.
        if not self.is_passable(*offset_position(x, y, d), d):
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
        # Check behind the output position to see if it is open.
        if not self.is_passable(*offset_position(x, y, (d + 2) % 4), d):
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
    def blocks_splitter_output(self, x: int, y: int, d: int, underground_exit: bool = False) -> bool:
        '''
        Return whether a belt, splitter, or underground at `(x, y)` facing `d` blocks any splitter outputs.

        If `underground_exit = True`, then the function will return `True` if the entity at `(x, y)` is an underground exit, even if it is facing `d`, since that would actually be blocking the output.
        '''
        assert is_valid_position(x, y)
        for sx, sy, sd in self.splitters:
            if (x, y) in get_splitter_output_positions(sx, sy, sd):
                if d != sd or underground_exit:
                    return True
        return False
    def blocks_splitter_input(self, x: int, y: int, d: int, underground_entrance: bool = False) -> bool:
        '''
        Return whether a belt, splitter, or underground at `(x, y)` facing `d` blocks any splitter inputs.

        If `underground_entrance = True`, then the function will return `True` if the entity at `(x, y)` is an underground entrance, even if it is facing `d`, since that would actually be blocking the input.
        '''
        assert is_valid_position(x, y)
        for sx, sy, sd in self.splitters:
            if (x, y) in get_splitter_input_positions(sx, sy, sd):
                if underground_entrance or d != sd:
                    return True
        return False
    def get_splitter_position_from_splitter_input_position(self, x: int, y: int) -> tuple[int, int, int]:
        '''
        Return a splitter's position and direction given its input position.
        '''
        for sx, sy, sd in self.splitters:
            for _sx, _sy in get_splitter_positions(sx, sy, sd):
                if (x, y) == offset_position(_sx, _sy, (sd + 2) % 4):
                    return _sx, _sy, sd
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
        if turtle in self.turtles:
            self.turtles.remove(turtle)
    def get_turtle_at(self, x: int, y: int) -> 'BeltTurtle':
        '''
        Return the turtle at `(x, y)`.
        '''
        assert is_valid_position(x, y)
        for turtle in self.turtles:
            if turtle.x == x and turtle.y == y:
                return turtle
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
        grid = self.grid.copy()
        for sx, sy, sd in self.splitters:
            x, y = get_splitter_other_position(sx, sy, sd)
            grid[y, x] = -1
        entities: list[dict] = [
            {
                'entity_number': (i := i + 1),
                'name': 'express-transport-belt' if grid[y, x] < 4 else 'express-underground-belt',
                'position': offset_pos(x, y),
                'direction': int(grid[y, x] % 4 * 2 + 2), # Cast to int to avoid numpy int32, which is not JSON serializable.
            } | ({ # This is the dictionary operator for calling update(), as in `a.update(b)` <=> `a |= b`.
                'type': 'output' if self.is_underground_exit(x, y) else 'input'
            } if 4 <= grid[y, x] < 8 else {})
            if grid[y, x] < 8 else
            {'entity_number': (i := i + 1), 'name': 'express-splitter', 'position': splitter_pos(x, y), 'direction': int((grid[y, x] - 7) * 2)}
            for y in range(self.height)
            for x in range(self.width)
            if grid[y, x] >= 0
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
    def generate_from_graph(self, graph: 'BeltGraph', max_underground_length: int = 8, splitter_positions: dict[int, tuple[int, int, int]] = None) -> bool:
        '''
        Generate a belt grid from `graph`.  While it is guaranteed to work for a sufficiently large grid, it is never compact.

        The exact width of the grid needed is calculated as `6V - 2` where `V` is the number of vertices in `graph`.  For example, 4 vertices require 22 width.
        
        The exact height of the grid needed is not known for this algorithm, but it generally follows the number of internal vertices logarithmically, like `log2(V) + 4`.  For example, 4 vertices require around 6 height, 8 vertices require around 7 height, etc.

        Return whether the generation was successful.
        '''
        if splitter_positions is None:
            # Place splitters at spaced positions in a straight line.  This requires the grid to be sufficiently long but not necessarily wide.
            # graph = graph.rearrange_vertices_by_dfs()
            splitter_positions = {}
            h2 = self.height // 2 - 1
            for n, v in enumerate(graph.internal_vertices()):
                x = 1 + 6 * n
                self.set_splitter(x, h2, 0)
                splitter_positions[v] = x, h2, 0

        # A-star from one splitter to the next.
        for u, v in graph.internal_edges():
            self.add_turtle(0, 0, 0, ignore=True, set_direction_immediately=False)
            paths = []
            d0 = splitter_positions[u][2]
            d1 = splitter_positions[v][2]
            for x0, y0 in get_splitter_positions(*splitter_positions[u]):
                if not self.is_splitter_connectable_forward(x0, y0):
                    continue
                for x1, y1 in get_splitter_positions(*splitter_positions[v]):
                    if not self.is_splitter_connectable_backward(x1, y1):
                        continue
                    self.turtles[-1].set_start_position(x0, y0, d0, clear_path_on_grid=True)
                    if self.turtles[-1].astar(x1, y1, d1, max_underground_length=max_underground_length):
                        paths.append(self.turtles[-1].path.copy())
                self.turtles[-1].reset(clear_path_on_grid=True)
            if not paths:
                return False
            best_path = min(paths, key=len)
            self.turtles[-1].load_path(best_path)
            self.remove_turtle(self.turtles[-1])

        def unblocked_input(p0: tuple[int, int, int], p1: tuple[int, int, int]) -> tuple[int, int, int]:
            if self.can_set_input(*p0):
                return p0
            if self.can_set_input(*p1):
                return p1
        for u in graph.inputs:
            v, = graph.out_vertices(u)
            (x0, y0), (x1, y1) = get_splitter_input_positions(*splitter_positions[v])
            p = unblocked_input((x0, y0, 0), (x1, y1, 0))
            if p is None:
                return False
            self.set_input(*p)
        def unblocked_output(p0: tuple[int, int, int], p1: tuple[int, int, int]) -> tuple[int, int, int]:
            if self.can_set_output(*p0):
                return p0
            if self.can_set_output(*p1):
                return p1
        for v in graph.outputs:
            u, = graph.in_vertices(v)
            (x0, y0), (x1, y1) = get_splitter_output_positions(*splitter_positions[u])
            p = unblocked_output((x0, y0, 0), (x1, y1, 0))
            if p is None:
                return False
            self.set_output(*p)
        return True
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
    # BELT_CHARS = '\u2192', '\u2191', '\u2190', '\u2193'
    # BELT_CHARS = '\u2b46', '\u27f0', '\u2b45', '\u27f1'
    BELT_CHARS = '\u2192', '\u2191', '\u2190', '\u2193'
    # UNDERGROUND_CHARS = '\u21e2', '\u21e1', '\u21e0', '\u21e3'
    UNDERGROUND_CHARS = BELT_CHARS
    # SPLITTER_CHARS = '\u21e8', '\u21e7', '\u21e6', '\u21e9'
    SPLITTER_CHARS = BELT_CHARS
    # EMPTY_CHAR = '\u274f'
    EMPTY_CHAR = chr(215) # "x" multiplication symbol
    BELT_COLOR = colorama.Fore.WHITE
    INPUT_COLOR = colorama.Fore.WHITE
    OUTPUT_COLOR = colorama.Fore.LIGHTRED_EX
    SPLITTER_COLOR = colorama.Fore.CYAN
    UNDERGROUND_ENTRANCE_COLOR = colorama.Fore.GREEN
    UNDERGROUND_EXIT_COLOR = colorama.Fore.YELLOW
    TURTLE_COLOR = colorama.Fore.MAGENTA
    EMPTY_COLOR = colorama.Fore.BLACK
    def __str__(self) -> str:
        def wrap_color(s: str, x: int, y: int) -> str:
            if self.turtle_mask[y, x]:
                return self.TURTLE_COLOR + s + colorama.Fore.RESET
            if self.is_input(x, y):
                return self.INPUT_COLOR + s + colorama.Fore.RESET
            if self.is_output(x, y):
                return self.OUTPUT_COLOR + s + colorama.Fore.RESET
            if self.is_underground(x, y):
                if self.is_underground_exit(x, y):
                    return self.UNDERGROUND_EXIT_COLOR + s + colorama.Fore.RESET
                return self.UNDERGROUND_ENTRANCE_COLOR + s + colorama.Fore.RESET
            if self.is_splitter(x, y):
                return self.SPLITTER_COLOR + s + colorama.Fore.RESET
            if self.turtle_mask[y, x]:
                return self.TURTLE_COLOR + s + colorama.Fore.RESET
            if self.is_open(x, y):
                return self.EMPTY_COLOR + s + colorama.Fore.RESET
            return self.BELT_COLOR + s + colorama.Fore.RESET
        return '\n'.join(
            ' '.join(
                wrap_color((
                    self.BELT_CHARS[v]
                    if v >= 0 and v < 4 else
                    self.UNDERGROUND_CHARS[v - 4]
                    if v >= 4 and v < 8 else
                    self.SPLITTER_CHARS[v - 8]
                    if v >= 8 else
                    self.EMPTY_CHAR
                ), x, y)
                for x, v in enumerate(r)
            )
            for y, r in reversed(list(enumerate(self.grid)))
        )

class BeltTurtle:
    def __init__(self, grid: BeltGrid, x: int, y: int, direction: int = 0):
        self.x = -1
        self.y = -1
        self.direction = -1
        self.start_x = -1
        self.start_y = -1
        self.start_direction = -1
        self.grid = grid
        self.path: list[tuple[int, int, int]] = []
        self.set_start_position(x, y, direction)
    @property
    def position(self) -> tuple[int, int, int]:
        return self.x, self.y, self.direction
    @property
    def start_position(self) -> tuple[int, int, int]:
        return self.start_x, self.start_y, self.start_direction
    def reset(self, clear_path_on_grid: bool = True):
        '''
        Clear the current path and reset the position to its starting position as defined in the constructor of `BeltTurtle`.
        '''
        if clear_path_on_grid:
            while len(self.path) > 1:
                self.backtrack()
        else:
            for x, y, _ in self.path:
                self.grid.turtle_mask[y, x] = False
        self.grid.turtle_mask[self.y, self.x] = False
        self.x = self.start_x
        self.y = self.start_y
        self.direction = self.start_direction
        self.grid.turtle_mask[self.y, self.x] = True
        if not self.grid.is_splitter(self.x, self.y) and not self.grid.is_input(self.x, self.y) and not self.grid.is_underground_exit(self.x, self.y):
            self.grid.grid[self.y, self.x] = -1 # This indicates that the turtle hasn't yet "decided" where to go yet, allowing branching off in a different direction or retrying undergrounds.
        self.path.clear()
        self.path.append(self.position)
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
    def set_start_position(self, x: int, y: int, d: int, clear_path_on_grid: bool = False):
        '''
        Set the start position and direction of the turtle.
        '''
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        self.start_x = x
        self.start_y = y
        self.start_direction = d
        self.reset(clear_path_on_grid=clear_path_on_grid)
    def set_position(self, x: int, y: int, d: int, max_underground_length: int = 8) -> bool:
        '''
        Return whether the turtle successfully stepped.
        '''
        assert is_valid_position(x, y)
        assert is_valid_direction(d)
        assert x != self.x or y != self.y
        # if not self.grid.is_passable(x, y, d) and not (self.grid.is_output(x, y) and d == self.grid.grid[y, x]):
        #     return False
        if not (x, y, d) in self.possible_steps(max_underground_length=max_underground_length):
            raise ValueError(f'Invalid step: {self.position} -> {x, y, d}')
        is_underground = l1_distance(x, y, self.x, self.y) > 1
        self.grid.turtle_mask[self.y, self.x] = False
        v = d + 4 * int(is_underground)
        if len(self.path) > 1:
            self.grid.underground_types[self.y, self.x] = l1_distance(self.x, self.y, *self.path[-2][:2]) > 1
        if self.grid.grid[self.y, self.x] < 4:
            self.grid.grid[self.y, self.x] = v
        if self.grid.grid[y, x] < 4:
            self.grid.grid[y, x] = v
        self.x = x
        self.y = y
        self.grid.turtle_mask[self.y, self.x] = True
        self.direction = d
        self.path.append((self.x, self.y, self.direction))
        return True
    def backtrack(self):
        '''
        Backtrack one step.  Return whether the turtle successfully backtracked.
        '''
        if len(self.path) < 2:
            # print(self.path)
            return False
        self.path.pop()
        prev_x, prev_y, prev_d = self.path[-1]
        # print(f'current pos: {self.x, self.y, self.direction}; backtracking to {prev_x, prev_y, prev_d}; path after pop: {self.path}')
        self.grid.underground_types[self.y, self.x] = False
        self.grid.turtle_mask[self.y, self.x] = False
        if not self.grid.is_splitter(self.x, self.y):
            self.grid.grid[self.y, self.x] = -1
        self.x = prev_x
        self.y = prev_y
        self.direction = prev_d
        self.grid.turtle_mask[self.y, self.x] = True
        if not self.grid.is_splitter(self.x, self.y) and not self.grid.is_input(self.x, self.y) and not self.grid.is_underground_exit(self.x, self.y):
            self.grid.grid[self.y, self.x] = -1 # This indicates that the turtle hasn't yet "decided" where to go yet, allowing branching off in a different direction or retrying undergrounds.
        return True
    def is_along_path(self, x: int, y: int, d: int) -> bool:
        '''
        Return whether the position and direction `(x, y, d)` is along the current path.

        Returns True when `(x, y, d)` is directly on the path, as well as for when an underground belt crosses over `(x, y, d)`.
        '''
        if not self.path:
            return False
        prev_x, prev_y, prev_d = self.path[0]
        if prev_x == x and prev_y == y and prev_d == d:
            return True
        for px, py, pd in self.path[1:]:
            if px == x and py == y and pd == d:
                return True
            if l1_distance(px, py, prev_x, prev_y) > 1 and (prev_x < x < px or prev_y < y < py):
                return True
            prev_x, prev_y, prev_d = px, py, pd
        return False
    def load_path(self, path: list[tuple[int, int, int]]) -> None:
        '''
        Load a path into the turtle, overwriting the current path.
        '''
        if not path:
            self.reset()
            return
        if not self.path:
            self.set_start_position(*path[0])
            for x, y, d in path[1:]:
                self.set_position(x, y, d)
            return
        for i, (p0, p1) in enumerate(zip(self.path, path)):
            if p0 != p1:
                break
        i -= 1
        if i >= 0:
            backtrack_pos = path[i]
            for _ in range(len(self.path)):
                if not self.backtrack() or self.position == backtrack_pos:
                    break
        else:
            self.set_start_position(*path[0], clear_path_on_grid=True)
        for x, y, d in path[i+1:]:
            if (x, y, d) != self.position:
                self.set_position(x, y, d)
    def possible_steps(self, max_underground_length: int) -> Generator[tuple[int, int, int], None, None]:
        '''
        Iterate over all possible positions and directions as tuples of the form `(x, y, d)` which the turtle can enter from its current position and direction.

        The maximum number of position-direction tuples iterated is equal to `3 + max_underground_length`.
        '''
        assert is_valid_max_underground_length(max_underground_length)
        def is_step_valid(x: int, y: int, d: int, underground_entrance: bool = False, underground_exit: bool = False) -> bool:
            if x == self.x and y == self.y:
                return False
            if not self.grid.is_within_bounds(x, y):
                return False
            if self.grid.is_turtle(x, y):
                return False
            if self.grid.is_input(x, y):
                return False
            if self.grid.is_output(x, y) and d != self.grid.grid[y, x]:
                return False
            if 0 <= self.grid.grid[y, x] < 4 and not (self.grid.is_output(x, y) and self.grid.grid[y, x] == d):
                return False
            if not self.grid.is_passable(x, y, d):
                return False
            if self.grid.is_underground_exit(x, y):
                return False
            if self.grid.blocks_splitter_output(x, y, d, underground_exit=underground_exit):
                # This can be allowed in the case of the splitter needing only one output, but this is not implemented yet.
                return False
            # Instead of below, you must check whether the constructed belt will actually be orthogonal to the splitter input position, because the turtle's current position is only tentative as it can turn into the input splitter even though it can come in orthogonally (this is a false catch with the code below).
            # if self.grid.blocks_splitter_input(x, y, d, underground_entrance=underground_entrance):
            #     # This can be allowed in the case of the splitter needing only one input, but this is not implemented yet.
            #     return False
            # correct implementation is handled outside of this function
            return True
        if self.grid.grid[self.y, self.x] >= 8:
            if self.grid.grid[self.y, self.x] - 8 == self.direction:
                x, y = offset_position(self.x, self.y, self.grid.grid[self.y, self.x] - 8)
                d = self.grid.grid[self.y, self.x] - 8
                if is_step_valid(x, y, d):
                    yield x, y, d
            return
        if self.grid.blocks_splitter_input(self.x, self.y, self.direction, underground_entrance=True):
            # While it seems incorrect to use underground_entrance=True here, it is used to force blocks_splitter_input() to return True even when the turtle is facing the splitter at one of its input positions.
            # This check in this particular case is important because we cannot let the turtle step away from the input position of the splitter, creating a belt that blocks the splitter's input.
            sx, sy, sd = self.grid.get_splitter_position_from_splitter_input_position(self.x, self.y)
            # TODO Handle this case for ALL splitters that have this position as an input.  It is possible but a rare scenario for belt balancers.
            if not self.grid.is_underground_exit(self.x, self.y) or self.grid.grid[self.y, self.x] - 4 == sd:
                # Make sure that underground exits cannot turn directly into splitters.
                if (self.direction + 2) % 4 != sd: # Avoid checking the opposite direction in case the turtle is drawing a backward belt.
                    yield sx, sy, sd
                    return
        turns = -1, 1, 0 # not (-1, 0, 1) solely because (-1, 1, 0) groups the non-underground steps at the beginning of the iterator, potentially simplifying other logic
        if self.grid.grid[self.y, self.x] >= 4 and self.grid.grid[self.y, self.x] < 8:
            turns = 0,
        for turn in turns:
            new_d = (self.direction + turn) % 4
            new_x, new_y = offset_position(self.x, self.y, new_d)
            if is_step_valid(new_x, new_y, new_d):
                yield new_x, new_y, new_d
            if turn == 0 and self.grid.grid[self.y, self.x] < 4 and not self.grid.is_input(self.x, self.y):
                # Check for underground belts.
                cont = True
                x, y = self.x, self.y
                for _ in range(max_underground_length + 2):
                    x, y = offset_position(x, y, self.direction)
                    if not self.grid.is_within_bounds(x, y):
                        break
                    if self.grid.is_underground(x, y) and self.grid.grid[y, x] % 2 == self.direction % 2:
                        cont = False
                        break
                if cont:
                    for x, y in self.grid.underground_exits(self.x, self.y, self.direction, max_underground_length):
                        d = self.direction
                        if not is_step_valid(x, y, d, underground_exit=True):
                            continue
                        if self.grid.grid[y, x] >= 8 and self.grid.grid[y, x] - 8 == d:
                            break
                        yield x, y, d
    def astar(self, target_x: int, target_y: int, target_d: int, max_underground_length: int = 8) -> bool:
        # Use A-star search to guide the turtle to the target position and direction.
        # The heuristic is the Manhattan distance from the current position to the target position.
        # The cost of moving from one position to another is 1, even when underground belts are used.

        def heuristic(p1: tuple[int, int, int], p2: tuple[int, int, int], underground: bool = False) -> int:
            cx, cy, cd = p1
            tx, ty, td = p2

            if underground:
                cx, cy = offset_position(cx, cy, cd)
            if cx == tx and cy == ty and cd == td:
                return 0
            
            return l2_distance(cx, cy, target_offset_x, target_offset_y)

        def reload_previous_path(prev_open_pos: tuple[int, int, int, int, int, int]):
            # This function is used to reconstruct the path from the current position to the previously observed open position, which may have been disconnected due to backtracking.
            if prev_open_pos not in paths:
                return
            self.load_path(paths[prev_open_pos])

        def record_scores(p_to: tuple[int, int, int], p_from: tuple[int, int, int], g: int) -> None:
            position_pair = (*p_to, *p_from)
            g_score[position_pair] = g
            f_score[position_pair] = g + heuristic(p_to, (target_x, target_y, target_d))
            paths[position_pair] = self.path.copy()
            if position_pair not in closed_set:
                open_set.add(position_pair)

        def debug_log():
            os.system('cls')
            print(self.grid)
            print(f'turtle: {self.x, self.y, self.direction}\ntarget: {target_x, target_y, target_d}')
            fs = list(sorted(open_set - closed_set, key=f_score.__getitem__))
            print('f_score:\n' + '\n'.join(f'{p}: {f_score.get(p, np.inf):.2f} | {paths[p]}' for p in fs[:4]))
            print(f'Possible steps: {list(self.possible_steps(max_underground_length=max_underground_length))}')
            input()

        target_offset_x, target_offset_y = offset_position(target_x, target_y, (target_d + 2) % 4)
        target_splitter_index = self.grid.get_splitter_index_from_position(target_x, target_y) if self.grid.is_splitter(target_x, target_y) else -1
        open_set = {(self.x, self.y, self.direction, self.x, self.y, self.direction)}
        closed_set = set()
        g_score = {(self.x, self.y, self.direction, self.x, self.y, self.direction): 0}
        f_score = {(self.x, self.y, self.direction, self.x, self.y, self.direction): heuristic((self.x, self.y, self.direction), (target_x, target_y, target_d))}
        paths = {(self.x, self.y, self.direction, self.x, self.y, self.direction): self.path.copy()}

        while open_set:
            # debug_log()
            current = min(open_set - closed_set, key=lambda p: f_score.get(p, np.inf))
            open_set.remove(current)
            closed_set.add(current)
            p_to, p_from = current[:3], current[3:]
            # print(f'current: {current}; p_to: {p_to}; p_from: {p_from}; turtle: {self.x, self.y, self.direction}')
            wrong_splitter = False
            if p_from != p_to:
                if p_from != self.position or p_to not in self.possible_steps(max_underground_length=max_underground_length):
                    # print(f'backtracking to {p_from}')
                    # input()
                    reload_previous_path(current)
                if p_to != self.position:
                    self.set_position(*p_to)
                    if (target_x, target_y, target_d) in self.possible_steps(max_underground_length=max_underground_length):
                        self.set_position(target_x, target_y, target_d)
                    if self.position == (target_x, target_y, target_d):
                        # Stop condition.
                        # debug_log()
                        return True
                if self.position != self.start_position and self.grid.is_splitter(self.x, self.y):
                    if self.grid.get_splitter_index_from_position(self.x, self.y) == target_splitter_index:
                        return True
                    wrong_splitter = True
            # debug_log()
            if not wrong_splitter:
                for x, y, d in self.possible_steps(max_underground_length=max_underground_length):
                    # tentative_g_score = g_score[current] + 1
                    tentative_g_score = len(self.path)
                    if tentative_g_score < g_score.get((x, y, d, *p_to), np.inf):
                        record_scores((x, y, d), p_to, tentative_g_score)
            if (target_x, target_y, target_d) == (13, 3, 0):
                debug_log()

        return False
