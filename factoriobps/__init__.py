

import zlib
import base64
import json

from typing import Iterable

__all__ = ['load_blueprint_string', 'load_blueprint_string_from_file', 'is_book_data', 'is_blueprint_data', 'blueprints_in_book', 'get_entity_set']

def load_blueprint_string(bp: str) -> dict:
    '''
    JSON decode a blueprint string.
    '''
    try:
        return json.loads(zlib.decompress(base64.b64decode(bp[1:].encode('utf-8'))))
    except Exception:
        raise Exception('Blueprint string is invalid.')

def load_blueprint_string_from_file(fpath: str) -> dict:
    '''
    JSON decode a blueprint string from a file.
    '''
    with open(fpath, 'r') as f:
        return load_blueprint_string(f.readline().strip())

def get_blueprint_string(bp_data: dict) -> str:
    '''
    JSON encode blueprint data `bp_data`.
    '''
    return '0' + base64.b64encode(zlib.compress(json.dumps(bp_data).encode('utf-8'))).decode('utf-8')

def print_bp_data(bp_data: dict) -> None:
    '''
    Pretty print a blueprint data.
    '''
    print(json.dumps(bp_data, indent=4))

def is_book_data(bp_data: dict) -> bool:
    '''
    Return whether `bp_data` represents a blueprint book.
    '''
    return 'blueprint_book' in bp_data

def is_blueprint_data(bp_data: dict) -> bool:
    '''
    Return whether `bp_data` represents a blueprint.
    '''
    return 'blueprint' in bp_data

def blueprints_in_book(bp_data: dict, top: bool = False) -> Iterable[dict]:
    '''
    Iterate over all non-book blueprints in `bp_data`.

    If `top=True`, then only top-level blueprints are yielded.
    If `top=False`, then all blueprints in the book and blueprints of any nested books are yielded.
    '''
    assert is_book_data(bp_data), f'The provided blueprint data does not represent a blueprint book.'
    for child_bp_data in bp_data['blueprint_book']['blueprints']:
        if is_blueprint_data(child_bp_data):
            yield child_bp_data
        if not top and is_book_data(child_bp_data):
            yield from blueprints_in_book(child_bp_data, top=False)

def get_entity_set(bp_data: dict) -> set[str]:
    '''
    Return the set of all entity types that `bp_data` contains.
    '''
    if is_book_data(bp_data):
        return set(e['name'] for bp in blueprints_in_book(bp_data) for e in bp['blueprint']['entities'])
    elif is_blueprint_data(bp_data):
        return set(e['name'] for e in bp_data['blueprint']['entities'])

def get_entity_map(bp_data: dict) -> dict[tuple[float, float], tuple[str, tuple[int, ...]]]:
    belts = []
    splitters = []
    undergrounds = []
    for entity in bp_data['blueprint']['entities']:
        if 'splitter' in entity['name']:
            f = ''
            if 'output_priority' in entity and 'filter' in entity and entity['filter'] == 'deconstruction-planner':
                f = entity['output_priority']
            splitters.append((
                entity['position']['x'], entity['position']['y'],
                entity['direction'] // 2 if 'direction' in entity else 0,
                f
            ))
        elif 'underground' in entity['name'] and 'belt' in entity['name']:
            undergrounds.append((
                entity['position']['x'], entity['position']['y'],
                entity['direction'] // 2 if 'direction' in entity else 0,
                entity['type']
            ))
        elif 'belt' in entity['name']:
            belts.append((entity['position']['x'], entity['position']['y'], entity['direction'] // 2 if 'direction' in entity else 0))

    entity_map = {}

    for x, y, d in belts:
        entity_map[(x, y)] = ('belt', (d,))
    for x, y, d, io_type in undergrounds:
        entity_map[(x, y)] = ('underground', (d, io_type))
    for i, (x, y, d, f) in enumerate(splitters):
        lf = f == 'left'
        rf = f == 'right'
        if d == 2 or d == 3:
            lf, rf = rf, lf
        if d == 0 or d == 2:
            entity_map[(x-0.5, y)] = ('splitter', (d, i, (x+0.5, y), (lf, rf)))
            entity_map[(x+0.5, y)] = ('splitter', (d, i, (x-0.5, y), (rf, lf)))
        else:
            entity_map[(x, y-0.5)] = ('splitter', (d, i, (x, y+0.5), (lf, rf)))
            entity_map[(x, y+0.5)] = ('splitter', (d, i, (x, y-0.5), (rf, lf)))
    
    return entity_map

def transpose_balancer_bp(bp_data: dict):
    '''
    Transpose a balancer blueprint such that all splitters, and underground belts, and belts are in reversed direction.
    '''
    assert is_blueprint_data(bp_data), f'The provided blueprint data does not represent a blueprint.'
    entity_map = get_entity_map(bp_data)

    # First determine the output tile of each belt, splitter, and underground belt.
    # This will be used to flip the direction of each belt, but not the splitters or underground belts.
    output_map = {}
    for (x, y), (entity_type, entity_data) in entity_map.items():
        if entity_type == 'belt':
            dx, dy = 0, -1
            if entity_data[0] == 1:
                dx, dy = 1, 0
            elif entity_data[0] == 2:
                dx, dy = 0, 1
            elif entity_data[0] == 3:
                dx, dy = -1, 0
            output_map[(x + dx, y + dy)] = entity_data[0]
        elif entity_type == 'splitter':
            d, i, (x1, y1), (lf, rf) = entity_data
            dx, dy = 0, -1
            if d == 1:
                dx, dy = 1, 0
            elif d == 2:
                dx, dy = 0, 1
            elif d == 3:
                dx, dy = -1, 0
            output_map[(x + dx, y + dy)] = d
        elif entity_type == 'underground':
            d, io_type = entity_data
            if io_type == 'input':
                continue
            dx, dy = 0, -1
            if d == 1:
                dx, dy = 1, 0
            elif d == 2:
                dx, dy = 0, 1
            elif d == 3:
                dx, dy = -1, 0
            output_map[(x + dx, y + dy)] = d
    # Next, use the output_map to reverse the direction of each belt, and also reverse the direction of each splitter and underground belt (which doesn't need output_map).
    for entity in bp_data['blueprint']['entities']:
        if 'direction' not in entity:
            entity['direction'] = 0
        if 'transport-belt' in entity['name']:
            x, y = entity['position']['x'], entity['position']['y']
            # If output_map doesn't contain (x, y), then this belt is an input belt and should be reversed by rotating 180 degrees.
            if (x, y) in output_map:
                entity['direction'] = (output_map[(x, y)] * 2 + 4) % 8
            else:
                entity['direction'] = (entity['direction'] + 4) % 8
        elif 'splitter' in entity['name'] or 'underground-belt' in entity['name']:
            if 'direction' not in entity:
                entity['direction'] = 0
            entity['direction'] = (entity['direction'] + 4) % 8
            if 'underground-belt' in entity['name']:
                entity['type'] = 'output' if entity['type'] == 'input' else 'input'
    return bp_data

if __name__ == '__main__':
    # Test transpose_balancer_bp()
    bp = load_blueprint_string('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')
    print_bp_data(bp)
    transpose_balancer_bp(bp)
    bp_str = get_blueprint_string(bp)
    print(bp_str)
