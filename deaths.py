"""Record each training game's final board as one JSON line for death_viewer.py."""

import json

# Step from one segment to the next one toward the tail, in grid cells.
STEPS = {'R': (1, 0), 'L': (-1, 0), 'D': (0, 1), 'U': (0, -1)}
_LETTERS = {step: letter for letter, step in STEPS.items()}


def death_record(game, game_number, epsilon, epsilon_scale=1.0, random_steps=()):
    """Describe a finished game in grid cells.

    The snake is stored as its head cell plus one direction letter per further
    segment, so a long snake costs about a byte per segment. ``epsilon`` is the
    scheduled exploration rate of the final move and ``epsilon_scale`` the
    ``--epsilon-ramp`` multiplier applied to it. ``random_steps`` lists the move
    numbers, counted from 1 like ``game.frame_iteration``, that were epsilon
    explorations rather than the network's choice.
    """
    from game import BLOCK_SIZE

    def cell(point):
        return int(point.x // BLOCK_SIZE), int(point.y // BLOCK_SIZE)

    cells = [cell(point) for point in game.snake]
    body = ''.join(_LETTERS[(x2 - x1, y2 - y1)]
                   for (x1, y1), (x2, y2) in zip(cells, cells[1:]))
    # Every move adds a head and the snake never shrinks, so the segment at index i
    # was placed by move steps - i; segments past the last move are the spawn body.
    steps = game.frame_iteration
    random_segments = sorted(steps - step for step in random_steps if steps - step < len(cells))
    return {
        'game': game_number,
        'score': game.score,
        'start_score': game.start_score,
        'end_reason': game.end_reason,
        'steps': steps,
        'epsilon': epsilon,
        'epsilon_scale': epsilon_scale,
        'random_move': steps in random_steps,
        'random_moves': len(random_steps),
        'random_segments': random_segments,
        'board': [game.w // BLOCK_SIZE, game.h // BLOCK_SIZE],
        'food': None if game.food is None else list(cell(game.food)),
        'head': list(cells[0]),
        'body': body,
    }


def append_death(path, record):
    with open(path, 'a') as handle:
        handle.write(json.dumps(record, separators=(',', ':')) + '\n')


def read_deaths(path, offset=0):
    """Return the records on complete lines after byte ``offset`` and the offset to resume from.

    A line training is still writing is left for the next read.
    """
    try:
        with open(path, 'rb') as handle:
            handle.seek(offset)
            data = handle.read()
    except FileNotFoundError:
        return [], offset
    end = data.rfind(b'\n') + 1
    return [json.loads(line) for line in data[:end].splitlines() if line.strip()], offset + end


def snake_cells(record):
    """Decode a record's snake into grid cells, head first."""
    x, y = record['head']
    cells = [(x, y)]
    for letter in record['body']:
        dx, dy = STEPS[letter]
        x, y = x + dx, y + dy
        cells.append((x, y))
    return cells
