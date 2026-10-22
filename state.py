"""Registered, data-only observation encoders for Snake agents."""

from copy import deepcopy

import numpy as np

def _encode_board_metadata(game, grid_width, grid_height, options):
    from game import BLOCK_SIZE

    head = game.snake[0]
    snake_len = len(game.snake)
    board = np.zeros((3, grid_height, grid_width), dtype=np.float32)

    hx, hy = int(head.x // BLOCK_SIZE), int(head.y // BLOCK_SIZE)
    if 0 <= hx < grid_width and 0 <= hy < grid_height:
        board[0, hy, hx] = 1.0

    fx, fy = int(game.food.x // BLOCK_SIZE), int(game.food.y // BLOCK_SIZE)
    if 0 <= fx < grid_width and 0 <= fy < grid_height:
        board[1, fy, fx] = 1.0

    # The gradient runs from 1 at the head to 0.5 at the tail, so every
    # segment, including the tail, stays distinct from empty cells.
    denominator = max(1, snake_len - 1)
    for index, segment in enumerate(game.snake):
        x, y = int(segment.x // BLOCK_SIZE), int(segment.y // BLOCK_SIZE)
        if 0 <= x < grid_width and 0 <= y < grid_height:
            value = 1.0 if options['body_encoding'] == 'binary' else 1.0 - 0.5 * index / denominator
            board[2, y, x] = value

    return {'board_state': board, 'metadata': _metadata(game, head, grid_width, grid_height)}


def _encode_metadata_only(game, grid_width, grid_height, _options):
    head = game.snake[0]
    return {'metadata': _metadata(game, head, grid_width, grid_height)}


def _metadata(game, head, grid_width, grid_height):
    from game import BLOCK_SIZE, Direction, Point

    direction_order = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    index = direction_order.index(game.direction)
    direction_straight = direction_order[index]
    direction_right = direction_order[(index + 1) % 4]
    direction_left = direction_order[(index - 1) % 4]

    def next_point(point, direction):
        x, y = point.x, point.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        return Point(x, y)

    max_distance = grid_width + grid_height
    food_distance = (
        abs(head.x - game.food.x) + abs(head.y - game.food.y)
    ) / (BLOCK_SIZE * max_distance)
    return np.array([
        int(game.direction == Direction.LEFT),
        int(game.direction == Direction.RIGHT),
        int(game.direction == Direction.UP),
        int(game.direction == Direction.DOWN),
        int(game.is_collision(next_point(head, direction_straight))),
        int(game.is_collision(next_point(head, direction_right))),
        int(game.is_collision(next_point(head, direction_left))),
        int(game.food.x < head.x),
        int(game.food.x > head.x),
        int(game.food.y < head.y),
        int(game.food.y > head.y),
        food_distance,
    ], dtype=np.float32)


STATE_ENCODERS = {
    'board_metadata': {
        'encode': _encode_board_metadata,
        'options': {'body_encoding': 'gradient'},
        'allowed_options': {'body_encoding'},
        'board_channels': 3,
        'metadata_size': 12,
    },
    'metadata_only': {
        'encode': _encode_metadata_only,
        'options': {},
        'allowed_options': set(),
        'board_channels': 0,
        'metadata_size': 12,
    },
}


DEFAULT_STATE_REPRESENTATION = {
    'encoder': 'board_metadata',
    'options': {'body_encoding': 'gradient'},
}


def resolve_state_representation(representation=None):
    """Return a complete state encoder config and reject unregistered choices."""
    if representation is None:
        representation = DEFAULT_STATE_REPRESENTATION
    if not isinstance(representation, dict):
        raise ValueError('state_representation must be a JSON object')
    unknown = representation.keys() - {'encoder', 'options'}
    if unknown:
        raise ValueError('Unknown state_representation settings: ' + ', '.join(sorted(unknown)))

    name = representation.get('encoder', 'board_metadata')
    if not isinstance(name, str) or name not in STATE_ENCODERS:
        raise ValueError(f'Unknown state encoder {name!r}; choose from {", ".join(STATE_ENCODERS)}')
    encoder = STATE_ENCODERS[name]
    options = representation.get('options', {})
    if not isinstance(options, dict):
        raise ValueError('state_representation options must be a JSON object')
    unknown_options = options.keys() - encoder['allowed_options']
    if unknown_options:
        raise ValueError('Unknown state encoder options: ' + ', '.join(sorted(unknown_options)))
    resolved_options = deepcopy(encoder['options']) | deepcopy(options)
    if name == 'board_metadata' and resolved_options['body_encoding'] not in ('gradient', 'binary'):
        raise ValueError('body_encoding must be gradient or binary')
    return {'encoder': name, 'options': resolved_options}


def get_observation_spec(representation, board_height, board_width):
    """Describe the tensors an encoder returns without allocating unused inputs."""
    resolved = resolve_state_representation(representation)
    encoder = STATE_ENCODERS[resolved['encoder']]
    has_board = encoder['board_channels'] > 0
    return {
        'has_board': has_board,
        'board_channels': encoder['board_channels'],
        'board_height': board_height if has_board else 0,
        'board_width': board_width if has_board else 0,
        'metadata_size': encoder['metadata_size'],
    }


def encode_state(representation, game, grid_width, grid_height):
    resolved = resolve_state_representation(representation)
    return encode_resolved_state(resolved, game, grid_width, grid_height)


def encode_resolved_state(representation, game, grid_width, grid_height):
    """Encode a configuration already validated by ``resolve_state_representation``."""
    encoder = STATE_ENCODERS[representation['encoder']]['encode']
    return encoder(game, grid_width, grid_height, representation['options'])
