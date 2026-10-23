"""Registered, data-only observation encoders for Snake agents."""

from copy import deepcopy
import math

import numpy as np

# Human-readable names for the encoded inputs, in the order the encoders emit them.
BOARD_CHANNELS = ('head', 'food', 'body')
METADATA_FEATURES = (
    'dir L', 'dir R', 'dir U', 'dir D',
    'danger S', 'danger R', 'danger L',
    'food L', 'food R', 'food U', 'food D', 'food dist',
)


def _metadata_features(options):
    # Disabling food_distance omits the normalized head-to-food distance.
    return tuple(name for name in METADATA_FEATURES
                 if name != 'food dist' or options['food_distance'])


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

    return {'board_state': board, 'metadata': _metadata(game, head, grid_width, grid_height, options)}


def _encode_metadata_only(game, grid_width, grid_height, options):
    head = game.snake[0]
    return {'metadata': _metadata(game, head, grid_width, grid_height, options)}


def _metadata(game, head, grid_width, grid_height, options):
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
    features = dict(zip(METADATA_FEATURES, (
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
    ), strict=True))
    return np.array([features[name] for name in _metadata_features(options)], dtype=np.float32)


STATE_ENCODERS = {
    'board_metadata': {
        'encode': _encode_board_metadata,
        'options': {'body_encoding': 'gradient', 'food_distance': True},
        'allowed_options': {'body_encoding', 'food_distance'},
        'board_channels': 3,
        'metadata_features': _metadata_features,
    },
    'metadata_only': {
        'encode': _encode_metadata_only,
        'options': {'food_distance': True},
        'allowed_options': {'food_distance'},
        'board_channels': 0,
        'metadata_features': _metadata_features,
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
    if type(resolved_options['food_distance']) is not bool:
        raise ValueError('food_distance must be true or false')
    return {'encoder': name, 'options': resolved_options}


def metadata_features(representation):
    """Names of the metadata features a representation emits, in order."""
    resolved = resolve_state_representation(representation)
    return STATE_ENCODERS[resolved['encoder']]['metadata_features'](resolved['options'])


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
        'metadata_size': len(metadata_features(resolved)),
    }


class StateLayout:
    """Pack encoded observations into flat float32 vectors and restore model inputs.

    A flat layout lets one host-to-device copy carry every input of a state.
    """

    def __init__(self, state_spec):
        shapes = {}
        if state_spec['has_board']:
            shapes['board_state'] = (state_spec['board_channels'],
                                     state_spec['board_height'], state_spec['board_width'])
        shapes['metadata'] = (state_spec['metadata_size'],)
        self.fields = []
        self.size = 0
        for key, shape in shapes.items():
            stop = self.size + math.prod(shape)
            self.fields.append((key, self.size, stop, shape))
            self.size = stop

    def pack(self, state, out=None):
        if out is None:
            out = np.empty(self.size, dtype=np.float32)
        for key, start, stop, _ in self.fields:
            out[start:stop] = np.ravel(state[key])
        return out

    def unpack(self, flat):
        """Split a (batch, size) tensor into model inputs with a leading batch axis."""
        # Column slices are strided views; MPS convolutions run far slower on them.
        return {key: flat[:, start:stop].reshape(-1, *shape).contiguous()
                for key, start, stop, shape in self.fields}


def encode_state(representation, game, grid_width, grid_height):
    resolved = resolve_state_representation(representation)
    return encode_resolved_state(resolved, game, grid_width, grid_height)


def encode_resolved_state(representation, game, grid_width, grid_height):
    """Encode a configuration already validated by ``resolve_state_representation``."""
    encoder = STATE_ENCODERS[representation['encoder']]['encode']
    return encoder(game, grid_width, grid_height, representation['options'])
