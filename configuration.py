"""Load and validate data-only model presets before training creates files."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re

from model import ARCHITECTURES, ARCHITECTURE_DEFAULTS, ARCHITECTURE_CAPABILITIES
from reward_defaults import DEFAULT_REWARDS
from state import resolve_state_representation, STATE_ENCODERS


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAINING = {
    'max_memory': 100_000,
    'replay_batch_size': 64,
    'replay_warmup': 1000,
    'train_every': 4,
    'parallel_games': 128,
    'lr': 0.001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_min': 0.001,
    'epsilon_decay': 0.997,
    'max_grad_norm': 5.0,
    'lr_step_size': 500,
    'lr_decay': 0.95,
    'min_lr': 0.00001,
}


def validate_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Names must start with a letter or digit and contain only letters, digits, _ or -')
    return name


# Settings that determine the network and its inputs; a checkpoint can only
# resume under the same values. Rewards and training settings may change.
CHECKPOINT_SHAPE_KEYS = ('architecture', 'architecture_options', 'state_representation')


def tunable_changes(saved, config):
    """List 'section.name: old -> new' for changed rewards and training settings."""
    return [
        f'{section}.{name}: {saved[section].get(name)!r} -> {value!r}'
        for section in ('rewards', 'training')
        for name, value in config[section].items()
        if saved[section].get(name) != value
    ]


def shape_conflicts(saved, config):
    """Names of settings whose change makes an existing checkpoint unusable."""
    return [key for key in CHECKPOINT_SHAPE_KEYS if saved[key] != config[key]]


def positive_int(value):
    try:
        parsed = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError('must be a positive integer') from err
    if parsed < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def start_length_arg(value):
    """Parse ``N`` as a fixed start length or ``MIN-MAX`` as an inclusive range."""
    low, separator, high = value.partition('-')
    if not separator:
        return positive_int(value)
    try:
        bounds = positive_int(low), positive_int(high)
    except argparse.ArgumentTypeError as err:
        raise argparse.ArgumentTypeError('must be N or MIN-MAX with positive integers') from err
    if bounds[0] > bounds[1]:
        raise argparse.ArgumentTypeError('range minimum cannot exceed its maximum')
    return bounds


def validate_epsilon_ramp(epsilon_ramp):
    """Accept ``None`` or a ``(start, stop)`` pair of points gained with 0 <= start < stop."""
    if epsilon_ramp is None:
        return None
    if (not isinstance(epsilon_ramp, tuple) or len(epsilon_ramp) != 2
            or any(type(n) is not int for n in epsilon_ramp)
            or not 0 <= epsilon_ramp[0] < epsilon_ramp[1]):
        raise ValueError(f'epsilon_ramp must be a (start, stop) pair of integers with '
                         f'0 <= start < stop, got {epsilon_ramp!r}')
    return epsilon_ramp


def epsilon_ramp_arg(value):
    """Parse ``START-STOP`` as the points gained where exploration ramps up."""
    low, separator, high = value.partition('-')
    try:
        if not separator:
            raise ValueError
        return validate_epsilon_ramp((int(low), int(high)))
    except ValueError:
        raise argparse.ArgumentTypeError(
            'must be START-STOP with integers 0 <= START < STOP') from None


def _object(value, label, allowed):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a JSON object')
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f'Unknown {label} settings: {", ".join(sorted(unknown))}')


def resolve_preset(preset):
    """Return a new, complete configuration; never mutate the supplied preset."""
    _object(preset, 'preset', {
        'architecture', 'architecture_options', 'training',
        'state_representation', 'rewards',
    })
    architecture = preset.get('architecture')
    if not isinstance(architecture, str) or architecture not in ARCHITECTURES:
        raise ValueError(f'Unknown architecture {architecture!r}; choose from {", ".join(ARCHITECTURES)}')

    defaults = ARCHITECTURE_DEFAULTS[architecture]
    options = preset.get('architecture_options', {})
    _object(options, 'architecture_options', defaults.keys())
    options = deepcopy(defaults) | deepcopy(options)
    if 'hidden_sizes' in options:
        sizes = options['hidden_sizes']
        if not isinstance(sizes, list) or not sizes or any(type(n) is not int or n < 1 for n in sizes):
            raise ValueError('hidden_sizes must be a nonempty array of positive integers')
    if architecture in ('residual_cnn', 'raw_skip_cnn'):
        for name in ('channels', 'residual_blocks', 'board_embedding_size',
                     'metadata_embedding_size', 'fusion_hidden_size'):
            if type(options[name]) is not int or options[name] < 1:
                raise ValueError(f'{name} must be a positive integer')

    state_representation = resolve_state_representation(preset.get('state_representation'))
    if (ARCHITECTURE_CAPABILITIES[architecture]['requires_board']
            and STATE_ENCODERS[state_representation['encoder']]['board_channels'] == 0):
        raise ValueError(f'{architecture} requires a state encoder with board input')

    rewards = preset.get('rewards', {})
    _object(rewards, 'rewards', DEFAULT_REWARDS.keys())
    rewards = DEFAULT_REWARDS | rewards
    for name, value in rewards.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')

    training = preset.get('training', {})
    _object(training, 'training', DEFAULT_TRAINING.keys())
    training = DEFAULT_TRAINING | training
    for name, value in training.items():
        if name == 'min_lr' and value is None:
            continue
        if type(DEFAULT_TRAINING[name]) is int:
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
        elif name in ('gamma', 'epsilon_start', 'epsilon_min'):
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must be between 0 and 1')
        elif name in ('epsilon_decay', 'lr_decay'):
            if not 0 < value <= 1:
                raise ValueError(f'{name} must be greater than 0 and at most 1')
        elif value <= 0:
            raise ValueError(f'{name} must be positive')

    if training['epsilon_min'] > training['epsilon_start']:
        raise ValueError('epsilon_min cannot exceed epsilon_start')
    if training['max_memory'] < max(training['replay_batch_size'], training['replay_warmup']):
        raise ValueError('max_memory must accommodate replay_batch_size and replay_warmup')
    if training['min_lr'] is not None and training['min_lr'] > training['lr']:
        raise ValueError('min_lr cannot exceed lr')
    return {
        'architecture': architecture,
        'state_representation': state_representation,
        'rewards': rewards,
        'architecture_options': options,
        'training': training,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def load_presets(path=REPO_ROOT / 'models.json'):
    with Path(path).open() as config_file:
        presets = json.load(config_file, object_pairs_hook=_unique_object)
    if not isinstance(presets, dict) or not presets:
        raise ValueError('models.json must contain an object with at least one named preset')
    resolved = {}
    for name, preset in presets.items():
        validate_name(name)
        try:
            resolved[name] = resolve_preset(preset)
        except ValueError as err:
            raise ValueError(f'Preset {name!r}: {err}') from err
    return resolved
