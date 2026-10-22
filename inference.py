"""Play greedy games with a trained run's checkpoint, without exploration or training.

The checkpoint is copied to a private snapshot before loading, so a run can be
evaluated while it is still training and saving to the same files.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import random
import shutil
import statistics
import tempfile

import numpy as np
import torch

from configuration import (
    REPO_ROOT, load_presets, positive_int, resolve_preset, start_length_arg, validate_name,
)
import game as game_module
from game import BLOCK_SIZE, SnakeGameAI, validate_start_length
from model import ARCHITECTURES, resolve_device
from state import StateLayout, encode_resolved_state, get_observation_spec

# Training boards use the game's default size; the network's input shape depends on it.
BOARD_WIDTH, BOARD_HEIGHT = 640, 480


def snapshot_checkpoint(path):
    """Load a private copy of ``path`` so concurrent saves cannot change it mid-read."""
    with tempfile.TemporaryDirectory(prefix='snake-inference-') as directory:
        copy = Path(directory) / path.name
        # Training replaces checkpoints atomically, so the copy is one complete save.
        shutil.copyfile(path, copy)
        return torch.load(copy, map_location='cpu', weights_only=True)


def resolve_checkpoint(args, presets):
    """Return the checkpoint path and the config used to rebuild its network."""
    if args.checkpoint in ('latest', 'best'):
        if args.run is None and args.model is None:
            raise ValueError('Select a run with --run or --model, or pass a checkpoint path')
        run_dir = REPO_ROOT / 'runs' / validate_name(args.run or args.model)
        path = run_dir / f'{args.checkpoint}.pth'
    else:
        run_dir = None
        path = Path(args.checkpoint)
    if not path.is_file():
        raise ValueError(f'Checkpoint does not exist: {path}')

    # Prefer the config saved with the run; fall back to the named preset.
    if run_dir is not None and (run_dir / 'config.json').is_file():
        config = json.loads((run_dir / 'config.json').read_text())
    elif args.model in presets:
        config = presets[args.model]
    elif args.model is not None:
        raise ValueError(f'Unknown preset {args.model!r}; available: ' + ', '.join(presets))
    else:
        config = None
    return path, config


def load_model(checkpoint, config, device, grid_width, grid_height):
    """Rebuild the network from the checkpoint's own config when it has one."""
    wrapped = isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
    saved_config = checkpoint.get('config') if wrapped else None
    if saved_config is not None:
        config = saved_config
    if config is None:
        raise ValueError('Checkpoint has no saved config; select its preset with --model')
    config = resolve_preset(config)

    state_spec = get_observation_spec(config['state_representation'], grid_height, grid_width)
    model = ARCHITECTURES[config['architecture']](
        board_height=grid_height,
        board_width=grid_width,
        metadata_size=state_spec['metadata_size'],
        output_size=3,
        state_spec=state_spec,
        **config['architecture_options'],
    )
    try:
        model.load_state_dict(checkpoint['model_state_dict'] if wrapped else checkpoint)
    except RuntimeError as err:
        raise ValueError(f'Checkpoint does not match its config: {err}') from err
    model.to(device).eval()
    metadata = checkpoint.get('metadata', {}) if wrapped else {}
    return model, config, StateLayout(state_spec), metadata


def play(model, config, layout, device, games, render, start_length):
    """Play ``games`` greedy games, in lockstep batches when headless."""
    representation = config['state_representation']
    # Rendering paces every lockstep game to the window clock, so watch one at a time.
    slot_count = 1 if render else min(games, config['training']['parallel_games'])
    boards = [SnakeGameAI(BOARD_WIDTH, BOARD_HEIGHT, render=render, rewards=config['rewards'],
                          start_length=start_length)
              for _ in range(slot_count)]
    grid_width, grid_height = BOARD_WIDTH // BLOCK_SIZE, BOARD_HEIGHT // BLOCK_SIZE
    results = []
    started = slot_count
    active = list(range(slot_count))
    try:
        while active:
            states = [encode_resolved_state(representation, boards[slot], grid_width, grid_height)
                      for slot in active]
            batch = layout.unpack(torch.from_numpy(np.stack([layout.pack(s) for s in states]))
                                  .to(device))
            with torch.inference_mode():
                moves = model(batch).argmax(dim=1).tolist()
            still_active = []
            for slot, move in zip(active, moves):
                board = boards[slot]
                action = [0, 0, 0]
                action[move] = 1
                _, done, score = board.play_step(action)
                if not done:
                    still_active.append(slot)
                    continue
                results.append((score, board.end_reason, board.frame_iteration))
                print(f'Game {len(results)}  Score {score}  Steps {board.frame_iteration}  '
                      f'End reason: {board.end_reason}', flush=True)
                if started < games:
                    board.reset()
                    started += 1
                    still_active.append(slot)
            active = still_active
    except KeyboardInterrupt:
        print('\nInterrupted; unfinished games are discarded.')
    return results


def summarize(results):
    if not results:
        print('No games completed.')
        return
    scores = [score for score, _, _ in results]
    reasons = Counter(reason for _, reason, _ in results)
    print(f'\n{len(results)} game(s)  Mean {statistics.fmean(scores):.2f}  '
          f'Median {statistics.median(scores):g}  Min {min(scores)}  Max {max(scores)}')
    print('End reasons: ' + ', '.join(f'{reason} {count}' for reason, count in reasons.most_common()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='named preset in models.json (also the default run name)')
    parser.add_argument('--run', help='run directory under runs/ (default: --model)')
    parser.add_argument('--checkpoint', default='best',
                        help="'best', 'latest', or a checkpoint file path (default: best)")
    parser.add_argument('--games', type=positive_int, default=100, help='games to play (default: 100)')
    parser.add_argument('--render', action='store_true', help='watch the games one at a time')
    parser.add_argument('--fps', type=positive_int, default=game_module.SPEED,
                        help=f'moves per second when rendering (default: {game_module.SPEED})')
    parser.add_argument('--start-length', type=start_length_arg,
                        help='start each game coiled into a solid block of N segments, or of a length drawn '
                             'from MIN-MAX per game, e.g. 4-30 (default: straight 3)')
    parser.add_argument('--seed', type=int, help='seed food and start placement for repeatable games')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    args = parser.parse_args(argv)

    try:
        validate_start_length(args.start_length, BOARD_WIDTH, BOARD_HEIGHT)
        device = resolve_device(args.device)
        path, config = resolve_checkpoint(args, load_presets())
        checkpoint = snapshot_checkpoint(path)
        model, config, layout, metadata = load_model(
            checkpoint, config, device, BOARD_WIDTH // BLOCK_SIZE, BOARD_HEIGHT // BLOCK_SIZE,
        )
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError, KeyError, TypeError) as err:
        parser.error(str(err))

    trained = (f'trained {metadata["n_games"]} games, saved {metadata.get("saved_at", "?")}'
               if 'n_games' in metadata else 'no training metadata')
    print(f'Checkpoint: {path} ({config["architecture"]}, {trained}) | Device: {device}', flush=True)
    if args.seed is not None:
        random.seed(args.seed)
    game_module.SPEED = args.fps
    try:
        summarize(play(model, config, layout, device, args.games, args.render, args.start_length))
    finally:
        import pygame
        pygame.quit()


if __name__ == '__main__':
    main()
