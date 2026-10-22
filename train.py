"""Train one JSON model preset in its own persistent run directory."""

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile

from configuration import (
    REPO_ROOT, epsilon_ramp_arg, load_presets, positive_int, resolve_preset, shape_conflicts,
    start_length_arg, tunable_changes, validate_name,
)
from model import resolve_device


def prepare_run(config, run_dir, device, compile_model=True):
    """Validate existing progress before creating or changing any run files."""
    from agent import Agent, read_last_game_number

    snapshot = run_dir / 'config.json'
    latest = run_dir / 'latest.pth'
    if run_dir.exists() and not run_dir.is_dir():
        raise ValueError(f'Run path is not a directory: {run_dir}')
    if snapshot.exists():
        try:
            saved_config = resolve_preset(json.loads(snapshot.read_text()))
        except ValueError as err:
            raise ValueError(f'Run configuration is invalid: {err}') from err
        conflicts = shape_conflicts(saved_config, config)
        if conflicts:
            raise ValueError(
                f'Run checkpoint cannot use this preset\'s changed {", ".join(conflicts)}; '
                'use a new --run name'
            )
        changes = tunable_changes(saved_config, config)
        if not latest.exists() and (run_dir / 'best.pth').exists():
            raise ValueError('Run is missing latest.pth; restore it before resuming')
        if not latest.exists() and read_last_game_number(run_dir / 'training_scores.csv'):
            raise ValueError('Run has score history but is missing latest.pth; restore it before resuming')
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError('Run directory is occupied without config.json; use a new --run name')

    agent = Agent(config=config, checkpoint_path=latest, device=device, strict_checkpoint=True,
                  compile_model=compile_model)
    # Validate CSV parsing before training can overwrite a checkpoint.
    read_last_game_number(run_dir / 'training_scores.csv')
    run_dir.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        if changes:
            print('Resuming with updated settings:\n  ' + '\n  '.join(changes), flush=True)
            write_config(snapshot, config)
    else:
        write_config(snapshot, config)
    return agent


def write_config(path, config):
    """Replace the run's configuration snapshot atomically."""
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def import_legacy(config, run_dir, device):
    """Copy validated legacy artifacts without touching the source files."""
    from agent import Agent, read_last_game_number

    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise ValueError('Import destination is occupied; use a new --run name')
    source = REPO_ROOT / 'model' / 'model.pth'
    scores = REPO_ROOT / 'training_scores.csv'
    if not source.is_file():
        raise ValueError(f'Legacy checkpoint does not exist: {source}')
    Agent(config=config, checkpoint_path=source, device=device, strict_checkpoint=True)
    read_last_game_number(scores)

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.import-', dir=run_dir.parent) as directory:
        staging = Path(directory)
        shutil.copy2(source, staging / 'latest.pth')
        shutil.copy2(source, staging / 'best.pth')
        if scores.exists():
            shutil.copy2(scores, staging / 'training_scores.csv')
        (staging / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
        os.replace(staging, run_dir)
    print(f'Imported legacy checkpoint and available score history into {run_dir}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='list presets from models.json and exit')
    parser.add_argument('--model', help='named preset in models.json')
    parser.add_argument('--run', help='independent run name (default: model preset name)')
    parser.add_argument('--games', type=positive_int, help='complete games this invocation (default: unlimited)')
    parser.add_argument('--render', action='store_true', help='show the game window')
    parser.add_argument('--reset-epsilon', action='store_true', help='reset exploration to the preset starting value')
    parser.add_argument('--start-length', type=start_length_arg,
                        help='start each game coiled into a solid block of N segments, or of a length drawn '
                             'from MIN-MAX per game, e.g. 4-30 (default: straight 3)')
    parser.add_argument('--epsilon-ramp', type=epsilon_ramp_arg, metavar='START-STOP',
                        help='act greedily (epsilon 0) until a game gains START points, then raise epsilon '
                             'linearly to its scheduled value at STOP points, e.g. 5-20')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument('--no-compile', action='store_true',
                        help='train with the eager model instead of torch.compile on CUDA or MPS')
    parser.add_argument('--import-legacy', action='store_true', help='copy legacy checkpoint/history into a run, then exit')
    args = parser.parse_args(argv)

    try:
        presets = load_presets()
        if args.list:
            for name, config in presets.items():
                print(f'{name}: {config["architecture"]}')
            return
        if args.model not in presets:
            raise ValueError('Select a preset with --model; available: ' + ', '.join(presets))
        run_name = validate_name(args.run if args.run is not None else args.model)
        resolve_device(args.device)
        config = presets[args.model]
        run_dir = REPO_ROOT / 'runs' / run_name
        if args.import_legacy:
            if (args.games is not None or args.render or args.reset_epsilon
                    or args.start_length is not None or args.epsilon_ramp is not None
                    or args.no_compile):
                raise ValueError('--import-legacy only imports; run training in a separate invocation')
            import_legacy(config, run_dir, args.device)
            return

        from agent import train
        from game import validate_start_length

        validate_start_length(args.start_length)
        agent = prepare_run(config, run_dir, args.device, compile_model=not args.no_compile)
        print(f'Model: {args.model} | Run: {run_name} | Output: {run_dir}', flush=True)
        train(max_games=args.games, checkpoint_path=run_dir / 'latest.pth',
              best_checkpoint_path=run_dir / 'best.pth',
              csv_path=run_dir / 'training_scores.csv', agent=agent,
              render=args.render, reset_epsilon=args.reset_epsilon,
              start_length=args.start_length, epsilon_ramp=args.epsilon_ramp)
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError, KeyError, TypeError) as err:
        parser.error(str(err))


if __name__ == '__main__':
    main()
