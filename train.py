"""Train one JSON model preset in its own persistent run directory."""

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile

from configuration import REPO_ROOT, load_presets, positive_int, resolve_preset, validate_name
from model import resolve_device


def prepare_run(config, run_dir, device):
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
        if saved_config != config:
            raise ValueError('Run configuration differs from this preset; use a new --run name')
        if not latest.exists() and (run_dir / 'best.pth').exists():
            raise ValueError('Run is missing latest.pth; restore it before resuming')
        if not latest.exists() and read_last_game_number(run_dir / 'training_scores.csv'):
            raise ValueError('Run has score history but is missing latest.pth; restore it before resuming')
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError('Run directory is occupied without config.json; use a new --run name')

    agent = Agent(config=config, checkpoint_path=latest, device=device, strict_checkpoint=True)
    # Validate CSV parsing before training can overwrite a checkpoint.
    read_last_game_number(run_dir / 'training_scores.csv')
    run_dir.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        with snapshot.open('x') as config_file:
            json.dump(config, config_file, indent=2, allow_nan=False)
            config_file.write('\n')
    return agent


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
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
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
            if args.games is not None or args.render or args.reset_epsilon:
                raise ValueError('--import-legacy only imports; run training in a separate invocation')
            import_legacy(config, run_dir, args.device)
            return

        from agent import train

        agent = prepare_run(config, run_dir, args.device)
        print(f'Model: {args.model} | Run: {run_name} | Output: {run_dir}', flush=True)
        train(max_games=args.games, checkpoint_path=run_dir / 'latest.pth',
              best_checkpoint_path=run_dir / 'best.pth',
              csv_path=run_dir / 'training_scores.csv', agent=agent,
              render=args.render, reset_epsilon=args.reset_epsilon)
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError, KeyError, TypeError) as err:
        parser.error(str(err))


if __name__ == '__main__':
    main()
