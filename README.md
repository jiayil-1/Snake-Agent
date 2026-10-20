# Snake-Agent

Snake-Agent trains a Deep Q-Network (DQN) to play Snake. Training runs against the local game environment without opening a game window. The agent can resume from a saved checkpoint and records scores for later plotting.

## Setup

Run these commands from the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements use `pygame-ce`, which provides the `pygame` module imported by the game. It has a prebuilt macOS universal2 wheel for CPython 3.14, avoiding the SDL-header source build shown by the original `pygame` package. If this virtual environment already has the original `pygame` distribution installed, remove it before installing the requirements:

```bash
python -m pip uninstall pygame
python -m pip install -r requirements.txt
```

## Train named models

Use `train.py` for independent models defined in the repository's `models.json`:

```bash
python train.py --list
python train.py --model residual_cnn --games 1000
python train.py --model mlp --games 1000
python train.py --model residual_cnn --run cnn-experiment --games 1000
```

The included `residual_cnn` preset uses the existing residual CNN. The `mlp` preset flattens the same board input, combines it with metadata, and uses two 128-unit hidden layers. Both use the same DQN trainer and three actions.

`--games` counts completed games in this invocation; omit it to train until stopped. Add `--render` to display the game, `--device cpu` to force CPU, or `--device cuda` / `--device mps` to select an available accelerator. The default device is `auto`: CUDA, then MPS, then CPU. `--reset-epsilon` resets exploration to the preset's `epsilon_start` while preserving learned weights and completed-game counters.

Each run writes to `runs/<run-name>/` relative to the repository, even when the command is launched from another directory. The run name defaults to the preset name. Choose another `--run` name to start a separate model with that preset. Use distinct run names when training multiple processes.

| Run file | Purpose |
| --- | --- |
| `latest.pth` | Weights, optimizer, scheduler, counters, exploration state, records, and configuration for resuming. Saved after every completed episode and on graceful termination. |
| `best.pth` | Checkpoint from the completed episode with the highest cumulative reward. |
| `training_scores.csv` | This run's cumulative game numbers, scores, and ending reasons. |
| `config.json` | Fully resolved configuration, including defaults, used by this run. |

Repeat the same command to resume automatically from `latest.pth`. `Ctrl+C` or a termination signal saves latest progress, including updates made during an unfinished episode, without logging that episode as completed. Resume starts a new episode with an empty replay buffer. Abrupt termination can only recover the last saved checkpoint. Generated run folders are ignored by Git.

### Configure another model

Add another top-level entry to `models.json`. For example, add this preset alongside the existing entries:

```json
"mlp-small": {
  "architecture": "mlp",
  "architecture_options": {"hidden_sizes": [64, 64]},
  "training": {
    "lr": 0.0005,
    "replay_batch_size": 32,
    "epsilon_decay": 0.995
  }
}
```

Then run `python train.py --model mlp-small --games 1000`. JSON must use double quotes and cannot contain comments or trailing commas. Names may contain letters, digits, underscores, and hyphens, starting with a letter or digit. Unknown settings, duplicate keys, unsupported architectures, and invalid numeric values are rejected before outputs are created.

The `residual_cnn` entry shows all supported training settings and their current defaults. Missing training settings use those defaults; `training: {}` selects them all. `architecture_options` defaults to `{}` for the CNN and `{"hidden_sizes": [128, 128]}` for the MLP. Hidden sizes must be a nonempty array of positive integers. Replay capacity must accommodate both batch size and warmup. Exploration values and gamma must be in `[0, 1]`, decay factors in `(0, 1]`, and `epsilon_min` cannot exceed `epsilon_start`. Other settings must be positive; `min_lr` may also be `null` to disable the learning-rate floor and cannot exceed the initial learning rate.

An existing run's resolved configuration must match the selected preset and its checkpoint. If you edit a preset's settings, select a new `--run` name. To keep both experiments easy to select, create a new preset name instead. A new run starts with fresh weights.

To add a different network architecture, implement a `torch.nn.Module` in `model.py` with constructor arguments `board_height`, `board_width`, `metadata_size`, and `output_size`, plus any architecture options. Its `forward(state)` receives `board_state` with shape `(batch, 3, height, width)` and `metadata` with shape `(batch, 12)`, and returns `(batch, 3)` Q-values. Support unbatched state inputs as the existing networks do. Register the class in `ARCHITECTURES`, add its JSON option defaults to `ARCHITECTURE_DEFAULTS`, and validate any new option types/ranges in `configuration.resolve_preset`. JSON selects registered Python classes; it never executes code.

### Import the existing model

```bash
python train.py --model residual_cnn --run legacy --import-legacy
python train.py --model residual_cnn --run legacy --games 1000
```

Import validates `model/model.pth`, copies it into both checkpoint slots, copies `training_scores.csv` if present, and records the resolved configuration. It exits without training and rejects an occupied destination. The original checkpoint and history remain untouched. For checkpoints saved before configurations were recorded, select the preset matching the architecture and original training settings. If the old CSV extends beyond its best-reward checkpoint, learned state resumes from that checkpoint while CSV numbering continues from the final logged game.

Plot a named run with the existing plotter:

```bash
python plot_training_scores.py --csv runs/mlp/training_scores.csv --out runs/mlp/training_scores_plot.png
```

## Legacy training

The original entrypoint remains available with its existing paths and best-reward-only save policy:

```bash
python agent.py
```

Training continues until you stop it with `Ctrl+C` or send a termination signal. A checkpoint is saved only when a completed game achieves a higher cumulative reward than the best reward stored in the current checkpoint. If `model/model.pth` exists, training loads it and resumes from its saved model and metadata; otherwise, it starts with a fresh model. Older checkpoints do not include a cumulative reward, so the first completed game after this change establishes the baseline. The agent uses CUDA when available, then Apple GPU acceleration through MPS when available, and CPU otherwise. Stopping or reaching a game limit does not force a checkpoint save.

Each move's state, action, immediate reward, next state, and terminal flag are stored in replay memory. Once the buffer has 1,000 transitions, the agent samples a 64-transition minibatch every four environment moves. A transition's reward is used only for its own TD target; the episode's cumulative reward is used for checkpoint selection. After warmup, the learning-rate schedule decays every 500 minibatch updates, about 2,000 environment moves at the default cadence. Existing checkpoints keep their current learning rate and converted scheduler phase. Exploration still decays by completed episode.

Use command-line options to limit a run, choose the checkpoint and score CSV paths, reset exploration epsilon, or show the game window:

```bash
python agent.py --games 1000 --checkpoint model/experiment.pth --csv output/scores.csv --render
```

`--games` counts complete games in this run and exits after that many games. A checkpoint is written only if one of those games achieves a new highest cumulative reward. If omitted, training runs until stopped. The `--checkpoint`, `--csv`, and `--render` options are also optional; their defaults preserve the behavior above.

Use `--reset-epsilon` to start the run with epsilon at `1.0`; it then decays by completed game as usual. This preserves the loaded model and cumulative game number.

Run from the repository directory because the checkpoint and score paths are relative to the current working directory. Training appends each game's number, score, and ending reason to `training_scores.csv`, and prints the ending reason in the command-line game summary. The reason is `timeout`, `self-collision`, or `wall-collision`. Existing rows keep their game and score values and have a blank ending reason because their causes were not recorded. On startup, the agent reads the final logged game number and continues numbering from there. The plotter calculates the rolling mean from the scores when it generates the chart. Move or rename the CSV before a run if you want a separate log that starts at game 1.

## Plot training scores

After training, generate or refresh the PNG chart with:

```bash
python plot_training_scores.py
```

By default, this reads `training_scores.csv` and writes `training_scores_plot.png`. Specify different files with `--csv` and `--out`:

```bash
python plot_training_scores.py --csv training_scores.csv --out training_scores_plot.png
```

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Shared command-line entrypoint for named model training and legacy import. |
| `models.json` | Editable named architecture and training presets. |
| `configuration.py` | Preset defaults, JSON loading, and validation. |
| `agent.py` | Shared training loop, agent state/action logic, replay memory, checkpoints, and legacy entrypoint. |
| `game.py` | Snake game environment used during training. |
| `model.py` | Registered CNN and MLP networks, device selection, and DQN trainer. |
| `helper.py` | Training score plotting and CSV logging. |
| `plot_training_scores.py` | Command-line tool that plots scores from the CSV file. |
| `requirements.txt` | Python packages needed to train and plot. |
| `model/model.pth` | Saved checkpoint loaded at startup and updated during training. |
| `training_scores.csv` | Per-game number, score, and ending reason data. |
| `training_scores_plot.png` | Generated chart of the training scores. |
| `arial.ttf` | Font used by the game's display when rendering is enabled. |

## Tests

Run unit tests and headless CLI integration tests:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy MPLBACKEND=Agg python -m unittest -v
```

The tests use temporary run directories and do not train or import into your repository's saved model or score history.
