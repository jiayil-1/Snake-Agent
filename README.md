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
python train.py --model raw_skip_cnn --games 1000
python train.py --model mlp --games 1000
python train.py --model residual_cnn --run cnn-experiment --games 1000
```

The included `residual_cnn` preset uses the existing residual CNN. The `raw_skip_cnn` preset is a lighter CNN whose convolutions run at full board resolution (3→6, residual 6→6 blocks, then 6→3); the raw board is added back as an identity shortcut before pooling. It omits the food distance metadata feature, so the network must judge distance from the board. The `mlp` preset flattens the same board input, combines it with metadata, and uses two 128-unit hidden layers. All use the same DQN trainer and three actions.

`--games` counts completed games in this invocation; omit it to train until stopped. Add `--render` to display the first parallel game, `--device cpu` to force CPU, or `--device cuda` / `--device mps` to select an available accelerator. The default device is `auto`: CUDA, then MPS, then CPU. `--reset-epsilon` resets exploration to the preset's `epsilon_start` while preserving learned weights and completed-game counters. `--start-length N` starts every game with an `N`-segment snake coiled into a solid, hole-free block of random shape, orientation, and position, with the head on the block's edge always facing a free cell (up to half the board, 384 segments on the default 32×24 grid). `--start-length MIN-MAX`, such as `--start-length 4-30`, draws each game's length uniformly from that inclusive range instead. A game's score starts at its length minus 3, as if it had grown from the classic start, so scores stay comparable across start lengths; starts shorter than three begin below zero. Without it, games use the classic straight three-segment start in the centre. It applies only to that invocation and is not saved in the run configuration. `--epsilon-ramp START-STOP`, such as `--epsilon-ramp 5-20`, makes each game act greedily (epsilon 0) from spawn until it has gained `START` points, then raises its epsilon linearly with each further point until it reaches the preset's scheduled epsilon (from `epsilon_start`, `epsilon_decay`, and `epsilon_min`) at `STOP` points gained, staying there for the rest of the game. Points are counted from the game's starting score, so `--start-length` games also spawn at epsilon 0. `START` may be 0 and must be less than `STOP`. Like `--start-length`, it applies only to that invocation. On CUDA and MPS, each replay update runs through `torch.compile`, which takes a few seconds to warm up at the first update and then trains about 1.3x faster; CPU training stays eager because compiling measured no faster there. `--no-compile` trains with the eager model instead.

Training plays `parallel_games` games (128 by default) in lockstep. Each step chooses every game's move with one batched forward pass, stores each game's transition in the shared replay memory, and trains at the usual `train_every` cadence counted across all games. A game that ends is logged and its slot starts a new game immediately, so games are numbered in the order they finish. With `--games`, no more than that many games start.

Each run writes to `runs/<run-name>/` relative to the repository, even when the command is launched from another directory. The run name defaults to the preset name. Choose another `--run` name to start a separate model with that preset. Use distinct run names when training multiple processes.

| Run file | Purpose |
| --- | --- |
| `latest.pth` | Weights, optimizer, scheduler, counters, exploration state, records, and configuration for resuming. Saved after each step of the parallel games in which an episode completed, and on graceful termination. |
| `best.pth` | Checkpoint from the completed episode with the highest cumulative reward. |
| `training_scores.csv` | This run's cumulative game numbers, scores, ending reasons, and starting scores. |
| `deaths.jsonl` | Each completed game's final board (snake, food, and direction of every segment), with its game number, score, ending reason, steps, the epsilon of its last move, whether that move was a random exploration, how many of the game's moves were random explorations, and which segments those moves placed, for `death_viewer.py`. |
| `config.json` | Fully resolved configuration, including defaults, used by this run. |

Repeat the same command to resume automatically from `latest.pth`. `Ctrl+C` or a termination signal saves latest progress, including updates made during unfinished episodes, without logging those episodes as completed. Resume starts new episodes with an empty replay buffer. Abrupt termination can only recover the last saved checkpoint. Git ignores generated run files except `.pth` checkpoints.

### Configure another model

Add another top-level entry to `models.json`. For example, this preset changes the body map, reward schedule, CNN size, and learning settings:

```json
"binary-board-experiment": {
  "architecture": "residual_cnn",
  "state_representation": {
    "encoder": "board_metadata",
    "options": {"body_encoding": "binary"}
  },
  "rewards": {
    "step_penalty": -0.02,
    "progress_reward": 0.1,
    "progress_penalty": -0.1,
    "food_reward": 8,
    "collision_penalty": -12,
    "timeout_penalty": -4,
    "win_reward": 100
  },
  "architecture_options": {
    "channels": 48,
    "residual_blocks": 4,
    "board_embedding_size": 160,
    "metadata_embedding_size": 64,
    "fusion_hidden_size": 128
  },
  "training": {
    "lr": 0.0005,
    "replay_batch_size": 32,
    "replay_warmup": 512,
    "train_every": 2,
    "gamma": 0.95,
    "epsilon_decay": 0.995,
    "lr_step_size": 400,
    "lr_decay": 0.9
  }
}
```

Then run `python train.py --model binary-board-experiment --games 1000`. To use the compact metadata-only state, select the MLP and omit the board input:

```json
"metadata-only": {
  "architecture": "mlp",
  "state_representation": {"encoder": "metadata_only"},
  "architecture_options": {"hidden_sizes": [64, 32]},
  "training": {"lr": 0.0005, "gamma": 0.95}
}
```

JSON must use double quotes and cannot contain comments or trailing commas. Names may contain letters, digits, underscores, and hyphens, starting with a letter or digit. Unknown settings, duplicate keys, unsupported architectures, and invalid numeric values are rejected before outputs are created. The CNN requires a board-based representation. The board encoder supports `gradient` body values (the default, head to tail from 1 to 0.5) or `binary` occupancy values (1 for every body segment). `metadata_only` returns only the existing 12 direction, danger, food direction, and food distance features. Both encoders accept `"food_distance": false` to drop the normalized head-to-food distance, leaving 11 metadata features.

The `residual_cnn` entry shows all supported training settings and their current defaults. Missing training settings use those defaults; `training: {}` selects them all. CNN architecture options control convolution channels, residual block count, board and metadata embedding sizes, and the fusion layer width. MLP `hidden_sizes` sets its nonempty sequence of positive layer widths. `parallel_games` sets how many games play at once; it changes how experience is collected, so it is part of a run's configuration. `death_replay_fraction` draws that share of each replay minibatch from the last `death_replay_window` moves of games that ended in a wall or self-collision, and the rest from all other stored moves; the default 0 samples the buffer uniformly. Timeouts are not prioritized, and a game's flagged moves stop being favoured once replay overwrites them. Replay capacity must accommodate both batch size and warmup. Exploration values, gamma, and `death_replay_fraction` must be in `[0, 1]`, decay factors and `target_update_tau` in `(0, 1]`, and `epsilon_min` cannot exceed `epsilon_start`. Other training settings must be positive; `min_lr` may also be `null` to disable the learning-rate floor and cannot exceed the initial learning rate. Reward values accept any finite signed numbers.

An existing run's network and state settings must match the selected preset and its checkpoint. Older run snapshots that lack `state_representation` or `rewards` are read as using the defaults; the saved files are left as they are until a training checkpoint is naturally saved. Rewards and training settings (such as `lr`, `gamma`, `replay_batch_size`, or `parallel_games`) may be edited between invocations: the run resumes from its checkpoint with the new values, prints each changed setting, and updates `config.json`. A changed `lr` or `lr_decay` restarts from the new base rate at the decay step already reached. Changing `architecture`, `architecture_options`, or `state_representation` makes the checkpoint unusable and is rejected; select a new `--run` name or create a new preset for that. A new run starts with fresh weights. Raw legacy checkpoints do not record their state settings, so they can only be imported for the default state representation.

To add another state representation, implement a deterministic encoder in `state.py` that returns `metadata` and optionally `board_state`, then register its option defaults, allowed options, channel count, and a function returning its metadata feature names for given options in `STATE_ENCODERS`. Update `resolve_state_representation` to validate its option values and add tests for exact values and shapes. `get_observation_spec` builds the compatible model input contract from that registry, so metadata-only encoders must report zero board channels and return no board tensor.

To add a different network architecture, implement a `torch.nn.Module` in `model.py` with constructor arguments `board_height`, `board_width`, `metadata_size`, and `output_size`, plus `state_spec` and any architecture options. `state_spec` describes whether the selected encoder supplies a board and its dimensions. A board-based `forward(state)` receives `board_state` with shape `(batch, 3, height, width)` and `metadata` with shape `(batch, metadata_size)`; metadata-only models receive just `metadata`. Return `(batch, 3)` Q-values and support unbatched inputs as the existing networks do. Register the class in `ARCHITECTURES`, its capabilities in `ARCHITECTURE_CAPABILITIES`, add JSON option defaults to `ARCHITECTURE_DEFAULTS`, and validate new option types/ranges in `configuration.resolve_preset`. JSON selects registered Python classes; it never executes code.

### Import the existing model

```bash
python train.py --model residual_cnn --run legacy --import-legacy
python train.py --model residual_cnn --run legacy --games 1000
```

Import validates `model/model.pth`, copies it into both checkpoint slots, copies `training_scores.csv` if present, and records the resolved configuration. It exits without training and rejects an occupied destination. The original checkpoint and history remain untouched. For checkpoints saved before configurations were recorded, select the preset matching the architecture and original training settings. If the old CSV extends beyond its best-reward checkpoint, learned state resumes from that checkpoint while CSV numbering continues from the final logged game.

Plot a named run with the existing plotter:

```bash
python plot_training_scores.py --csv runs/<run-name>/training_scores.csv --out runs/<run-name>/training_scores_plot.png
```

Render a checkpoint's learned weights to a PNG:

```bash
python visualize_weights.py --model raw_skip_cnn                      # runs/raw_skip_cnn/best_weights.png
python visualize_weights.py --model raw_skip_cnn --checkpoint latest  # runs/raw_skip_cnn/latest_weights.png
python visualize_weights.py --checkpoint model/model.pth --out weights.png
```

`--model`, `--run`, and `--checkpoint` select a checkpoint as in `inference.py`, and the output defaults to `<checkpoint name>_weights.png` beside it. Like inference, it reads a private snapshot, so it is safe to run during training. The image shows every 3×3 convolution kernel (rows are output channels, columns input channels), each dense weight matrix, and per-layer weight distributions. Two panels estimate what the network reads: board input importance maps each board cell, after the CNNs' two max-pools, to the L2 norm of its column in the first dense layer, per head, food, and body channel (or combined over the residual CNN's 32 feature channels), and metadata feature importance does the same for each metadata input. Large weights only mean a strong connection; they do not show how often an input is active.

Click through a run's deaths in a window drawn like the game:

```bash
python death_viewer.py --model raw_skip_cnn
python death_viewer.py --run cnn-experiment
python death_viewer.py --deaths path/to/deaths.jsonl
```

Each game's final board is shown as the game drew its last frame, inside a one-cell wall ring so a head that left the board stays visible; a collision outlines the head in red. A translucent line through the segment centres traces the path from tail to head; press `P` to hide or show it. The header shows the game number, ending reason, score and starting score, length, steps, and epsilon. With `--epsilon-ramp`, epsilon reads as the scheduled value times the ramp multiplier, with the effective value the last move used. Segments placed by a random exploration are drawn green, and a green swatch beside the ending reason counts the game's random moves. Games logged before this tracking only call out a random final move. The list beside the board sorts by game, score, effective epsilon, final move (`random` or `greedy`; `-` for games logged before moves were tracked), or ending reason: click a column (again to reverse it) or press `G`, `S`, `E`, or `M`. Step with the arrow keys, the mouse wheel, the Prev and Next buttons, `PgUp`/`PgDn`, or `Home`/`End`, or click a row. The viewer follows the file, so games a running training process finishes join the list within a second. Only games trained after death recording was added have a recorded board.

## Evaluate a model

`inference.py` plays games with the network's best move every step: no exploration, replay, training, or file writes. It copies the checkpoint to a temporary snapshot before loading, so it is safe to run while the same run is training.

```bash
python inference.py --model raw_skip_cnn                       # 100 headless games from best.pth
python inference.py --model raw_skip_cnn --checkpoint latest --games 500 --seed 1
python inference.py --run cnn-experiment --render --fps 15     # watch games one at a time
python inference.py --checkpoint model/model.pth               # any checkpoint file
```

`--checkpoint` takes `best` (default), `latest`, or a file path. The network is rebuilt from the config saved in the checkpoint, falling back to the run's `config.json` or the `--model` preset for older checkpoints. Headless games run in lockstep batches of the run's `parallel_games`. `--start-length` and `--device` work as in training, and `--seed` makes food and start placement repeatable for comparing checkpoints. Each game's score, steps, and end reason are printed, followed by the mean, median, range, and end-reason counts.

## Legacy training

The original entrypoint remains available with its existing paths and best-reward-only save policy:

```bash
python agent.py
```

Training continues until you stop it with `Ctrl+C` or send a termination signal. A checkpoint is saved when a completed game achieves a higher cumulative reward than the best reward stored in the current checkpoint, and the latest progress is also saved whenever training exits gracefully. This includes updates made during unfinished episodes. If `model/model.pth` exists, training loads it and resumes from its saved model and metadata; otherwise, it starts with a fresh model. Older checkpoints do not include a cumulative reward, so the first completed game after this change establishes the baseline. The agent uses CUDA when available, then Apple GPU acceleration through MPS when available, and CPU otherwise. Abrupt termination can only recover the last saved checkpoint.

Training plays 128 games in parallel, as described above. Each move's state, action, immediate reward, next state, and terminal flag are stored in replay memory. Once the buffer has 1,000 transitions, the agent samples a 64-transition minibatch every four environment moves, counted across all games. TD targets follow Double DQN: the online network picks the best next move and a separate target network values it, which keeps each update from chasing its own change and curbs overestimated Q-values. After every minibatch update the target moves `target_update_tau` (default 0.005) of the way toward the online weights; checkpoints saved before the target network existed start it as a copy of the model. A transition's reward is used only for its own TD target; the episode's cumulative reward is used for checkpoint selection. After warmup, the learning-rate schedule decays every 500 minibatch updates, about 2,000 environment moves at the default cadence. Existing checkpoints keep their current learning rate and converted scheduler phase. Exploration still decays by completed episode. An exploratory move is drawn only from the moves that do not hit a wall or the body on that step, counting a tail that moves away as free; when every move collides, the network chooses instead. Random moves therefore never cause a collision, though a game can still time out on one.

Use command-line options to limit a run, choose the checkpoint and score CSV paths, reset exploration epsilon, or show the game window:

```bash
python agent.py --games 1000 --checkpoint model/experiment.pth --csv output/scores.csv --render
```

`--games` counts complete games in this run and exits after that many games, saving the latest checkpoint before exit. If omitted, training runs until stopped and saves on graceful exit. The `--checkpoint`, `--csv`, `--render`, `--start-length`, `--epsilon-ramp`, and `--no-compile` options are also optional; their defaults preserve the behavior above.

Use `--reset-epsilon` to start the run with epsilon at `1.0`; it then decays by completed game as usual. This preserves the loaded model and cumulative game number.

Run from the repository directory because the checkpoint and score paths are relative to the current working directory. Training appends each game's number, final score, ending reason, and starting score to `training_scores.csv`, and prints the ending reason in the command-line game summary. The reason is `timeout`, `self-collision`, `wall-collision`, or `win` when the snake fills the board; a win adds `win_reward` on top of the final food reward. Existing rows keep their game and score values and have a blank ending reason or starting score when those were not recorded. On startup, the agent reads the final logged game number and continues numbering from there. The plotter calculates the rolling mean from the scores when it generates the chart. Move or rename the CSV before a run if you want a separate log that starts at game 1.

## Plot training scores

After training, generate or refresh the PNG chart with:

```bash
python plot_training_scores.py
```

When the CSV records starting scores, a second panel plots each game's score gained (final score minus starting score) with its 100-game rolling mean, so long `--start-length` starts are not credited for length the agent never earned. Games without a recorded starting score are left out of that panel.

By default, this reads `training_scores.csv` and writes `training_scores_plot.png`. Specify different files with `--csv` and `--out`:

```bash
python plot_training_scores.py --csv training_scores.csv --out training_scores_plot.png
```

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Shared command-line entrypoint for named model training and legacy import. |
| `inference.py` | Greedy evaluation of a saved checkpoint, safe to run during training. |
| `models.json` | Editable named architecture and training presets. |
| `configuration.py` | Preset defaults, JSON loading, and validation. |
| `state.py` | Registered observation encoders and their input-shape descriptions. |
| `reward_defaults.py` | Default game rewards shared by config validation and the environment. |
| `agent.py` | Shared training loop, agent state/action logic, replay memory, checkpoints, and legacy entrypoint. |
| `game.py` | Snake game environment used during training. |
| `model.py` | Registered CNN and MLP networks, device selection, and DQN trainer. |
| `helper.py` | Training score plotting and CSV logging. |
| `plot_training_scores.py` | Command-line tool that plots scores from the CSV file. |
| `requirements.txt` | Python packages needed to train and plot. |
| `model/model.pth` | Saved checkpoint loaded at startup and updated during training. |
| `training_scores.csv` | Per-game number, score, ending reason, and starting score data. |
| `training_scores_plot.png` | Generated chart of the training scores. |
| `arial.ttf` | Font used by the game's display when rendering is enabled. |

## Tests

Run unit tests and headless CLI integration tests:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy MPLBACKEND=Agg python -m unittest -v
```

The tests use temporary run directories and do not train or import into your repository's saved model or score history.
