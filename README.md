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

## Train the agent

```bash
python agent.py
```

Training continues until you stop it with `Ctrl+C` or send a termination signal. A checkpoint is saved only when a completed game achieves a higher cumulative reward than the best reward stored in the current checkpoint. If `model/model.pth` exists, training loads it and resumes from its saved model and metadata; otherwise, it starts with a fresh model. Older checkpoints do not include a cumulative reward, so the first completed game after this change establishes the baseline. The agent uses CUDA when available, then Apple GPU acceleration through MPS when available, and CPU otherwise. Stopping or reaching a game limit does not force a checkpoint save.

Each move's state, action, immediate reward, next state, and terminal flag are stored in replay memory. Once the buffer has 1,000 transitions, the agent samples a 64-transition minibatch every four environment moves. A transition's reward is used only for its own TD target; the episode's cumulative reward is used for checkpoint selection. After warmup, the learning-rate schedule decays every 500 minibatch updates, about 2,000 environment moves at the default cadence. Existing checkpoints keep their current learning rate and converted scheduler phase. Exploration still decays by completed episode.

Use command-line options to limit a run, choose the checkpoint and score CSV paths, or show the game window:

```bash
python agent.py --games 1000 --checkpoint model/experiment.pth --csv output/scores.csv --render
```

`--games` counts complete games in this run and exits after that many games. A checkpoint is written only if one of those games achieves a new highest cumulative reward. If omitted, training runs until stopped. The `--checkpoint`, `--csv`, and `--render` options are also optional; their defaults preserve the behavior above.

Run from the repository directory because the checkpoint and score paths are relative to the current working directory. Training appends each game's number and score to `training_scores.csv`. On startup, the agent reads the final logged game number and continues numbering from there. The plotter calculates the rolling mean from the scores when it generates the chart. Move or rename the CSV before a run if you want a separate log that starts at game 1.

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
| `agent.py` | Training entry point, agent state/action logic, replay memory, and checkpoint saving/loading. |
| `game.py` | Snake game environment used during training. |
| `model.py` | DQN network and trainer. |
| `helper.py` | Training score plotting and CSV logging. |
| `plot_training_scores.py` | Command-line tool that plots scores from the CSV file. |
| `requirements.txt` | Python packages needed to train and plot. |
| `model/model.pth` | Saved checkpoint loaded at startup and updated during training. |
| `training_scores.csv` | Per-game number and score data. |
| `training_scores_plot.png` | Generated chart of the training scores. |
| `arial.ttf` | Font used by the game's display when rendering is enabled. |
