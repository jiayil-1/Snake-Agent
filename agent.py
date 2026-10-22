import argparse
import torch
import random
import numpy as np
import os
import signal
import csv
from copy import deepcopy
from datetime import datetime
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE, validate_start_length
from model import ARCHITECTURES, QTrainer, resolve_device
from configuration import (
    DEFAULT_TRAINING, epsilon_ramp_arg, positive_int, resolve_preset, shape_conflicts,
    start_length_arg, validate_epsilon_ramp,
)
from helper import append_score, ensure_score_columns, plot
from state import StateLayout, encode_resolved_state, get_observation_spec

# ── Hyperparameters ──────────────────────────────────────────────────────
MAX_MEMORY = DEFAULT_TRAINING['max_memory']
BATCH_SIZE = DEFAULT_TRAINING['replay_batch_size']
REPLAY_WARMUP = DEFAULT_TRAINING['replay_warmup']
TRAIN_EVERY = DEFAULT_TRAINING['train_every']
LR = DEFAULT_TRAINING['lr']
GAMMA = DEFAULT_TRAINING['gamma']
EPSILON_START = DEFAULT_TRAINING['epsilon_start']
EPSILON_MIN = DEFAULT_TRAINING['epsilon_min']
EPSILON_DECAY = DEFAULT_TRAINING['epsilon_decay']
HIDE_PLOT = True
TRAINING_SCORES_CSV = 'training_scores.csv'
DEFAULT_CHECKPOINT_PATH = 'model/model.pth'


class ReplayMemory:
    """Fixed-capacity ring buffer that packs each transition into one float32 row.

    Row layout: state, next state, action index, reward, done. A sampled
    minibatch therefore reaches the training device in a single copy.
    """

    def __init__(self, capacity, layout):
        self.capacity = capacity
        self.layout = layout
        self._size = 0
        self._next_index = 0
        # Zero-filled allocations are committed lazily, so memory grows as replay fills.
        self._rows = np.zeros((capacity, 2 * layout.size + 3), dtype=np.float32)

    def __len__(self):
        return self._size

    def append(self, state, action, reward, next_state, done):
        """Store one transition, replacing the oldest when full. Actions are indices."""
        size = self.layout.size
        row = self._rows[self._next_index]
        self.layout.pack(state, out=row[:size])
        self.layout.pack(next_state, out=row[size:2 * size])
        row[2 * size:] = action, reward, done
        self._next_index = (self._next_index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size, device):
        """Return distinct uniformly sampled transitions as tensors on device."""
        indices = random.sample(range(self._size), batch_size)
        rows = torch.from_numpy(self._rows[indices]).to(device)
        size = self.layout.size
        return (
            self.layout.unpack(rows[:, :size]),
            rows[:, 2 * size].long(),
            rows[:, 2 * size + 1],
            self.layout.unpack(rows[:, size:2 * size]),
            rows[:, 2 * size + 2],
        )


class Agent:

    def __init__(self, board_width=640, board_height=480,
                 checkpoint_path=DEFAULT_CHECKPOINT_PATH,
                 replay_batch_size=None,
                 replay_warmup=None,
                 train_every=None, config=None, device='auto',
                 strict_checkpoint=False, compile_model=True):
        self.config = resolve_preset(config if config is not None else {'architecture': 'residual_cnn'})
        overrides = {'replay_batch_size': replay_batch_size,
                     'replay_warmup': replay_warmup, 'train_every': train_every}
        self.config['training'].update({key: value for key, value in overrides.items()
                                        if value is not None})
        self.config = resolve_preset(self.config)
        settings = self.config['training']
        self.strict_checkpoint = strict_checkpoint
        self.n_games = 0
        self.epsilon_start_game = 0
        self.record = 0
        self.best_reward = float('-inf')
        self.epsilon = settings['epsilon_start']
        self.gamma = settings['gamma']
        self.replay_batch_size = settings['replay_batch_size']
        self.replay_warmup = settings['replay_warmup']
        self.train_every = settings['train_every']
        self.parallel_games = settings['parallel_games']
        self.environment_steps = 0
        self.board_width = board_width
        self.board_height = board_height
        self.grid_width = self.board_width // BLOCK_SIZE
        self.grid_height = self.board_height // BLOCK_SIZE

        self.state_spec = get_observation_spec(
            self.config['state_representation'], self.grid_height, self.grid_width,
        )
        self.board_channels = self.state_spec['board_channels']
        self.metadata_size = self.state_spec['metadata_size']
        self.state_layout = StateLayout(self.state_spec)

        self.device = resolve_device(device)

        self.model = ARCHITECTURES[self.config['architecture']](
            board_height=self.grid_height,
            board_width=self.grid_width,
            metadata_size=self.metadata_size,
            output_size=3,
            state_spec=self.state_spec,
            **self.config['architecture_options'],
        ).to(self.device)
        self.trainer = QTrainer(
            self.model, gamma=self.gamma, device=self.device.type,
            compile_model=compile_model,
            **{key: settings[key] for key in
               ('lr', 'max_grad_norm', 'lr_step_size', 'lr_decay', 'min_lr')},
        )
        self._load_checkpoint(checkpoint_path)

        # Replay is repopulated in memory after checkpoint resume.
        self.memory = ReplayMemory(settings['max_memory'], self.state_layout)

    def _load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except FileNotFoundError:
            print(f'No checkpoint found at {checkpoint_path}. Starting with a fresh model.')
            return

        wrapped = isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
        if self.strict_checkpoint:
            saved_config = checkpoint.get('config') if wrapped else None
            if saved_config is not None:
                try:
                    saved_config = resolve_preset(saved_config)
                except ValueError as err:
                    raise ValueError(f'Checkpoint configuration is invalid: {err}') from err
                conflicts = shape_conflicts(saved_config, self.config)
                if conflicts:
                    raise ValueError(
                        f'Checkpoint cannot use this preset\'s changed {", ".join(conflicts)}; '
                        'use a new --run name'
                    )
            else:
                default_semantics = resolve_preset({'architecture': self.config['architecture']})
                if self.config['state_representation'] != default_semantics['state_representation']:
                    raise ValueError(
                        'Checkpoint has no experiment config; nondefault state cannot be verified'
                    )
        model_state = checkpoint['model_state_dict'] if wrapped else checkpoint
        # Strict loading can copy matching tensors before reporting a mismatch.
        # Validate on a copy so rejected checkpoints never alter the live model.
        candidate = deepcopy(self.model)
        try:
            candidate.load_state_dict(model_state)
        except RuntimeError as err:
            if self.strict_checkpoint:
                raise ValueError(f'Checkpoint architecture mismatch: {err}') from err
            print(f'Checkpoint architecture mismatch ({err}). Starting with a fresh model.')
            return
        self.model.load_state_dict(candidate.state_dict())

        if wrapped:
            optimizer_state = checkpoint.get('optimizer_state_dict')
            if optimizer_state is not None:
                try:
                    self.trainer.optimizer.load_state_dict(optimizer_state)
                except ValueError as err:
                    if self.strict_checkpoint:
                        raise ValueError(f'Optimizer state mismatch: {err}') from err
                    print(f'Optimizer state mismatch ({err}). Continuing with fresh optimizer state.')

            scheduler_state = checkpoint.get('scheduler_state_dict')
            if scheduler_state is not None and hasattr(self.trainer, 'scheduler'):
                try:
                    scheduler_state = dict(scheduler_state)
                    old_step_size = scheduler_state.get('step_size')
                    new_step_size = self.trainer.scheduler.step_size
                    if old_step_size and old_step_size != new_step_size:
                        old_last_epoch = scheduler_state.get('last_epoch', 0)
                        scheduler_state['last_epoch'] = (
                            old_last_epoch * new_step_size // old_step_size
                        )
                        scheduler_state['_step_count'] = (
                            scheduler_state['last_epoch'] + 1
                        )
                        scheduler_state['step_size'] = new_step_size
                    self.trainer.scheduler.load_state_dict(scheduler_state)
                except ValueError as err:
                    if self.strict_checkpoint:
                        raise ValueError(f'Scheduler state mismatch: {err}') from err
                    print(f'Scheduler state mismatch ({err}). Continuing with fresh scheduler state.')

            self.trainer.apply_settings(self.config['training'])

            training_state = checkpoint.get('training_state', {})
            self.environment_steps = training_state.get('environment_steps', 0)
            self.trainer.optimizer_steps = training_state.get(
                'optimizer_steps', self.trainer.scheduler.last_epoch,
            )

            metadata = checkpoint.get('metadata', {})
            print('Loaded checkpoint metadata:', metadata)
            self.record = metadata.get('record', 0)
            self.best_reward = metadata.get('best_reward', float('-inf'))
            self.n_games = metadata.get('n_games', 0)
            self.epsilon_start_game = metadata.get('epsilon_start_game', 0)
            self._update_epsilon()
            print(f'Loading agent with record {self.record}, n_games {self.n_games}, '
                  f'epsilon {self.epsilon:.4f}')
        else:
            print('Loaded legacy checkpoint (no metadata found).')

    # ── State representation ─────────────────────────────────────────────
    def get_state(self, game):
        return encode_resolved_state(
            self.config['state_representation'], game, self.grid_width, self.grid_height,
        )


    # ── Replay buffer interface ──────────────────────────────────────────
    def remember(self, state, action, reward, next_state, done):
        """Store a transition whose action is one-hot, as returned by get_action."""
        self.memory.append(state, int(np.argmax(action)), reward, next_state, done)
        self.environment_steps += 1

    def train_long_memory(self):
        """Train one replay minibatch when warmup and update cadence are met."""
        if len(self.memory) < max(self.replay_warmup, self.replay_batch_size):
            return False
        if self.environment_steps % self.train_every != 0:
            return False

        self.trainer.train_batch(*self.memory.sample(self.replay_batch_size, self.device))
        return True

    def train_short_memory(self, state, action, reward, next_state, done):
        """Keep the legacy immediate single-transition update API available."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        return self.get_actions([state])[0]

    def get_actions(self, states, epsilon_scales=None):
        """Epsilon-greedy one-hot actions, with one forward pass for all greedy rows.

        ``epsilon_scales`` optionally multiplies epsilon per state, from 0 (greedy) to 1.
        """
        self._update_epsilon()
        if epsilon_scales is None:
            epsilon_scales = [1.0] * len(states)
        moves = [random.randint(0, 2) if random.random() < self.epsilon * scale else None
                 for scale in epsilon_scales]
        greedy = [index for index, move in enumerate(moves) if move is None]
        if greedy:
            flat = np.stack([self.state_layout.pack(states[index]) for index in greedy])
            batch = self.state_layout.unpack(torch.from_numpy(flat).to(self.device))
            with torch.no_grad():
                best = self.model(batch).argmax(dim=1).tolist()
            for index, move in zip(greedy, best):
                moves[index] = move
        final_moves = []
        for move in moves:
            final_move = [0, 0, 0]
            final_move[move] = 1
            final_moves.append(final_move)
        return final_moves

    def _update_epsilon(self):
        games_since_reset = max(0, self.n_games - self.epsilon_start_game)
        settings = self.config['training']
        self.epsilon = max(
            settings['epsilon_min'],
            settings['epsilon_start'] * (settings['epsilon_decay'] ** games_since_reset),
        )


def epsilon_ramp_scale(points_gained, epsilon_ramp):
    """Epsilon multiplier: 0 through ``start`` points gained, rising linearly to 1 at ``stop``."""
    start, stop = epsilon_ramp
    return min(1.0, max(0.0, (points_gained - start) / (stop - start)))


def save_checkpoint_atomic(agent, metadata=None,
                           checkpoint_path=DEFAULT_CHECKPOINT_PATH):
    checkpoint_path = os.fspath(checkpoint_path)
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    temp_path = f"{checkpoint_path}.tmp.{os.getpid()}"

    checkpoint_metadata = {
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'record': agent.record,
        'best_reward': agent.best_reward,
        'n_games': agent.n_games,
        'epsilon_start_game': agent.epsilon_start_game,
    }
    if metadata:
        checkpoint_metadata.update(metadata)

    checkpoint = {
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.trainer.optimizer.state_dict(),
        'scheduler_state_dict': agent.trainer.scheduler.state_dict(),
        'training_state': {
            'environment_steps': agent.environment_steps,
            'optimizer_steps': agent.trainer.optimizer_steps,
        },
        'metadata': checkpoint_metadata,
        'config': agent.config,
    }

    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def read_last_game_number(csv_path=TRAINING_SCORES_CSV):
    try:
        with open(csv_path, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            last_game_number = 0
            for row in reader:
                game_number = row.get('game')
                if game_number:
                    last_game_number = int(game_number)
            return last_game_number
    except FileNotFoundError:
        return 0


def read_recent_scores(csv_path, count):
    """Return the scores of the last ``count`` logged games, oldest first."""
    try:
        with open(csv_path, newline='') as csv_file:
            return [float(row['score']) for row in deque(csv.DictReader(csv_file), maxlen=count)]
    except FileNotFoundError:
        return []


def train(max_games=None, checkpoint_path=DEFAULT_CHECKPOINT_PATH,
          csv_path=TRAINING_SCORES_CSV, render=False, reset_epsilon=False,
          config=None, device='auto', agent=None, best_checkpoint_path=None,
          start_length=None, compile_model=True, epsilon_ramp=None):
    """Train one agent on lockstep parallel games.

    Finished slots start a new game immediately until ``max_games`` have
    started. A separate best path enables latest-plus-best saving.
    ``start_length`` starts every game coiled into a random solid block of that many segments,
    or of a length drawn per game from an inclusive ``(min, max)`` range.
    ``epsilon_ramp`` is a ``(start, stop)`` pair of points gained since spawn: each game
    explores with epsilon 0 until ``start``, rising linearly to the scheduled epsilon at ``stop``.
    """
    validate_epsilon_ramp(epsilon_ramp)
    plot_scores = []
    plot_mean_scores = []
    # Seed the rolling mean from logged history so it continues across resumes.
    recent_scores = deque(read_recent_scores(csv_path, 100), maxlen=100)
    resolved_config = resolve_preset(
        config if config is not None else agent.config if agent is not None
        else {'architecture': 'residual_cnn'},
    )
    if agent is not None and config is not None and resolved_config != agent.config:
        raise ValueError('Training config differs from the supplied agent config')
    game_number_offset = read_last_game_number(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    slot_count = resolved_config['training']['parallel_games']
    if max_games is not None:
        slot_count = min(slot_count, max_games)
    # Only the first game opens a window; rendering every slot is unreadable.
    games = [SnakeGameAI(render=render and slot == 0, rewards=resolved_config['rewards'],
                         start_length=start_length)
             for slot in range(slot_count)]
    if agent is None:
        agent = Agent(
            board_width=games[0].w,
            board_height=games[0].h,
            checkpoint_path=checkpoint_path,
            config=resolved_config,
            device=device,
            strict_checkpoint=best_checkpoint_path is not None,
            compile_model=compile_model,
        )
    if reset_epsilon:
        agent.epsilon_start_game = agent.n_games
        agent._update_epsilon()
        print(f'Epsilon reset to {agent.epsilon:.4f} for this run.')
    record = agent.record
    games_this_run = 0
    games_started = slot_count
    episode_rewards = [0.0] * slot_count
    termination_requested = False
    graceful_exit = False

    def request_termination(signum, _frame):
        nonlocal termination_requested
        if not termination_requested:
            termination_requested = True
            print(f'Received signal {signum}. Will stop after the current step of all games.',
                  flush=True)

    signals = [signal.SIGTERM, signal.SIGINT]
    if hasattr(signal, 'SIGUSR1'):
        signals.append(signal.SIGUSR1)
    previous_handlers = {sig: signal.signal(sig, request_termination) for sig in signals}

    try:
        ensure_score_columns(csv_path)
        print(f'Training started with {slot_count} parallel game(s).', flush=True)
        # Each slot carries its state forward; later states come from the previous step.
        states = [agent.get_state(game) for game in games]
        active = list(range(slot_count))
        while active:
            # One batched forward pass chooses the move for every active game.
            epsilon_scales = None if epsilon_ramp is None else [
                epsilon_ramp_scale(games[slot].score - games[slot].start_score, epsilon_ramp)
                for slot in active
            ]
            moves = agent.get_actions([states[slot] for slot in active], epsilon_scales)
            still_active = []
            completed_games = 0
            new_highest_reward = False
            for slot, final_move in zip(active, moves):
                game = games[slot]
                reward, done, score = game.play_step(final_move)
                episode_rewards[slot] += reward
                state_new = agent.get_state(game)

                # Keep each transition and train from replay at the configured cadence.
                agent.remember(states[slot], final_move, reward, state_new, done)
                agent.train_long_memory()
                states[slot] = state_new
                if not done:
                    still_active.append(slot)
                    continue

                # ── Episode finished ─────────────────────────────────────
                episode_reward = episode_rewards[slot]
                end_reason = game.end_reason
                print(f'\n\nFinished with reward {episode_reward:.1f}')
                agent.n_games += 1
                games_this_run += 1
                completed_games += 1

                if score > record:
                    record = score
                agent.record = record

                if episode_reward > agent.best_reward:
                    agent.best_reward = episode_reward
                    new_highest_reward = True
                    print(f'New highest reward {episode_reward:.1f}.')

                print(f'Game {agent.n_games}  Score {score}  '
                      f'End reason: {end_reason}  Record: {record}  '
                      f'Eps: {agent.epsilon:.4f}')

                append_score(csv_path, game_number_offset + games_this_run, score, end_reason,
                             game.start_score)
                if not HIDE_PLOT:
                    plot_scores.append(score)
                    recent_scores.append(score)
                    plot_mean_scores.append(sum(recent_scores) / len(recent_scores))
                    plot(plot_scores, plot_mean_scores)

                if max_games is None or games_started < max_games:
                    game.reset()
                    states[slot] = agent.get_state(game)
                    episode_rewards[slot] = 0.0
                    games_started += 1
                    still_active.append(slot)
            active = still_active

            # Save once per step of all games so checkpoint writes stay off the hot path.
            if new_highest_reward:
                save_checkpoint_atomic(
                    agent,
                    checkpoint_path=(best_checkpoint_path if best_checkpoint_path is not None
                                     else checkpoint_path),
                )
                print(f'Saved checkpoint for highest reward {agent.best_reward:.1f}.')
            if completed_games and best_checkpoint_path is not None:
                save_checkpoint_atomic(agent, checkpoint_path=checkpoint_path)

            if termination_requested:
                graceful_exit = True
                print('Termination requested. Completed games are logged; '
                      'unfinished games are discarded. Latest progress will be saved.',
                      flush=True)
                break
        else:
            graceful_exit = True
            print(f'Completed {games_this_run} game(s). Exiting.')

    except (KeyboardInterrupt, SystemExit):
        graceful_exit = True
        print('\nTermination requested. Latest progress will be saved.')
    finally:
        try:
            if graceful_exit:
                save_checkpoint_atomic(agent, checkpoint_path=checkpoint_path)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Snake agent.')
    parser.add_argument(
        '--games',
        type=positive_int,
        help='number of complete games to run before exiting (default: unlimited)',
    )
    parser.add_argument(
        '--checkpoint',
        default=DEFAULT_CHECKPOINT_PATH,
        help=f'checkpoint file to load and save (default: {DEFAULT_CHECKPOINT_PATH})',
    )
    parser.add_argument(
        '--csv',
        default=TRAINING_SCORES_CSV,
        help=f'training score CSV path (default: {TRAINING_SCORES_CSV})',
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='show the game window while training',
    )
    parser.add_argument(
        '--reset-epsilon',
        action='store_true',
        help='reset exploration epsilon to 1.0 for this run',
    )
    parser.add_argument(
        '--start-length',
        type=start_length_arg,
        help='start each game coiled into a solid block of N segments, or of a length drawn '
             'from MIN-MAX per game, e.g. 4-30 (default: straight 3)',
    )
    parser.add_argument(
        '--epsilon-ramp',
        type=epsilon_ramp_arg,
        metavar='START-STOP',
        help='act greedily (epsilon 0) until a game gains START points, then raise epsilon '
             'linearly to its scheduled value at STOP points, e.g. 5-20',
    )
    parser.add_argument(
        '--no-compile',
        action='store_true',
        help='train with the eager model instead of torch.compile on CUDA or MPS',
    )
    args = parser.parse_args()
    try:
        validate_start_length(args.start_length)
    except ValueError as err:
        parser.error(str(err))
    train(
        max_games=args.games,
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        render=args.render,
        reset_epsilon=args.reset_epsilon,
        start_length=args.start_length,
        compile_model=not args.no_compile,
        epsilon_ramp=args.epsilon_ramp,
    )
