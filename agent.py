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
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import ARCHITECTURES, QTrainer, resolve_device
from configuration import DEFAULT_TRAINING, positive_int, resolve_preset
from helper import plot

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

class Agent:

    def __init__(self, board_width=640, board_height=480,
                 checkpoint_path=DEFAULT_CHECKPOINT_PATH,
                 replay_batch_size=None,
                 replay_warmup=None,
                 train_every=None, config=None, device='auto',
                 strict_checkpoint=False):
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
        self.environment_steps = 0
        self.board_width = board_width
        self.board_height = board_height
        self.grid_width = self.board_width // BLOCK_SIZE
        self.grid_height = self.board_height // BLOCK_SIZE

        # 3-channel board: head, food, body (all binary 0/1)
        self.board_channels = 3
        # Metadata: 4 dir + 3 danger + 4 food_dir + 1 food_dist = 12
        self.metadata_size = 12

        self.device = resolve_device(device)

        self.model = ARCHITECTURES[self.config['architecture']](
            board_height=self.grid_height,
            board_width=self.grid_width,
            metadata_size=self.metadata_size,
            output_size=3,
            **self.config['architecture_options'],
        ).to(self.device)
        self.trainer = QTrainer(
            self.model, gamma=self.gamma, device=self.device.type,
            **{key: settings[key] for key in
               ('lr', 'max_grad_norm', 'lr_step_size', 'lr_decay', 'min_lr')},
        )
        self._load_checkpoint(checkpoint_path)

        # Replay is repopulated in memory after checkpoint resume.
        self.memory = deque(maxlen=settings['max_memory'])

    def _load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except FileNotFoundError:
            print(f'No checkpoint found at {checkpoint_path}. Starting with a fresh model.')
            return

        wrapped = isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
        if (self.strict_checkpoint and wrapped and 'config' in checkpoint
                and checkpoint['config'] != self.config):
            raise ValueError('Checkpoint configuration differs from this preset; use a new --run name')
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
        head = game.snake[0]
        snake_len = len(game.snake)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Head, food, and head-to-tail body strength, all in [0, 1].
        board = np.zeros((3, self.grid_height, self.grid_width), dtype=np.float32)

        # Channel 0: head
        hx = int(head.x // BLOCK_SIZE)
        hy = int(head.y // BLOCK_SIZE)
        if 0 <= hx < self.grid_width and 0 <= hy < self.grid_height:
            board[0, hy, hx] = 1

        # Channel 1: food
        fx = int(game.food.x // BLOCK_SIZE)
        fy = int(game.food.y // BLOCK_SIZE)
        if 0 <= fx < self.grid_width and 0 <= fy < self.grid_height:
            board[1, fy, fx] = 1

        # Channel 2: body gradient, with the head at 1 and tail at 0.
        # A one-segment snake has only the head value, 1.
        denominator = max(1, snake_len - 1)
        for index, seg in enumerate(game.snake):
            tx = int(seg.x // BLOCK_SIZE)
            ty = int(seg.y // BLOCK_SIZE)
            if 0 <= tx < self.grid_width and 0 <= ty < self.grid_height:
                board[2, ty, tx] = 1.0 - index / denominator

        # ── Relative direction helpers ───────────────────────────────────
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(game.direction)
        dir_straight = clock_wise[idx]
        dir_right = clock_wise[(idx + 1) % 4]
        dir_left = clock_wise[(idx - 1) % 4]

        def _next_point(pt, direction):
            x, y = pt.x, pt.y
            if direction == Direction.RIGHT: x += BLOCK_SIZE
            elif direction == Direction.LEFT: x -= BLOCK_SIZE
            elif direction == Direction.DOWN: y += BLOCK_SIZE
            elif direction == Direction.UP: y -= BLOCK_SIZE
            return Point(x, y)

        # Danger: is there a wall or body one step in each relative direction?
        danger_straight = game.is_collision(_next_point(head, dir_straight))
        danger_right = game.is_collision(_next_point(head, dir_right))
        danger_left = game.is_collision(_next_point(head, dir_left))

        # Normalized Manhattan distance to food
        max_dist = self.grid_width + self.grid_height
        food_dist = (abs(head.x - game.food.x) + abs(head.y - game.food.y)) / (BLOCK_SIZE * max_dist)

        # ── Metadata vector (12 features) ────────────────────────────────
        metadata = np.array([
            # Current direction (4)
            int(dir_l), int(dir_r), int(dir_u), int(dir_d),
            # Danger flags (3)
            int(danger_straight), int(danger_right), int(danger_left),
            # Food-relative direction (4)
            int(game.food.x < head.x),  # food left
            int(game.food.x > head.x),  # food right
            int(game.food.y < head.y),  # food up
            int(game.food.y > head.y),  # food down
            # Normalized food distance (1)
            food_dist,
        ], dtype=np.float32)

        return {
            'metadata': metadata,
            'board_state': board,
        }

    def _state_to_tensor(self, state):
        return {
            'metadata': torch.tensor(state['metadata'], dtype=torch.float).unsqueeze(0).to(self.device),
            'board_state': torch.tensor(state['board_state'], dtype=torch.float).unsqueeze(0).to(self.device),
        }

    # ── Replay buffer interface ──────────────────────────────────────────
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.environment_steps += 1

    def train_long_memory(self):
        """Train one replay minibatch when warmup and update cadence are met."""
        if len(self.memory) < max(self.replay_warmup, self.replay_batch_size):
            return False
        if self.environment_steps % self.train_every != 0:
            return False

        mini_sample = random.sample(self.memory, self.replay_batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        return True

    def train_short_memory(self, state, action, reward, next_state, done):
        """Keep the legacy immediate single-transition update API available."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        self._update_epsilon()
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = self._state_to_tensor(state)
            with torch.no_grad():
                prediction = self.model(state0)[0]
            move = int(torch.argmax(prediction).item())
            final_move[move] = 1
        return final_move

    def _update_epsilon(self):
        games_since_reset = max(0, self.n_games - self.epsilon_start_game)
        settings = self.config['training']
        self.epsilon = max(
            settings['epsilon_min'],
            settings['epsilon_start'] * (settings['epsilon_decay'] ** games_since_reset),
        )


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


def train(max_games=None, checkpoint_path=DEFAULT_CHECKPOINT_PATH,
          csv_path=TRAINING_SCORES_CSV, render=False, reset_epsilon=False,
          config=None, device='auto', agent=None, best_checkpoint_path=None):
    """Train one agent. A separate best path enables latest-plus-best saving."""
    plot_scores = []
    plot_mean_scores = []
    recent_scores = deque(maxlen=100)
    game_number_offset = read_last_game_number(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    game = SnakeGameAI(render=render)
    if agent is None:
        agent = Agent(
            board_width=game.w,
            board_height=game.h,
            checkpoint_path=checkpoint_path,
            config=config,
            device=device,
            strict_checkpoint=best_checkpoint_path is not None,
        )
    if reset_epsilon:
        agent.epsilon_start_game = agent.n_games
        agent._update_epsilon()
        print(f'Epsilon reset to {agent.epsilon:.4f} for this run.')
    record = agent.record
    games_this_run = 0
    total_reward = 0
    termination_requested = False
    graceful_exit = False

    def request_termination(signum, _frame):
        nonlocal termination_requested
        if not termination_requested:
            termination_requested = True
            print(f'Received signal {signum}. Will stop after the current game step.',
                  flush=True)

    signals = [signal.SIGTERM, signal.SIGINT]
    if hasattr(signal, 'SIGUSR1'):
        signals.append(signal.SIGUSR1)
    previous_handlers = {sig: signal.signal(sig, request_termination) for sig in signals}

    try:
        print('Training started.', flush=True)
        while True:
            # 1. Get old state
            state_old = agent.get_state(game)

            # 2. Choose action
            final_move = agent.get_action(state_old)

            # 3. Execute action
            reward, done, score = game.play_step(final_move)
            total_reward += reward
            state_new = agent.get_state(game)

            # Keep each transition and train from replay at the configured cadence.
            agent.remember(state_old, final_move, reward, state_new, done)
            agent.train_long_memory()

            # ── Mid-episode termination ──────────────────────────────────
            if termination_requested and not done:
                graceful_exit = True
                message = ('Latest progress will be saved.' if best_checkpoint_path is not None
                           else 'Exiting without a checkpoint.')
                print(f'Termination requested mid-episode. {message}', flush=True)
                break

            # ── Episode finished ─────────────────────────────────────────
            if done:
                episode_reward = total_reward
                end_reason = game.end_reason
                print(f'\n\nFinished with reward {episode_reward:.1f}')
                total_reward = 0
                game.reset()
                agent.n_games += 1
                games_this_run += 1

                if score > record:
                    record = score
                agent.record = record

                new_highest_reward = episode_reward > agent.best_reward
                if new_highest_reward:
                    agent.best_reward = episode_reward
                    save_checkpoint_atomic(
                        agent,
                        metadata={
                            'record': record,
                            'best_reward': agent.best_reward,
                            'n_games': agent.n_games,
                            'epsilon_start_game': agent.epsilon_start_game,
                        },
                        checkpoint_path=(best_checkpoint_path if best_checkpoint_path is not None
                                         else checkpoint_path),
                    )
                    print(f'New highest reward {episode_reward:.1f}. Saved checkpoint.')

                run_limit_reached = (
                    max_games is not None and games_this_run >= max_games
                )

                print(f'Game {agent.n_games}  Score {score}  '
                      f'End reason: {end_reason}  Record: {record}  '
                      f'Eps: {agent.epsilon:.4f}')

                plot_scores.append(score)
                recent_scores.append(score)
                mean_score = sum(recent_scores) / len(recent_scores)
                plot_mean_scores.append(mean_score)
                plot(
                    plot_scores,
                    plot_mean_scores,
                    csv_path=csv_path,
                    png_path=None,
                    hidden=HIDE_PLOT,
                    game_number_offset=game_number_offset,
                    end_reason=end_reason,
                )
                if best_checkpoint_path is not None:
                    save_checkpoint_atomic(agent, checkpoint_path=checkpoint_path)

                if termination_requested:
                    graceful_exit = True
                    print('Termination requested. Episode completed. Exiting.',
                          flush=True)
                    break
                if run_limit_reached:
                    graceful_exit = True
                    print(f'Completed {games_this_run} game(s). Exiting.')
                    break

    except (KeyboardInterrupt, SystemExit):
        graceful_exit = True
        message = ('Latest progress will be saved.' if best_checkpoint_path is not None
                   else 'Exiting without a checkpoint.')
        print(f'\nTermination requested. {message}')
    finally:
        try:
            if graceful_exit and best_checkpoint_path is not None:
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
    args = parser.parse_args()
    train(
        max_games=args.games,
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        render=args.render,
        reset_epsilon=args.reset_epsilon,
    )
