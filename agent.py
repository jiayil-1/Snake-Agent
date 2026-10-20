import torch
import random
import numpy as np
import os
import signal
from copy import deepcopy
from datetime import datetime
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

# ── Hyperparameters ──────────────────────────────────────────────────────
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.998
HIDE_PLOT = True
SAVE_EVERY = 100

class Agent:

    def __init__(self, board_width=640, board_height=480):
        self.n_games = 0
        self.record = 0
        self.epsilon = EPSILON_START
        self.gamma = GAMMA
        self.board_width = board_width
        self.board_height = board_height
        self.grid_width = self.board_width // BLOCK_SIZE
        self.grid_height = self.board_height // BLOCK_SIZE

        # 3-channel board: head, food, body (all binary 0/1)
        self.board_channels = 3
        # Metadata: 4 dir + 3 danger + 4 food_dir + 1 food_dist = 12
        self.metadata_size = 12

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model = Linear_QNet(
            board_height=self.grid_height,
            board_width=self.grid_width,
            metadata_size=self.metadata_size,
            output_size=3,
        ).to(self.device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self._load_checkpoint('model/model.pth')

        # Simple replay buffer (deque) — ~4.6 KB per transition at uint8
        # 100K transitions ≈ 460 MB, well under 16 GB
        self.memory = deque(maxlen=MAX_MEMORY)

    def _load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            print(f'No checkpoint found at {checkpoint_path}. Starting with a fresh model.')
            return

        wrapped = isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
        model_state = checkpoint['model_state_dict'] if wrapped else checkpoint
        # Strict loading can copy matching tensors before reporting a mismatch.
        # Validate on a copy so rejected checkpoints never alter the live model.
        candidate = deepcopy(self.model)
        try:
            candidate.load_state_dict(model_state)
        except RuntimeError as err:
            print(f'Checkpoint architecture mismatch ({err}). Starting with a fresh model.')
            return
        self.model.load_state_dict(candidate.state_dict())

        if wrapped:
            optimizer_state = checkpoint.get('optimizer_state_dict')
            if optimizer_state is not None:
                try:
                    self.trainer.optimizer.load_state_dict(optimizer_state)
                except ValueError as err:
                    print(f'Optimizer state mismatch ({err}). Continuing with fresh optimizer state.')

            scheduler_state = checkpoint.get('scheduler_state_dict')
            if scheduler_state is not None and hasattr(self.trainer, 'scheduler'):
                try:
                    self.trainer.scheduler.load_state_dict(scheduler_state)
                except ValueError as err:
                    print(f'Scheduler state mismatch ({err}). Continuing with fresh scheduler state.')

            metadata = checkpoint.get('metadata', {})
            print('Loaded checkpoint metadata:', metadata)
            self.record = metadata.get('record', 0)
            self.n_games = metadata.get('n_games', 0)
            self.epsilon = max(
                EPSILON_MIN,
                EPSILON_START * (EPSILON_DECAY ** self.n_games),
            )
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

        # ── 3-channel board (all binary 0/1) ──────────────────────────
        board = np.zeros((3, self.grid_height, self.grid_width), dtype=np.uint8)

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

        # Channel 2: body (all segments including head)
        for seg in game.snake:
            tx = int(seg.x // BLOCK_SIZE)
            ty = int(seg.y // BLOCK_SIZE)
            if 0 <= tx < self.grid_width and 0 <= ty < self.grid_height:
                board[2, ty, tx] = 1

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

    def train_long_memory(self):
        """Sample a batch from replay buffer and train."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train on a single transition (online TD update)."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        self.epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY ** self.n_games))
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


def save_checkpoint_atomic(agent, metadata=None, file_name='model.pth'):
    model_folder_path = './model'
    os.makedirs(model_folder_path, exist_ok=True)

    final_path = os.path.join(model_folder_path, file_name)
    temp_path = f"{final_path}.tmp.{os.getpid()}"

    checkpoint_metadata = {
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    if metadata:
        checkpoint_metadata.update(metadata)

    checkpoint = {
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.trainer.optimizer.state_dict(),
        'scheduler_state_dict': agent.trainer.scheduler.state_dict(),
        'metadata': checkpoint_metadata,
    }

    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, final_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def train():
    plot_scores = []
    plot_mean_scores = []
    recent_scores = deque(maxlen=100)
    game = SnakeGameAI(render=False)
    agent = Agent(board_width=game.w, board_height=game.h)
    record = agent.record
    total_reward = 0
    termination_requested = False
    termination_signal = None

    def request_termination(signum, _frame):
        nonlocal termination_requested, termination_signal
        if not termination_requested:
            termination_requested = True
            termination_signal = signum
            print(f'Received signal {signum}. Will terminate after a safe checkpoint.',
                  flush=True)

    signal.signal(signal.SIGTERM, request_termination)
    signal.signal(signal.SIGINT, request_termination)
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, request_termination)

    try:
        while True:
            # 1. Get old state
            state_old = agent.get_state(game)

            # 2. Choose action
            final_move = agent.get_action(state_old)

            # 3. Execute action
            reward, done, score = game.play_step(final_move)
            total_reward += reward
            state_new = agent.get_state(game)

            # 4. Online TD update + store in replay buffer
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            # ── Mid-episode termination ──────────────────────────────────
            if termination_requested and not done:
                print('Termination requested mid-episode. Saving checkpoint…',
                      flush=True)
                save_checkpoint_atomic(agent, metadata={
                    'record': record,
                    'n_games': agent.n_games,
                    'terminated': True,
                    'termination_signal': termination_signal,
                })
                print('Final checkpoint saved. Exiting.', flush=True)
                break

            # ── Episode finished ─────────────────────────────────────────
            if done:
                print(f'\n\nFinished with reward {total_reward:.1f}')
                total_reward = 0
                game.reset()
                agent.n_games += 1

                # Experience replay
                agent.train_long_memory()

                if score > record:
                    record = score

                # Save checkpoint every SAVE_EVERY games or on new record
                if score >= record or agent.n_games % SAVE_EVERY == 0:
                    save_checkpoint_atomic(agent, metadata={
                        'record': record,
                        'n_games': agent.n_games,
                    })
                    print(f'Saving model @ record: {record}')

                print(f'Game {agent.n_games}  Score {score}  Record: {record}  '
                      f'Eps: {agent.epsilon:.4f}')

                plot_scores.append(score)
                recent_scores.append(score)
                mean_score = sum(recent_scores) / len(recent_scores)
                plot_mean_scores.append(mean_score)
                plot(
                    plot_scores,
                    plot_mean_scores,
                    csv_path='training_scores.csv',
                    png_path=None,
                    hidden=HIDE_PLOT,
                )

                if termination_requested:
                    print('Termination requested. Episode completed and '
                          'checkpoint saved. Exiting.', flush=True)
                    break

    except (KeyboardInterrupt, SystemExit):
        print('\nTermination requested. Saving final checkpoint…')
        save_checkpoint_atomic(agent, metadata={
            'record': record,
            'n_games': agent.n_games,
            'terminated': True,
            'termination_signal': termination_signal,
        })
        print('Final checkpoint saved. Exiting.')
    finally:
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass


if __name__ == '__main__':
    train()