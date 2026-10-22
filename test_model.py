import contextlib
import copy
import io
import os
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')

import torch
import numpy as np
from torch import nn

from agent import Agent, save_checkpoint_atomic
from game import BLOCK_SIZE, Direction, Point, SnakeGameAI
from model import Linear_QNet, QTrainer, RawSkip_QNet


class ResidualModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = Linear_QNet(24, 32, 12, 3)

    def test_pooled_encoder_and_output_shapes(self):
        for height, width in [(24, 32), (8, 12), (4, 4)]:
            model = Linear_QNet(height, width, 12, 3)
            board = torch.randn(2, 3, height, width)
            observed = []
            handles = [
                layer.register_forward_hook(
                    lambda module, inputs, output: observed.append(output.shape[-2:])
                ) for layer in model.cnn.modules() if isinstance(layer, nn.Conv2d)
            ]
            try:
                output = model({
                    'board_state': board,
                    'metadata': torch.randn(2, 12),
                })
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(len(observed), 7)
            self.assertEqual(observed[0], (height, width))
            self.assertTrue(
                all(shape == (height // 4, width // 4) for shape in observed[1:])
            )
            self.assertEqual(output.shape, (2, 3))
            self.assertEqual(
                model.cnn_fc[0].in_features,
                32 * (height // 4) * (width // 4),
            )
            metadata = torch.randn(2, 12)
            with torch.no_grad():
                batch = model({'board_state': board, 'metadata': metadata})
                single = model({'board_state': board[0], 'metadata': metadata[0]})
            self.assertEqual(batch.shape, (2, 3))
            torch.testing.assert_close(single, batch[:1])

    def test_residual_shortcut_and_gradients(self):
        blocks = list(self.model.cnn.children())[2:]
        self.assertEqual(len(blocks), 3)
        for block in blocks:
            zero_branch = copy.deepcopy(block)
            for parameter in zero_branch.parameters():
                nn.init.zeros_(parameter)
            x = torch.randn(2, 32, 7, 9, requires_grad=True)
            output = zero_branch(x)
            torch.testing.assert_close(output, torch.relu(x))
            output.sum().backward()
            torch.testing.assert_close(x.grad, (x > 0).float())

        output = self.model({
            'board_state': torch.randn(2, 3, 24, 32),
            'metadata': torch.randn(2, 12),
        })
        output.square().mean().backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_replay_warmup_cadence_and_reward_preservation(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            agent = Agent(
                replay_batch_size=2,
                replay_warmup=4,
                train_every=2,
            )

            class RecordingTrainer:
                def __init__(self):
                    self.batches = []
                    self.optimizer_steps = 0

                def train_batch(self, states, actions, rewards, next_states, dones):
                    self.batches.append((rewards.tolist(), dones.tolist()))
                    self.optimizer_steps += 1

            agent.trainer = RecordingTrainer()
            states = [
                {
                    'metadata': np.full(12, index, dtype=np.float32),
                    'board_state': np.full((3, 24, 32), index, dtype=np.uint8),
                }
                for index in range(6)
            ]
            rewards = [0.25, -0.5, 1.75, -2.0, 3.25, -4.5]
            dones = [False, False, False, True, False, True]

            for index, reward in enumerate(rewards):
                agent.remember(
                    states[index], [1, 0, 0], reward,
                    states[(index + 1) % len(states)], dones[index],
                )
                trained = agent.train_long_memory()
                self.assertEqual(trained, index in (3, 5))

            stored_rewards = agent.memory.sample(len(agent.memory), 'cpu')[2]
            self.assertEqual(sorted(stored_rewards.tolist()), sorted(rewards))
            self.assertEqual(len(agent.trainer.batches), 2)
            for batch_rewards, batch_dones in agent.trainer.batches:
                self.assertEqual(len(batch_rewards), 2)
                self.assertTrue(set(batch_rewards).issubset(set(rewards)))
                self.assertTrue(set(batch_dones).issubset({0.0, 1.0}))
            self.assertEqual(agent.trainer.optimizer_steps, 2)
            self.assertEqual(agent.environment_steps, 6)

    def test_vectorized_targets_match_loop_for_terminal_and_nonterminal_rows(self):
        predictions = torch.tensor(
            [[1.0, 2.0, 3.0], [-1.0, 4.0, 0.5], [7.0, 8.0, 9.0]],
            requires_grad=True,
        )
        actions = torch.tensor([1, 2, 0])
        rewards = torch.tensor([0.5, -2.0, 3.0])
        next_q_values = torch.tensor(
            [[2.0, 4.0, 1.0], [8.0, 3.0, 4.0], [-2.0, 1.0, 0.0]],
        )
        dones = torch.tensor([False, True, False])

        observed = QTrainer.build_td_target(
            predictions, actions, rewards, next_q_values, dones, gamma=0.9,
        )
        expected = predictions.detach().clone()
        for row in range(len(dones)):
            next_value = 0.0 if dones[row] else 0.9 * next_q_values[row].max()
            expected[row, actions[row]] = rewards[row] + next_value

        torch.testing.assert_close(observed, expected)
        self.assertFalse(observed.requires_grad)
        self.assertEqual(
            nn.MSELoss()(predictions, observed).item(),
            nn.MSELoss()(predictions, expected).item(),
        )

    def test_headless_replay_training_and_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            agent = Agent(
                replay_batch_size=2,
                replay_warmup=4,
                train_every=2,
            )
            game = SnakeGameAI(render=False)
            before = agent.model.head[-1].weight.detach().clone()

            for _ in range(8):
                state = agent.get_state(game)
                action = agent.get_action(state)
                reward, done, _ = game.play_step(action)
                next_state = agent.get_state(game)
                agent.remember(state, action, reward, next_state, done)
                agent.train_long_memory()
                if done:
                    game.reset()

            self.assertFalse(torch.equal(before, agent.model.head[-1].weight))
            self.assertTrue(all(torch.isfinite(p).all() for p in agent.model.parameters()))
            self.assertEqual(agent.environment_steps, 8)
            self.assertEqual(agent.trainer.optimizer_steps, 3)
            save_checkpoint_atomic(agent, {'record': 8, 'n_games': 100})
            restored = Agent(
                replay_batch_size=2,
                replay_warmup=4,
                train_every=2,
            )
            self.assertEqual((restored.record, restored.n_games), (8, 100))
            self.assertEqual(restored.environment_steps, 8)
            self.assertEqual(restored.trainer.optimizer_steps, 3)
            self.assertEqual(len(restored.memory), 0)
            self.assertEqual(restored.trainer.scheduler.state_dict(),
                             agent.trainer.scheduler.state_dict())
            self.assertEqual(restored.trainer.current_lr(), agent.trainer.current_lr())
            for key, value in agent.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[key])
            self.assertEqual(len(restored.trainer.optimizer.state),
                             len(agent.trainer.optimizer.state))

            for index in range(4):
                state = restored.get_state(game)
                action = restored.get_action(state)
                reward, done, _ = game.play_step(action)
                next_state = restored.get_state(game)
                restored.remember(state, action, reward, next_state, done)
                trained = restored.train_long_memory()
                self.assertEqual(trained, index == 3)
                if done:
                    game.reset()
            self.assertEqual(restored.trainer.optimizer_steps, 4)

            agent.model.save('weights.pth')
            restored._load_checkpoint('model/weights.pth')

    def test_legacy_scheduler_progress_maps_to_batch_update_cadence(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            agent = Agent()
            self.assertEqual(
                (agent.replay_batch_size, agent.replay_warmup, agent.train_every),
                (64, 1000, 4),
            )
            self.assertEqual(agent.trainer.scheduler.step_size, 500)
            save_checkpoint_atomic(agent)
            checkpoint_path = Path('model/model.pth')
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            scheduler_state = checkpoint['scheduler_state_dict']
            scheduler_state.update({
                'step_size': 2000,
                'last_epoch': 3999,
                '_step_count': 4000,
                '_last_lr': [0.00095],
            })
            for group in checkpoint['optimizer_state_dict']['param_groups']:
                group['lr'] = 0.00095
            torch.save(checkpoint, 'legacy-scheduler.pth')

            restored = Agent(checkpoint_path='legacy-scheduler.pth')
            self.assertEqual(restored.trainer.scheduler.step_size, 500)
            self.assertEqual(restored.trainer.scheduler.last_epoch, 999)
            self.assertEqual(restored.trainer.scheduler._step_count, 1000)
            self.assertEqual(restored.trainer.current_lr(), 0.00095)

            restored.trainer.optimizer.zero_grad()
            for parameter in restored.model.parameters():
                parameter.grad = torch.zeros_like(parameter)
            restored.trainer.optimizer.step()
            restored.trainer.scheduler.step()
            self.assertAlmostEqual(restored.trainer.current_lr(), 0.00095 * 0.95)

    def test_old_checkpoint_rejection_preserves_fresh_parameters(self):
        # Old CNN shapes, with matching metadata/head tensors that would otherwise
        # be copied before load_state_dict reports the incompatible CNN.
        old_state = {
            key: torch.full_like(value, 0.25)
            for key, value in self.model.state_dict().items()
            if not key.startswith('cnn.')
        }
        old_state['cnn.0.weight'] = torch.zeros(16, 3, 3, 3)
        old_state['cnn.0.bias'] = torch.zeros(16)
        old_state['cnn.3.weight'] = torch.zeros(32, 16, 3, 3)
        old_state['cnn.3.bias'] = torch.zeros(32)
        old_state['cnn_fc.0.weight'] = torch.zeros(128, 1536)
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            agent = Agent()
            before = copy.deepcopy(agent.model.state_dict())
            for wrapped in [False, True]:
                with self.subTest(wrapped=wrapped):
                    checkpoint = ({'model_state_dict': old_state,
                                   'metadata': {'record': 99, 'n_games': 500}}
                                  if wrapped else old_state)
                    path = Path('old.pth')
                    torch.save(checkpoint, path)
                    original_bytes = path.read_bytes()
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        agent._load_checkpoint(path)
                    self.assertIn('architecture mismatch', output.getvalue())
                    for key, value in before.items():
                        torch.testing.assert_close(value, agent.model.state_dict()[key],
                                                   rtol=0, atol=0)
                    self.assertEqual((agent.record, agent.n_games, agent.epsilon), (0, 0, 1.0))
                    self.assertEqual(path.read_bytes(), original_bytes)
                    self.assertEqual(len(agent.trainer.optimizer.state), 0)


class RawSkipModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = RawSkip_QNet(24, 32, 12, 3)

    def test_full_resolution_convs_and_output_shapes(self):
        for height, width in [(24, 32), (8, 12), (4, 4)]:
            model = RawSkip_QNet(height, width, 12, 3)
            board = torch.randn(2, 3, height, width)
            metadata = torch.randn(2, 12)
            observed = []
            handles = [
                layer.register_forward_hook(
                    lambda module, inputs, output: observed.append(tuple(output.shape[1:]))
                ) for layer in model.cnn.modules() if isinstance(layer, nn.Conv2d)
            ]
            try:
                with torch.no_grad():
                    batch = model({'board_state': board, 'metadata': metadata})
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(len(observed), 8)
            self.assertTrue(all(shape == (6, height, width) for shape in observed[:-1]))
            self.assertEqual(observed[-1], (3, height, width))
            self.assertEqual(batch.shape, (2, 3))
            self.assertEqual(model.cnn_fc[0].in_features, 3 * (height // 4) * (width // 4))
            with torch.no_grad():
                single = model({'board_state': board[0], 'metadata': metadata[0]})
            torch.testing.assert_close(single, batch[:1])

    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 46796)

    def test_raw_board_shortcut_and_gradients(self):
        zero_branch = copy.deepcopy(self.model)
        for parameter in zero_branch.cnn.parameters():
            nn.init.zeros_(parameter)
        board = torch.randn(2, 3, 24, 32)
        pool = nn.MaxPool2d(2)
        torch.testing.assert_close(
            zero_branch.encode_board(board),
            torch.flatten(pool(pool(torch.relu(board))), 1),
        )

        output = self.model({
            'board_state': torch.randn(2, 3, 24, 32),
            'metadata': torch.randn(2, 12),
        })
        output.square().mean().backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


class BatchedActionTests(unittest.TestCase):
    def make_agent(self, directory, epsilon):
        return Agent(
            config={'architecture': 'mlp', 'training': {
                'epsilon_start': epsilon, 'epsilon_min': epsilon,
            }},
            device='cpu', checkpoint_path=Path(directory) / 'missing.pth',
        )

    def count_forward_passes(self, agent):
        calls = []
        handle = agent.model.register_forward_hook(lambda *_: calls.append(1))
        self.addCleanup(handle.remove)
        return calls

    def test_greedy_actions_share_one_forward_pass_and_match_single_states(self):
        rng = np.random.default_rng(0)
        states = [{'board_state': rng.random((3, 24, 32), dtype=np.float32),
                   'metadata': rng.random(12, dtype=np.float32)} for _ in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            agent = self.make_agent(directory, 0.0)
            calls = self.count_forward_passes(agent)
            moves = agent.get_actions(states)
            self.assertEqual(len(calls), 1)
            self.assertEqual(moves, [agent.get_action(state) for state in states])
            self.assertEqual(len(calls), 1 + len(states))

    def test_random_actions_skip_the_model(self):
        states = [{'board_state': np.zeros((3, 24, 32), dtype=np.float32),
                   'metadata': np.zeros(12, dtype=np.float32)}] * 4
        with tempfile.TemporaryDirectory() as directory:
            agent = self.make_agent(directory, 1.0)
            calls = self.count_forward_passes(agent)
            moves = agent.get_actions(states)
            self.assertEqual(calls, [])
            self.assertTrue(all(sorted(move) == [0, 0, 1] for move in moves))

    def test_epsilon_scales_make_zero_scaled_rows_greedy(self):
        states = [{'board_state': np.zeros((3, 24, 32), dtype=np.float32),
                   'metadata': np.zeros(12, dtype=np.float32)}] * 3
        with tempfile.TemporaryDirectory() as directory:
            agent = self.make_agent(directory, 1.0)
            batch_sizes = []
            handle = agent.model.register_forward_hook(
                lambda _module, inputs, _output: batch_sizes.append(inputs[0]['metadata'].shape[0]))
            self.addCleanup(handle.remove)
            agent.get_actions(states, [0.0, 1.0, 0.0])
            self.assertEqual(batch_sizes, [2])


class EpsilonRampTests(unittest.TestCase):
    def test_scale_is_zero_until_start_then_linear_to_one_at_stop(self):
        from agent import epsilon_ramp_scale
        self.assertEqual([epsilon_ramp_scale(points, (2, 6)) for points in range(9)],
                         [0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])

    def test_training_ramps_each_game_from_its_starting_score(self):
        import agent as agent_module

        class ScoringGame(SnakeGameAI):
            """Scores a point every step and ends after five steps."""
            def play_step(self, action):
                self.score += 1
                self.end_reason = 'test'
                return 0.0, self.score - self.start_score == 5, self.score

        scales = []
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(config={'architecture': 'mlp', 'training': {'parallel_games': 1}},
                          device='cpu', checkpoint_path=Path(directory) / 'missing.pth')
            original = agent.get_actions
            agent.get_actions = lambda states, epsilon_scales=None: (
                scales.append(epsilon_scales) or original(states, epsilon_scales))
            with patch.object(agent_module, 'SnakeGameAI', ScoringGame), \
                    contextlib.redirect_stdout(io.StringIO()):
                agent_module.train(max_games=1, agent=agent, start_length=10,
                                   checkpoint_path=Path(directory) / 'latest.pth',
                                   csv_path=Path(directory) / 'scores.csv',
                                   epsilon_ramp=(1, 3))
        self.assertEqual(scales, [[0.0], [0.0], [0.5], [1.0], [1.0]])

    def test_invalid_ramps_are_rejected(self):
        import argparse
        from configuration import epsilon_ramp_arg, validate_epsilon_ramp
        self.assertEqual(epsilon_ramp_arg('0-20'), (0, 20))
        for value in ('5', '5-5', '6-5', '-1-3', 'a-b', '3-'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                epsilon_ramp_arg(value)
        for value in ((5, 5), (1.0, 2), [1, 2], (1, 2, 3)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'epsilon_ramp'):
                validate_epsilon_ramp(value)


class ReplayMemoryTests(unittest.TestCase):
    def test_ring_buffer_replaces_oldest_and_samples_aligned_rows(self):
        from agent import ReplayMemory
        from state import StateLayout

        spec = {'has_board': True, 'board_channels': 3, 'board_height': 2,
                'board_width': 2, 'metadata_size': 12}

        def state(value):
            return {'metadata': np.full(12, value, dtype=np.float32),
                    'board_state': np.arange(12, dtype=np.float32).reshape(3, 2, 2) + value}

        memory = ReplayMemory(3, StateLayout(spec))
        for index in range(5):
            memory.append(state(index), index % 3, index / 4, state(index + 10), index == 4)
        self.assertEqual(len(memory), 3)

        states, actions, rewards, next_states, dones = memory.sample(3, 'cpu')
        order = torch.argsort(rewards)
        torch.testing.assert_close(rewards[order], torch.tensor([0.5, 0.75, 1.0]))
        self.assertEqual(actions[order].tolist(), [2, 0, 1])
        self.assertEqual(dones[order].tolist(), [0.0, 0.0, 1.0])
        for row, value in enumerate([2, 3, 4]):
            for batch, offset in ((states, 0), (next_states, 10)):
                expected = state(value + offset)
                for key in ('metadata', 'board_state'):
                    np.testing.assert_array_equal(batch[key][order][row].numpy(), expected[key])

        compact = StateLayout({**spec, 'has_board': False, 'board_channels': 0})
        self.assertEqual([field[0] for field in compact.fields], ['metadata'])
        self.assertEqual(compact.size, 12)


class FusedOptimizerTests(unittest.TestCase):
    def test_unfused_optimizer_checkpoint_resumes_fused(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            agent = Agent(replay_batch_size=2, replay_warmup=2, train_every=1)
            game = SnakeGameAI(render=False)
            for _ in range(3):
                state = agent.get_state(game)
                reward, done, _ = game.play_step([1, 0, 0])
                agent.remember(state, [1, 0, 0], reward, agent.get_state(game), done)
                agent.train_long_memory()
                if done:
                    game.reset()
            save_checkpoint_atomic(agent)
            checkpoint = torch.load('model/model.pth', weights_only=True)
            optimizer_state = checkpoint['optimizer_state_dict']
            for group in optimizer_state['param_groups']:
                group['fused'] = None
            for state in optimizer_state['state'].values():
                state['step'] = state['step'].cpu()
            torch.save(checkpoint, 'unfused.pth')

            restored = Agent(replay_batch_size=2, replay_warmup=2, train_every=1,
                             checkpoint_path='unfused.pth')
            optimizer = restored.trainer.optimizer
            self.assertTrue(all(group['fused'] for group in optimizer.param_groups))
            for parameter, state in optimizer.state.items():
                self.assertEqual(state['step'].device, parameter.device)
                self.assertEqual(state['step'].item(), 2)
            observation = restored.get_state(game)
            restored.trainer.train_step(observation, [0, 1, 0], 1.0, observation, False)
            self.assertTrue(all(state['step'].item() == 3 for state in optimizer.state.values()))


class RewardConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.rewards = {
            'step_penalty': -2,
            'progress_reward': 3,
            'progress_penalty': -4,
            'food_reward': 7,
            'collision_penalty': -13,
            'timeout_penalty': -9,
            'win_reward': 50,
        }

    def prepare_game(self, game, head, food):
        game.head = head
        game.snake = [head, Point(head.x - 20, head.y), Point(head.x - 40, head.y)]
        game.direction = Direction.RIGHT
        game.food = food
        game.score = 0
        game.frame_iteration = 0

    def test_custom_rewards_apply_per_environment_and_do_not_mutate_defaults(self):
        custom = SnakeGameAI(w=200, h=200, render=False, rewards=self.rewards)
        default = SnakeGameAI(w=200, h=200, render=False)
        self.prepare_game(custom, Point(100, 100), Point(0, 0))
        self.prepare_game(default, Point(100, 100), Point(0, 0))

        custom_reward, custom_done, _ = custom.play_step([1, 0, 0])
        default_reward, default_done, _ = default.play_step([1, 0, 0])

        self.assertFalse(custom_done)
        self.assertFalse(default_done)
        self.assertEqual(custom_reward, -6)
        self.assertEqual(default_reward, -0.01)
        self.assertEqual(SnakeGameAI(render=False).rewards['food_reward'], 10)

    def test_food_progress_collision_and_timeout_rewards(self):
        game = SnakeGameAI(w=200, h=200, render=False, rewards=self.rewards)
        self.prepare_game(game, Point(100, 100), Point(140, 100))
        self.assertEqual(game.play_step([1, 0, 0])[0], 1)

        self.prepare_game(game, Point(100, 100), Point(120, 100))
        food_reward, done, score = game.play_step([1, 0, 0])
        self.assertEqual((food_reward, done, score), (5, False, 1))

        self.prepare_game(game, Point(100, 100), Point(180, 0))
        game.head = Point(0, 100)
        game.snake = [game.head, Point(20, 100), Point(40, 100)]
        game.direction = Direction.LEFT
        self.assertEqual(game.play_step([1, 0, 0])[:2], (-15, True))
        self.assertEqual(game.end_reason, 'wall-collision')

        self.prepare_game(game, Point(100, 100), Point(0, 0))
        game.frame_iteration = 100 * len(game.snake)
        self.assertEqual(game.play_step([1, 0, 0])[:2], (-11, True))
        self.assertEqual(game.end_reason, 'timeout')

    def test_eating_the_last_free_cell_wins(self):
        game = SnakeGameAI(w=80, h=20, render=False, rewards=self.rewards)
        self.prepare_game(game, Point(40, 0), Point(60, 0))
        self.assertEqual(game.play_step([1, 0, 0]), (55, True, 1))
        self.assertEqual(game.end_reason, 'win')

    def test_food_lands_on_the_only_free_cell(self):
        game = SnakeGameAI(w=200, h=200, render=False)
        free = Point(120, 60)
        game.snake = [Point(x, y) for y in range(0, 200, 20) for x in range(0, 200, 20)
                      if Point(x, y) != free]
        self.assertTrue(game._place_food())
        self.assertEqual(game.food, free)


class StartLengthTests(unittest.TestCase):
    def test_random_start_is_a_solid_block_with_the_head_outside_and_a_safe_first_move(self):
        random.seed(3)
        layouts = set()
        facing = {
            (BLOCK_SIZE, 0): Direction.RIGHT, (-BLOCK_SIZE, 0): Direction.LEFT,
            (0, BLOCK_SIZE): Direction.DOWN, (0, -BLOCK_SIZE): Direction.UP,
        }
        for length in (1, 2, 3, 25, 192):
            game = SnakeGameAI(render=False, start_length=length)
            for _ in range(20):
                game.reset()
                snake = game.snake
                self.assertEqual(len(snake), length)
                self.assertEqual(game.score, length - 3)
                self.assertEqual(len(set(snake)), length)
                self.assertIs(game.head, snake[0])
                self.assertNotIn(game.food, snake)
                for a, b in zip(snake, snake[1:]):
                    self.assertEqual(abs(a.x - b.x) + abs(a.y - b.y), BLOCK_SIZE)
                for pt in snake:
                    self.assertTrue(0 <= pt.x < game.w and 0 <= pt.y < game.h)
                    self.assertEqual((pt.x % BLOCK_SIZE, pt.y % BLOCK_SIZE), (0, 0))
                if length > 1:
                    # The head never faces back into its neck.
                    neck = (snake[1].x - snake[0].x, snake[1].y - snake[0].y)
                    self.assertNotEqual(game.direction, facing[neck])
                # Flood the padded bounding box from its corner: every free cell
                # must be reached (no holes), and so must the cell ahead of the head.
                cells = {(pt.x // BLOCK_SIZE, pt.y // BLOCK_SIZE) for pt in snake}
                xs, ys = [x for x, _ in cells], [y for _, y in cells]
                box = {(x, y) for x in range(min(xs) - 1, max(xs) + 2)
                       for y in range(min(ys) - 1, max(ys) + 2)}
                outside, frontier = set(), [(min(xs) - 1, min(ys) - 1)]
                while frontier:
                    x, y = frontier.pop()
                    if (x, y) in box and (x, y) not in cells and (x, y) not in outside:
                        outside.add((x, y))
                        frontier += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                self.assertEqual(outside, box - cells)
                dx, dy = {v: k for k, v in facing.items()}[game.direction]
                ahead = ((game.head.x + dx) // BLOCK_SIZE, (game.head.y + dy) // BLOCK_SIZE)
                self.assertIn(ahead, outside)
                if length == 25:
                    layouts.add(tuple(snake))
                _, done, _ = game.play_step([1, 0, 0])
                self.assertFalse(done)
        self.assertGreater(len(layouts), 15)

    def test_range_start_draws_each_length_inclusively(self):
        random.seed(5)
        game = SnakeGameAI(render=False, start_length=(4, 7))
        lengths = set()
        for _ in range(60):
            game.reset()
            lengths.add(len(game.snake))
            self.assertEqual(game.score, len(game.snake) - 3)
        self.assertEqual(lengths, {4, 5, 6, 7})

    def test_default_start_is_unchanged_and_invalid_lengths_fail(self):
        game = SnakeGameAI(render=False)
        self.assertEqual(game.snake, [Point(320, 240), Point(300, 240), Point(280, 240)])
        self.assertEqual(game.direction, Direction.RIGHT)
        self.assertEqual(game.score, 0)
        for length in (0, -1, 385, 3.0, True, (5,), (5, 4), (0, 4), (4, 385), (4, 5, 6), (4.0, 5)):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, 'start_length'):
                SnakeGameAI(render=False, start_length=length)


if __name__ == '__main__':
    unittest.main()
