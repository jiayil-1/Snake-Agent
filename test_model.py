import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')

import torch
import numpy as np
from torch import nn

from agent import Agent, save_checkpoint_atomic
from game import SnakeGameAI
from model import Linear_QNet, QTrainer


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

                def train_step(self, states, actions, rewards, next_states, dones):
                    self.batches.append((tuple(rewards), tuple(dones)))
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

            self.assertEqual([experience[2] for experience in agent.memory], rewards)
            self.assertEqual(len(agent.trainer.batches), 2)
            for batch_rewards, batch_dones in agent.trainer.batches:
                self.assertEqual(len(batch_rewards), 2)
                self.assertTrue(set(batch_rewards).issubset(set(rewards)))
                self.assertTrue(all(isinstance(done, bool) for done in batch_dones))
            self.assertEqual(agent.trainer.optimizer_steps, 2)
            self.assertEqual(agent.environment_steps, 6)

    def test_vectorized_targets_match_loop_for_terminal_and_nonterminal_rows(self):
        predictions = torch.tensor(
            [[1.0, 2.0, 3.0], [-1.0, 4.0, 0.5], [7.0, 8.0, 9.0]],
            requires_grad=True,
        )
        actions = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
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
            expected[row, actions[row].argmax()] = rewards[row] + next_value

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


if __name__ == '__main__':
    unittest.main()
