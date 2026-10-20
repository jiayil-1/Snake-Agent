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
