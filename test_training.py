import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'snake-matplotlib'))

import numpy as np
import torch

from configuration import load_presets, resolve_preset
from model import ARCHITECTURES, QTrainer
from state import encode_state, get_observation_spec


class ConfigurationTests(unittest.TestCase):
    def test_defaults_and_json_presets(self):
        presets = load_presets(Path(__file__).with_name('models.json'))
        self.assertEqual(set(presets), {'residual_cnn', 'mlp'})
        cnn = presets['residual_cnn']
        self.assertEqual(cnn['training']['replay_batch_size'], 64)
        self.assertEqual(cnn['training']['replay_warmup'], 1000)
        self.assertEqual(cnn['training']['train_every'], 4)
        self.assertEqual(cnn['training']['lr_step_size'], 500)
        self.assertEqual(presets['mlp']['architecture_options']['hidden_sizes'], [128, 128])
        self.assertEqual(cnn['state_representation'], {
            'encoder': 'board_metadata', 'options': {'body_encoding': 'gradient'},
        })
        self.assertEqual(cnn['rewards']['food_reward'], 10)
        self.assertEqual(cnn['architecture_options']['channels'], 32)
        self.assertEqual(resolve_preset(cnn), cnn)

    def test_invalid_presets_fail(self):
        bad_presets = [
            [], {}, {'architecture': 'unknown'},
            {'architecture': 'mlp', 'extra': 1},
            {'architecture': 'mlp', 'training': {'typo': 1}},
            {'architecture': 'mlp', 'training': {'lr': 0}},
            {'architecture': 'mlp', 'training': {'gamma': 1.1}},
            {'architecture': 'mlp', 'training': {'epsilon_min': 0.8, 'epsilon_start': 0.5}},
            {'architecture': 'mlp', 'training': {'epsilon_decay': float('nan')}},
            {'architecture': 'mlp', 'training': {'replay_batch_size': True}},
            {'architecture': 'mlp', 'training': {'max_memory': 2, 'replay_warmup': 3}},
            {'architecture': 'mlp', 'training': {'min_lr': 1}},
            {'architecture': 'mlp', 'architecture_options': {'hidden_sizes': []}},
            {'architecture': 'mlp', 'architecture_options': {'hidden_sizes': [128, -1]}},
            {'architecture': 'residual_cnn', 'architecture_options': {'hidden_sizes': [128]}},
            {'architecture': 'mlp', 'state_representation': {'encoder': 'unknown'}},
            {'architecture': 'mlp', 'state_representation': {
                'encoder': 'board_metadata', 'options': {'body_encoding': 'other'},
            }},
            {'architecture': 'residual_cnn', 'state_representation': {'encoder': 'metadata_only'}},
            {'architecture': 'mlp', 'rewards': {'food_reward': float('inf')}},
            {'architecture': 'mlp', 'architecture_options': {'hidden_sizes': [4], 'unused': 1}},
            {'architecture': 'residual_cnn', 'architecture_options': {'residual_blocks': 0}},
        ]
        for preset in bad_presets:
            with self.subTest(preset=preset), self.assertRaises(ValueError):
                resolve_preset(preset)

    def test_json_syntax_duplicate_keys_and_preset_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'models.json'
            for content in ['{', '{}', '{"../bad": {"architecture": "mlp"}}',
                            '{"a": {"architecture": "mlp", "architecture": "residual_cnn"}}']:
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(ValueError):
                        load_presets(path)


class ArchitectureTests(unittest.TestCase):
    def test_both_networks_train_on_shared_state(self):
        torch.manual_seed(7)
        state = {
            'board_state': np.random.default_rng(7).random((3, 24, 32), dtype=np.float32),
            'metadata': np.ones(12, dtype=np.float32),
        }
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                config = resolve_preset({'architecture': architecture})
                factory = ARCHITECTURES[architecture]
                model = factory(24, 32, 12, 3, **config['architecture_options'])
                before = copy.deepcopy(model.state_dict())
                tensors = {key: torch.as_tensor(value) for key, value in state.items()}
                self.assertEqual(model(tensors).shape, (1, 3))
                batch = {key: value.unsqueeze(0).repeat(2, *([1] * value.dim()))
                         for key, value in tensors.items()}
                self.assertEqual(model(batch).shape, (2, 3))
                trainer = QTrainer(model, lr=0.001, gamma=0.99, device='cpu')
                trainer.train_step(state, [1, 0, 0], 10, state, True)
                self.assertEqual(trainer.optimizer_steps, 1)
                self.assertTrue(any(not torch.equal(before[key], value)
                                    for key, value in model.state_dict().items()))
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_state_encoders_expose_only_their_required_inputs(self):
        from game import SnakeGameAI

        game = SnakeGameAI(render=False)
        gradient_config = resolve_preset({'architecture': 'mlp'})
        gradient = encode_state(gradient_config['state_representation'], game, 32, 24)
        self.assertEqual(set(gradient), {'board_state', 'metadata'})
        self.assertEqual(gradient['board_state'].shape, (3, 24, 32))
        self.assertEqual(gradient['metadata'].shape, (12,))
        expected_board = np.zeros((3, 24, 32), dtype=np.float32)
        head = game.snake[0]
        expected_board[0, 12, 16] = 1.0
        expected_board[1, int(game.food.y // 20), int(game.food.x // 20)] = 1.0
        expected_board[2, 12, 16] = 1.0
        expected_board[2, 12, 15] = 0.75
        expected_board[2, 12, 14] = 0.5
        expected_metadata = np.array([
            0, 1, 0, 0, 0, 0, 0,
            int(game.food.x < head.x), int(game.food.x > head.x),
            int(game.food.y < head.y), int(game.food.y > head.y),
            (abs(head.x - game.food.x) + abs(head.y - game.food.y)) / (20 * 56),
        ], dtype=np.float32)
        np.testing.assert_array_equal(gradient['board_state'], expected_board)
        np.testing.assert_allclose(gradient['metadata'], expected_metadata)

        binary_config = resolve_preset({
            'architecture': 'mlp',
            'state_representation': {
                'encoder': 'board_metadata', 'options': {'body_encoding': 'binary'},
            },
        })
        binary = encode_state(binary_config['state_representation'], game, 32, 24)
        self.assertEqual(set(np.unique(binary['board_state'][2])), {0.0, 1.0})
        self.assertEqual(binary['board_state'][2].sum(), len(game.snake))
        self.assertTrue(np.all(gradient['board_state'][2] <= 1.0))
        self.assertTrue(np.any((gradient['board_state'][2] > 0) & (gradient['board_state'][2] < 1)))

        compact_config = resolve_preset({
            'architecture': 'mlp', 'state_representation': {'encoder': 'metadata_only'},
        })
        compact = encode_state(compact_config['state_representation'], game, 32, 24)
        self.assertEqual(set(compact), {'metadata'})
        np.testing.assert_array_equal(compact['metadata'], gradient['metadata'])
        self.assertEqual(get_observation_spec(compact_config['state_representation'], 24, 32), {
            'has_board': False, 'board_channels': 0, 'board_height': 0,
            'board_width': 0, 'metadata_size': 12,
        })

    def test_network_options_and_metadata_only_training(self):
        config = resolve_preset({
            'architecture': 'mlp',
            'architecture_options': {'hidden_sizes': [16, 8]},
            'state_representation': {'encoder': 'metadata_only'},
        })
        spec = get_observation_spec(config['state_representation'], 24, 32)
        model = ARCHITECTURES['mlp'](
            24, 32, 12, 3, state_spec=spec, **config['architecture_options'],
        )
        state = {'metadata': np.arange(12, dtype=np.float32)}
        self.assertEqual(model({key: torch.as_tensor(value) for key, value in state.items()}).shape, (1, 3))
        trainer = QTrainer(model, lr=0.001, gamma=0.99, device='cpu')
        trainer.train_step(state, [1, 0, 0], 1.0, state, True)
        self.assertEqual(trainer.optimizer_steps, 1)

        cnn_options = resolve_preset({
            'architecture': 'residual_cnn',
            'architecture_options': {
                'channels': 16, 'residual_blocks': 2, 'board_embedding_size': 32,
                'metadata_embedding_size': 8, 'fusion_hidden_size': 24,
            },
        })
        cnn = ARCHITECTURES['residual_cnn'](
            24, 32, 12, 3,
            state_spec=get_observation_spec(cnn_options['state_representation'], 24, 32),
            **cnn_options['architecture_options'],
        )
        output = cnn({
            'board_state': torch.rand(2, 3, 24, 32),
            'metadata': torch.rand(2, 12),
        })
        self.assertEqual(output.shape, (2, 3))


class ConfiguredAgentTests(unittest.TestCase):
    def test_configuration_controls_training_and_round_trips(self):
        from agent import Agent, save_checkpoint_atomic
        from game import SnakeGameAI

        config = resolve_preset({
            'architecture': 'mlp',
            'architecture_options': {'hidden_sizes': [16, 16]},
            'training': {'lr': 0.002, 'gamma': 0.8, 'epsilon_start': 0.6,
                         'epsilon_min': 0.1, 'epsilon_decay': 0.9,
                         'max_memory': 10, 'replay_batch_size': 2,
                         'replay_warmup': 2, 'train_every': 1, 'lr_step_size': 3},
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pth'
            agent = Agent(config=config, device='cpu', checkpoint_path=path,
                          strict_checkpoint=True)
            self.assertEqual(agent.memory.maxlen, 10)
            self.assertEqual(agent.trainer.gamma, 0.8)
            game = SnakeGameAI(render=False)
            for _ in range(4):
                state = agent.get_state(game)
                reward, done, _ = game.play_step([1, 0, 0])
                agent.remember(state, [1, 0, 0], reward, agent.get_state(game), done)
                agent.train_long_memory()
                if done:
                    game.reset()
            agent.n_games = 10
            agent.epsilon_start_game = 8
            agent.record = 5
            agent.best_reward = 12.5
            agent._update_epsilon()
            self.assertAlmostEqual(agent.epsilon, 0.6 * 0.9 ** 2)
            save_checkpoint_atomic(agent, checkpoint_path=path)
            restored = Agent(config=config, device='cpu', checkpoint_path=path,
                             strict_checkpoint=True)
            self.assertEqual(restored.config, config)
            self.assertEqual((restored.n_games, restored.record, restored.best_reward), (10, 5, 12.5))
            self.assertEqual(restored.epsilon, agent.epsilon)
            self.assertEqual(restored.environment_steps, 4)
            self.assertEqual(restored.trainer.optimizer_steps, 3)
            self.assertEqual(restored.trainer.scheduler.state_dict(), agent.trainer.scheduler.state_dict())
            self.assertEqual(restored.trainer.current_lr(), agent.trainer.current_lr())
            self.assertEqual(len(restored.memory), 0)
            for key, value in agent.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[key])

            changed = copy.deepcopy(config)
            changed['training']['gamma'] = 0.7
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'configuration'):
                Agent(config=changed, device='cpu', checkpoint_path=path, strict_checkpoint=True)
            self.assertEqual(path.read_bytes(), original)


class CheckpointPolicyTests(unittest.TestCase):
    def test_train_rejects_config_that_conflicts_with_agent_before_writing(self):
        from agent import Agent, train

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest, best, csv = root / 'latest.pth', root / 'best.pth', root / 'scores.csv'
            agent = Agent(config={'architecture': 'mlp'}, device='cpu', checkpoint_path=latest)
            changed = resolve_preset({
                'architecture': 'mlp', 'rewards': {'food_reward': 3},
            })
            with self.assertRaisesRegex(ValueError, 'differs from the supplied agent config'):
                train(max_games=1, config=changed, agent=agent, checkpoint_path=latest,
                      best_checkpoint_path=best, csv_path=csv)
            self.assertFalse(latest.exists())
            self.assertFalse(best.exists())
            self.assertFalse(csv.exists())

    def test_latest_advances_without_replacing_best(self):
        from agent import Agent, train
        from game import SnakeGameAI

        config = resolve_preset({'architecture': 'mlp', 'architecture_options': {'hidden_sizes': [8]}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest, best, csv = root / 'latest.pth', root / 'best.pth', root / 'scores.csv'
            agent = Agent(config=config, device='cpu', checkpoint_path=latest, strict_checkpoint=True)

            class OneStepGame(SnakeGameAI):
                def play_step(self, action):
                    self.end_reason = 'timeout'
                    return 10.0, True, 1

            with patch('agent.SnakeGameAI', OneStepGame):
                train(max_games=1, agent=agent, checkpoint_path=latest,
                      best_checkpoint_path=best, csv_path=csv)
            best_bytes = best.read_bytes()
            agent.best_reward = 100
            with patch('agent.SnakeGameAI', OneStepGame):
                train(max_games=1, agent=agent, checkpoint_path=latest,
                      best_checkpoint_path=best, csv_path=csv)
            self.assertEqual(best.read_bytes(), best_bytes)
            checkpoint = torch.load(latest, weights_only=True)
            self.assertEqual(checkpoint['metadata']['n_games'], 2)
            self.assertEqual(checkpoint['metadata']['best_reward'], 100)
            self.assertEqual([row.split(',')[0] for row in csv.read_text().splitlines()[1:]], ['1', '2'])

    def test_mid_episode_interrupt_saves_latest_without_logging(self):
        from agent import Agent, train
        from game import SnakeGameAI

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest, best, csv = root / 'latest.pth', root / 'best.pth', root / 'scores.csv'
            agent = Agent(config={'architecture': 'mlp'}, device='cpu', checkpoint_path=latest)

            class InterruptedGame(SnakeGameAI):
                def play_step(self, action):
                    raise KeyboardInterrupt

            with patch('agent.SnakeGameAI', InterruptedGame):
                train(agent=agent, checkpoint_path=latest, best_checkpoint_path=best, csv_path=csv)
            self.assertEqual(torch.load(latest, weights_only=True)['metadata']['n_games'], 0)
            self.assertFalse(csv.exists())
            self.assertFalse(best.exists())


class TrainingCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        source = Path(__file__).resolve().parent
        for name in ('agent.py', 'model.py', 'game.py', 'helper.py', 'configuration.py',
                     'state.py', 'reward_defaults.py', 'train.py'):
            if (source / name).exists():
                shutil.copy2(source / name, self.root / name)
        self.presets = {
            name: {'architecture': name, 'training': {
                'replay_batch_size': 2, 'replay_warmup': 2,
                'train_every': 1, 'max_memory': 32,
            }} for name in ('residual_cnn', 'mlp')
        }
        self.write_presets()
        self.cwd = self.root / 'another-directory'
        self.cwd.mkdir()

    def write_presets(self):
        (self.root / 'models.json').write_text(json.dumps(self.presets))

    def cli(self, *args, success=True, code=None):
        command = ([sys.executable, '-u', '-c', code] if code else
                   [sys.executable, '-u', str(self.root / 'train.py'), *args])
        result = subprocess.run(command, cwd=self.cwd if code is None else self.root,
                                env=os.environ.copy(), capture_output=True, text=True, timeout=45)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def one_step_cli(self, *args):
        import_args = repr(list(args))
        code = f'''
import agent
import train
class OneStepGame(agent.SnakeGameAI):
    def play_step(self, action):
        self.end_reason = 'test'
        return self.rewards['step_penalty'], True, self.score
agent.SnakeGameAI = OneStepGame
train.main({import_args})
'''
        return self.cli(code=code)

    def test_cli_list_and_invalid_input_create_no_outputs(self):
        result = self.cli('--list')
        self.assertIn('residual_cnn', result.stdout)
        self.assertIn('mlp', result.stdout)
        self.assertNotIn('pygame-ce', result.stdout)
        for args in [(), ('--model', 'unknown'), ('--model', 'mlp', '--run', '../outside')]:
            self.cli(*args, success=False)
        self.presets['mlp']['training']['typo'] = 1
        self.write_presets()
        self.assertIn('typo', self.cli('--model', 'mlp', success=False).stderr)
        self.assertFalse((self.root / 'runs').exists())

    def test_real_runs_are_isolated_and_resume_from_latest(self):
        self.cli('--model', 'mlp', '--run', 'first', '--games', '2', '--device', 'cpu')
        first = self.root / 'runs' / 'first'
        before = {p.name: p.read_bytes() for p in first.iterdir()}
        checkpoint = torch.load(first / 'latest.pth', weights_only=True)
        self.assertGreater(checkpoint['training_state']['optimizer_steps'], 0)
        self.assertEqual(checkpoint['metadata']['n_games'], 2)
        self.assertEqual(json.loads((first / 'config.json').read_text()), checkpoint['config'])
        self.cli('--model', 'residual_cnn', '--games', '1', '--device', 'cpu')
        self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()}, before)
        cnn = torch.load(self.root / 'runs' / 'residual_cnn' / 'latest.pth', weights_only=True)
        self.assertGreater(cnn['training_state']['optimizer_steps'], 0)

        self.cli('--model', 'mlp', '--run', 'first', '--games', '1', '--device', 'cpu', '--reset-epsilon')
        resumed = torch.load(first / 'latest.pth', weights_only=True)
        self.assertEqual(resumed['metadata']['n_games'], 3)
        self.assertEqual(resumed['metadata']['epsilon_start_game'], 2)
        self.assertGreater(resumed['training_state']['optimizer_steps'], checkpoint['training_state']['optimizer_steps'])
        with (first / 'training_scores.csv').open() as handle:
            self.assertEqual([int(row['game']) for row in csv.DictReader(handle)], [1, 2, 3])

        before = {p.name: p.read_bytes() for p in first.iterdir()}
        self.presets['mlp']['training']['gamma'] = 0.5
        self.write_presets()
        error = self.cli('--model', 'mlp', '--run', 'first', '--games', '1', success=False)
        self.assertIn('new --run', error.stderr)
        self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()}, before)

        del self.presets['mlp']['training']['gamma']
        self.write_presets()
        incompatible = torch.load(first / 'latest.pth', weights_only=True)
        incompatible['model_state_dict']['layers.0.weight'] = torch.zeros(1, 1)
        torch.save(incompatible, first / 'latest.pth')
        before = {p.name: p.read_bytes() for p in first.iterdir()}
        error = self.cli('--model', 'mlp', '--run', 'first', '--games', '1', success=False)
        self.assertIn('architecture mismatch', error.stderr)
        self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()}, before)

    def test_metadata_only_nondefault_run_resumes_and_rejects_changes(self):
        self.presets['metadata-exp'] = {
            'architecture': 'mlp',
            'state_representation': {'encoder': 'metadata_only'},
            'rewards': {'step_penalty': -0.02, 'food_reward': 6},
            'architecture_options': {'hidden_sizes': [16, 8]},
            'training': {
                'replay_batch_size': 1, 'replay_warmup': 1,
                'train_every': 1, 'max_memory': 32,
            },
        }
        self.write_presets()
        self.one_step_cli('--model', 'metadata-exp', '--games', '1', '--device', 'cpu')
        run = self.root / 'runs' / 'metadata-exp'
        config = json.loads((run / 'config.json').read_text())
        checkpoint = torch.load(run / 'latest.pth', weights_only=True)
        self.assertEqual(config, checkpoint['config'])
        self.assertEqual(config['state_representation']['encoder'], 'metadata_only')
        self.assertEqual(config['rewards']['step_penalty'], -0.02)
        self.assertEqual(checkpoint['model_state_dict']['layers.0.weight'].shape, (16, 12))

        self.one_step_cli('--model', 'metadata-exp', '--games', '1', '--device', 'cpu')
        resumed = torch.load(run / 'latest.pth', weights_only=True)
        self.assertEqual(resumed['metadata']['n_games'], 2)
        before = {path.name: path.read_bytes() for path in run.iterdir()}
        self.presets['metadata-exp']['rewards']['food_reward'] = 7
        self.write_presets()
        error = self.cli('--model', 'metadata-exp', '--games', '1', success=False)
        self.assertIn('new --run', error.stderr)
        self.assertEqual({path.name: path.read_bytes() for path in run.iterdir()}, before)

    def test_older_named_config_and_checkpoint_fill_default_axes_in_memory(self):
        self.one_step_cli('--model', 'mlp', '--run', 'old-schema', '--games', '1', '--device', 'cpu')
        run = self.root / 'runs' / 'old-schema'
        old_config = json.loads((run / 'config.json').read_text())
        old_config.pop('state_representation')
        old_config.pop('rewards')
        (run / 'config.json').write_text(json.dumps(old_config))

        checkpoint = torch.load(run / 'latest.pth', weights_only=True)
        checkpoint['config'].pop('state_representation')
        checkpoint['config'].pop('rewards')
        torch.save(checkpoint, run / 'latest.pth')
        old_snapshot_bytes = (run / 'config.json').read_bytes()

        self.one_step_cli('--model', 'mlp', '--run', 'old-schema', '--games', '1', '--device', 'cpu')
        self.assertEqual((run / 'config.json').read_bytes(), old_snapshot_bytes)
        resumed = torch.load(run / 'latest.pth', weights_only=True)
        self.assertEqual(resumed['metadata']['n_games'], 2)
        self.assertIn('state_representation', resumed['config'])
        self.assertIn('rewards', resumed['config'])

    def test_legacy_import_preserves_sources_and_can_resume(self):
        from agent import Agent, save_checkpoint_atomic

        source = self.root / 'model' / 'model.pth'
        agent = Agent(checkpoint_path=source, device='cpu')
        agent.n_games = 2
        save_checkpoint_atomic(agent, checkpoint_path=source)
        checkpoint = torch.load(source, weights_only=True)
        checkpoint.pop('config')  # The original workflow predates configuration snapshots.
        torch.save(checkpoint, source)
        scores = self.root / 'training_scores.csv'
        scores.write_text('game,score,end_reason\n7,2,timeout\n')
        originals = (source.read_bytes(), scores.read_bytes())

        self.cli('--model', 'mlp', '--run', 'wrong', '--import-legacy', '--device', 'cpu', success=False)
        self.assertFalse((self.root / 'runs' / 'wrong').exists())
        self.presets['custom-state'] = {
            'architecture': 'residual_cnn',
            'state_representation': {
                'encoder': 'board_metadata', 'options': {'body_encoding': 'binary'},
            },
        }
        self.write_presets()
        error = self.cli('--model', 'custom-state', '--run', 'custom-state',
                         '--import-legacy', '--device', 'cpu', success=False)
        self.assertIn('nondefault state or rewards cannot be verified', error.stderr)
        self.assertFalse((self.root / 'runs' / 'custom-state').exists())
        self.presets['custom-rewards'] = {
            'architecture': 'residual_cnn', 'rewards': {'food_reward': 12},
        }
        self.write_presets()
        error = self.cli('--model', 'custom-rewards', '--run', 'custom-rewards',
                         '--import-legacy', '--device', 'cpu', success=False)
        self.assertIn('nondefault state or rewards cannot be verified', error.stderr)
        self.assertFalse((self.root / 'runs' / 'custom-rewards').exists())
        self.cli('--model', 'residual_cnn', '--run', 'legacy', '--import-legacy', '--device', 'cpu')
        run = self.root / 'runs' / 'legacy'
        self.assertEqual((run / 'latest.pth').read_bytes(), originals[0])
        self.assertEqual((run / 'best.pth').read_bytes(), originals[0])
        self.assertEqual((run / 'training_scores.csv').read_bytes(), originals[1])
        self.cli('--model', 'residual_cnn', '--run', 'legacy', '--import-legacy', success=False)
        self.cli('--model', 'residual_cnn', '--run', 'legacy', '--games', '1', '--device', 'cpu')
        with (run / 'training_scores.csv').open() as handle:
            self.assertEqual([int(row['game']) for row in csv.DictReader(handle)], [7, 8])
        self.assertEqual((source.read_bytes(), scores.read_bytes()), originals)

    def test_real_sigterm_saves_partial_progress_without_episode_row(self):
        code = '''
import os
import signal
import game
import train
original = game.SnakeGameAI.play_step
def interrupted_step(self, action):
    result = original(self, action)
    os.kill(os.getpid(), signal.SIGTERM)
    return result
game.SnakeGameAI.play_step = interrupted_step
train.main(['--model', 'mlp', '--run', 'interrupted', '--device', 'cpu'])
'''
        self.cli(code=code)
        run = self.root / 'runs' / 'interrupted'
        checkpoint = torch.load(run / 'latest.pth', weights_only=True)
        self.assertEqual(checkpoint['metadata']['n_games'], 0)
        self.assertEqual(checkpoint['training_state']['environment_steps'], 1)
        self.assertFalse((run / 'training_scores.csv').exists())
        self.assertFalse((run / 'best.pth').exists())


if __name__ == '__main__':
    unittest.main()
