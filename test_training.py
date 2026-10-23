import contextlib
import copy
import csv
import io
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
from state import encode_state, get_observation_spec, metadata_features


class ConfigurationTests(unittest.TestCase):
    def test_defaults_and_json_presets(self):
        presets = load_presets(Path(__file__).with_name('models.json'))
        self.assertEqual(set(presets), {'residual_cnn', 'raw_skip_cnn', 'mlp'})
        cnn = presets['residual_cnn']
        self.assertEqual(cnn['training']['replay_batch_size'], 64)
        self.assertEqual(cnn['training']['replay_warmup'], 1000)
        self.assertEqual(cnn['training']['train_every'], 4)
        self.assertEqual(cnn['training']['lr_step_size'], 500)
        self.assertEqual(presets['mlp']['architecture_options']['hidden_sizes'], [128, 128])
        self.assertEqual(cnn['state_representation'], {
            'encoder': 'board_metadata',
            'options': {'body_encoding': 'gradient', 'food_distance': True},
        })
        self.assertEqual(cnn['rewards']['food_reward'], 10)
        self.assertEqual(cnn['architecture_options']['channels'], 32)
        self.assertEqual(resolve_preset(cnn), cnn)
        raw_skip = presets['raw_skip_cnn']
        self.assertEqual(raw_skip['architecture_options']['channels'], 6)
        self.assertFalse(raw_skip['state_representation']['options']['food_distance'])
        self.assertEqual(resolve_preset(raw_skip), raw_skip)

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
            {'architecture': 'mlp', 'state_representation': {
                'encoder': 'board_metadata', 'options': {'food_distance': 0},
            }},
            {'architecture': 'mlp', 'state_representation': {
                'encoder': 'metadata_only', 'options': {'body_encoding': 'binary'},
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

        for encoder in ('board_metadata', 'metadata_only'):
            with self.subTest(encoder=encoder):
                representation = resolve_preset({
                    'architecture': 'mlp',
                    'state_representation': {'encoder': encoder, 'options': {'food_distance': False}},
                })['state_representation']
                without_distance = encode_state(representation, game, 32, 24)
                np.testing.assert_array_equal(without_distance['metadata'], gradient['metadata'][:-1])
                self.assertEqual(get_observation_spec(representation, 24, 32)['metadata_size'], 11)
                self.assertNotIn('food dist', metadata_features(representation))
                self.assertEqual(len(metadata_features(representation)), 11)

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

    @unittest.skipUnless(torch.cuda.is_available() or torch.backends.mps.is_available(),
                         'torch.compile is only used on CUDA or MPS')
    def test_compiled_loss_matches_eager_updates(self):
        device = 'cuda' if torch.cuda.is_available() else 'mps'
        config = resolve_preset({'architecture': 'raw_skip_cnn'})
        spec = get_observation_spec(config['state_representation'], 24, 32)
        eager_model = ARCHITECTURES['raw_skip_cnn'](
            24, 32, 12, 3, state_spec=spec, **config['architecture_options'],
        )
        compiled_model = copy.deepcopy(eager_model)
        eager = QTrainer(eager_model, lr=0.001, gamma=0.99, device=device, compile_model=False)
        compiled = QTrainer(compiled_model, lr=0.001, gamma=0.99, device=device)
        self.assertNotEqual(compiled._loss, compiled._td_loss)
        generator = torch.Generator().manual_seed(0)
        for _ in range(3):
            batch = (
                {'board_state': torch.rand(8, 3, 24, 32, generator=generator),
                 'metadata': torch.rand(8, 12, generator=generator)},
                torch.randint(0, 3, (8,), generator=generator),
                torch.rand(8, generator=generator),
                {'board_state': torch.rand(8, 3, 24, 32, generator=generator),
                 'metadata': torch.rand(8, 12, generator=generator)},
                (torch.rand(8, generator=generator) < 0.25).float(),
            )
            on_device = [{key: value.to(device) for key, value in part.items()}
                         if isinstance(part, dict) else part.to(device) for part in batch]
            eager.train_batch(*on_device)
            compiled.train_batch(*on_device)
        for (name, expected), actual in zip(eager_model.state_dict().items(),
                                            compiled_model.state_dict().values()):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5, msg=name)

    def test_cpu_trainer_stays_eager(self):
        model = ARCHITECTURES['mlp'](24, 32, 12, 3, hidden_sizes=[8])
        trainer = QTrainer(model, lr=0.001, gamma=0.99, device='cpu')
        self.assertEqual(trainer._loss, trainer._td_loss)


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
            self.assertEqual(agent.memory.capacity, 10)
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

            # Tuned hyperparameters apply on resume, continuing the decay
            # schedule from the restored step (one decay after three updates).
            tuned = copy.deepcopy(config)
            tuned['training'].update({'gamma': 0.7, 'lr': 0.004, 'lr_decay': 0.5})
            resumed = Agent(config=tuned, device='cpu', checkpoint_path=path,
                            strict_checkpoint=True)
            self.assertEqual(resumed.trainer.gamma, 0.7)
            self.assertEqual(resumed.trainer.optimizer_steps, 3)
            self.assertAlmostEqual(resumed.trainer.current_lr(), 0.004 * 0.5)
            self.assertEqual(resumed.trainer.scheduler.gamma, 0.5)

            changed = copy.deepcopy(config)
            changed['architecture_options']['hidden_sizes'] = [8]
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'architecture_options'):
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

    def test_live_mean_continues_from_logged_history(self):
        from agent import Agent, read_recent_scores, train
        from game import SnakeGameAI

        class ScoreTenGame(SnakeGameAI):
            def play_step(self, action):
                self.end_reason = 'timeout'
                return 1.0, True, 10

        config = resolve_preset({'architecture': 'mlp', 'training': {'parallel_games': 1}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest, best, csv_path = root / 'latest.pth', root / 'best.pth', root / 'scores.csv'
            self.assertEqual(read_recent_scores(csv_path, 100), [])
            # 150 logged games: the last 100 scores are 50..149.
            csv_path.write_text('game,score,end_reason\n' + ''.join(
                f'{n + 1},{n},timeout\n' for n in range(150)))
            self.assertEqual(read_recent_scores(csv_path, 100), [float(n) for n in range(50, 150)])
            agent = Agent(config=config, device='cpu', checkpoint_path=latest, strict_checkpoint=True)
            with (patch('agent.SnakeGameAI', ScoreTenGame), patch('agent.HIDE_PLOT', False),
                  patch('agent.plot') as plot):
                train(max_games=1, agent=agent, checkpoint_path=latest,
                      best_checkpoint_path=best, csv_path=csv_path)
            _, means = plot.call_args.args
            self.assertEqual(means, [(sum(range(51, 150)) + 10) / 100])

    def test_parallel_slots_restart_until_the_run_limit(self):
        from agent import Agent, train
        from game import SnakeGameAI

        config = resolve_preset({'architecture': 'mlp', 'training': {
            'parallel_games': 3, 'replay_batch_size': 2, 'replay_warmup': 2,
            'train_every': 2, 'max_memory': 32,
        }})
        # Games end after these step counts, in the order they start. A sixth
        # start would exhaust the iterator and fail the run.
        lengths = iter([3, 1, 2, 4, 1])

        class CountdownGame(SnakeGameAI):
            def reset(self):
                super().reset()
                self.length = self.remaining = next(lengths)

            def play_step(self, action):
                self.remaining -= 1
                self.end_reason = 'timeout'
                return 1.0, self.remaining == 0, self.length

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest, best, csv_path = root / 'latest.pth', root / 'best.pth', root / 'scores.csv'
            agent = Agent(config=config, device='cpu', checkpoint_path=latest, strict_checkpoint=True)
            with patch('agent.SnakeGameAI', CountdownGame):
                train(max_games=5, agent=agent, checkpoint_path=latest,
                      best_checkpoint_path=best, csv_path=csv_path)
            self.assertIsNone(next(lengths, None))
            with csv_path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row['game']) for row in rows], [1, 2, 3, 4, 5])
            # Scores are game lengths, logged in completion order across slots.
            self.assertEqual([int(row['score']) for row in rows], [1, 2, 3, 1, 4])
            self.assertEqual({row['start_score'] for row in rows}, {'0'})
            checkpoint = torch.load(latest, weights_only=True)
            self.assertEqual(checkpoint['metadata']['n_games'], 5)
            self.assertEqual(checkpoint['metadata']['best_reward'], 4.0)
            self.assertEqual(checkpoint['training_state']['environment_steps'], 11)
            # Warmup is met at step 2; updates follow at every second step after it.
            self.assertEqual(checkpoint['training_state']['optimizer_steps'], 5)
            self.assertEqual(torch.load(best, weights_only=True)['metadata']['best_reward'], 4.0)

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


class ScoreLogTests(unittest.TestCase):
    def test_older_logs_gain_blank_columns_and_plot_gained_scores(self):
        from helper import append_score, ensure_score_columns
        import plot_training_scores

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oldest, older = root / 'oldest.csv', root / 'older.csv'
            oldest.write_text('game,score\n1,2\n')
            older.write_text('game,score,end_reason\n1,2,timeout\n')
            for path in (oldest, older):
                ensure_score_columns(path)
                append_score(path, 2, 30, 'self-collision', 25)
            self.assertEqual(oldest.read_text().splitlines(),
                             ['game,score,end_reason,start_score', '1,2,,', '2,30,self-collision,25'])
            self.assertEqual(older.read_text().splitlines(),
                             ['game,score,end_reason,start_score', '1,2,timeout,', '2,30,self-collision,25'])

            games, scores, starts = plot_training_scores.load_training_data(older)
            self.assertEqual((games, scores, starts), ([1, 2], [2.0, 30.0], [None, 25.0]))
            out = root / 'plot.png'
            with patch.object(plot_training_scores, 'plot_series',
                              wraps=plot_training_scores.plot_series) as plot_series:
                plot_training_scores.plot_training_data(games, scores, starts, out)
            # Only the game with a logged start contributes to the gained panel.
            self.assertEqual(plot_series.call_args_list[1].args[1:], ((2,), (5.0,), 'Score Gained'))
            self.assertTrue(out.exists())


class WeightPlotTests(unittest.TestCase):
    def test_every_architecture_plots_and_maps_its_input_columns(self):
        import visualize_weights

        cases = [
            ({'architecture': 'raw_skip_cnn'}, (3, 6, 8)),
            ({'architecture': 'raw_skip_cnn', 'state_representation': {
                'encoder': 'board_metadata', 'options': {'food_distance': False},
            }}, (3, 6, 8)),
            ({'architecture': 'residual_cnn'}, (32, 6, 8)),
            ({'architecture': 'mlp'}, (3, 24, 32)),
            ({'architecture': 'mlp', 'state_representation': {'encoder': 'metadata_only'}}, None),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for preset, board_shape in cases:
                with self.subTest(preset=preset):
                    config = resolve_preset(preset)
                    spec = get_observation_spec(config['state_representation'], 24, 32)
                    model = ARCHITECTURES[config['architecture']](
                        24, 32, 12, 3, state_spec=spec, **config['architecture_options'])
                    board_layer, shape, metadata_layer, columns = visualize_weights.input_layers(
                        model, 24, 32)
                    self.assertEqual(shape, board_shape)
                    self.assertEqual(board_layer is None, board_shape is None)
                    self.assertEqual(metadata_layer.weight[:, columns].shape[1],
                                     spec['metadata_size'])

                    checkpoint = Path(directory) / f'{len(os.listdir(directory))}.pth'
                    torch.save({'model_state_dict': model.state_dict(), 'config': config}, checkpoint)
                    with contextlib.redirect_stdout(io.StringIO()):
                        visualize_weights.main(['--checkpoint', str(checkpoint)])
                    self.assertTrue(checkpoint.with_name(f'{checkpoint.stem}_weights.png').exists())


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
                'train_every': 1, 'max_memory': 32, 'parallel_games': 2,
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
        for args in [(), ('--model', 'unknown'), ('--model', 'mlp', '--run', '../outside'),
                     ('--model', 'mlp', '--start-length', '0'),
                     ('--model', 'mlp', '--start-length', '385'),
                     ('--model', 'mlp', '--start-length', '30-4'),
                     ('--model', 'mlp', '--start-length', '4-385'),
                     ('--model', 'mlp', '--start-length', '4-'),
                     ('--model', 'mlp', '--epsilon-ramp', '5'),
                     ('--model', 'mlp', '--epsilon-ramp', '5-5')]:
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

        # Training settings may change between invocations; the run resumes
        # with the new values and records them.
        self.presets['mlp']['training']['gamma'] = 0.5
        self.write_presets()
        result = self.cli('--model', 'mlp', '--run', 'first', '--games', '1', '--device', 'cpu')
        self.assertIn('training.gamma: 0.99 -> 0.5', result.stdout)
        tuned = torch.load(first / 'latest.pth', weights_only=True)
        self.assertEqual(tuned['metadata']['n_games'], 4)
        self.assertEqual(tuned['config']['training']['gamma'], 0.5)
        self.assertEqual(json.loads((first / 'config.json').read_text()), tuned['config'])

        # Network-shape changes cannot reuse the checkpoint.
        before = {p.name: p.read_bytes() for p in first.iterdir()}
        self.presets['mlp']['architecture_options'] = {'hidden_sizes': [16]}
        self.write_presets()
        error = self.cli('--model', 'mlp', '--run', 'first', '--games', '1', success=False)
        self.assertIn('architecture_options', error.stderr)
        self.assertIn('new --run', error.stderr)
        self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()}, before)

        del self.presets['mlp']['architecture_options']
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

        # --epsilon-ramp is per invocation, so it leaves the run configuration unchanged.
        self.one_step_cli('--model', 'metadata-exp', '--games', '1', '--device', 'cpu',
                          '--epsilon-ramp', '0-5')
        self.assertEqual(json.loads((run / 'config.json').read_text()), config)
        resumed = torch.load(run / 'latest.pth', weights_only=True)
        self.assertEqual(resumed['metadata']['n_games'], 2)
        # Changed rewards and hyperparameters resume the same run.
        self.presets['metadata-exp']['rewards']['food_reward'] = 7
        self.presets['metadata-exp']['training'].update({'replay_batch_size': 2, 'lr': 0.0005})
        self.write_presets()
        self.one_step_cli('--model', 'metadata-exp', '--games', '1', '--device', 'cpu')
        self.assertEqual(sorted(path.name for path in (self.root / 'runs').iterdir()), ['metadata-exp'])
        tuned = torch.load(run / 'latest.pth', weights_only=True)
        self.assertEqual(tuned['metadata']['n_games'], 3)
        self.assertEqual(tuned['config']['rewards']['food_reward'], 7)
        self.assertEqual(tuned['config']['training']['replay_batch_size'], 2)
        self.assertEqual(tuned['optimizer_state_dict']['param_groups'][0]['initial_lr'], 0.0005)
        self.assertEqual(json.loads((run / 'config.json').read_text()), tuned['config'])

        self.presets['metadata-exp']['state_representation'] = {'encoder': 'board_metadata'}
        self.presets['metadata-exp']['architecture'] = 'residual_cnn'
        del self.presets['metadata-exp']['architecture_options']
        self.write_presets()
        before = {path.name: path.read_bytes() for path in run.iterdir()}
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
        self.assertIn('nondefault state cannot be verified', error.stderr)
        self.assertFalse((self.root / 'runs' / 'custom-state').exists())
        self.presets['custom-rewards'] = {
            'architecture': 'residual_cnn', 'rewards': {'food_reward': 12},
        }
        self.write_presets()
        # Rewards are tunable, so a legacy checkpoint can resume under new ones.
        self.cli('--model', 'custom-rewards', '--run', 'custom-rewards',
                 '--import-legacy', '--device', 'cpu')
        self.assertTrue((self.root / 'runs' / 'custom-rewards' / 'latest.pth').exists())
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
        # The signal lands in the first game; the second still finishes its step.
        self.assertEqual(checkpoint['training_state']['environment_steps'], 2)
        self.assertFalse((run / 'training_scores.csv').exists())
        self.assertFalse((run / 'best.pth').exists())


if __name__ == '__main__':
    unittest.main()
