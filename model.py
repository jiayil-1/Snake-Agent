import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime


class ResidualBlock(nn.Module):
    """Two convolutions with an identity shortcut."""

    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.conv2(self.relu(self.conv1(x)))
        return self.relu(x + residual)


class Linear_QNet(nn.Module):
    """DQN: pooled residual CNN with a pooled stem skip + metadata MLP.

    Board input: 3 channels (head, food, head-to-tail body gradient 1 → 0.5),
    values in [0, 1], native grid size.
    CNN: Conv(3→32) → ReLU → 2x MaxPool → 3 residual blocks + pooled skip
         → flatten → Linear → 128
    Metadata: Linear → 64
    Fusion: Linear(192→128) → ReLU → Linear(128→output_size)
    """

    def __init__(self, board_height, board_width, metadata_size, output_size,
                 channels=32, residual_blocks=3, board_embedding_size=128,
                 metadata_embedding_size=64, fusion_hidden_size=128,
                 state_spec=None):
        super().__init__()
        state_spec = state_spec or {
            'has_board': True, 'board_channels': 3,
            'board_height': board_height, 'board_width': board_width,
            'metadata_size': metadata_size,
        }
        if not state_spec['has_board']:
            raise ValueError('residual_cnn requires a state representation with board input')
        if state_spec['board_height'] < 4 or state_spec['board_width'] < 4:
            raise ValueError('residual_cnn requires board dimensions of at least 4 by 4')
        self.board_height = state_spec['board_height']
        self.board_width = state_spec['board_width']
        self.metadata_size = state_spec['metadata_size']
        self.output_size = output_size

        # Keep this module layout stable for existing checkpoint weights.
        cnn_layers = [
            nn.Conv2d(state_spec['board_channels'], channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ]
        cnn_layers.extend(ResidualBlock(channels) for _ in range(residual_blocks))
        self.cnn = nn.Sequential(*cnn_layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_relu = nn.ReLU()

        cnn_height = self.board_height // 4
        cnn_width = self.board_width // 4
        cnn_flat = channels * cnn_height * cnn_width

        # Project pooled board and metadata features before fusion.
        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_flat, board_embedding_size),
            nn.ReLU(),
        )

        self.meta_fc = nn.Sequential(
            nn.Linear(self.metadata_size, metadata_embedding_size),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(board_embedding_size + metadata_embedding_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Linear(fusion_hidden_size, output_size),
        )

    def forward(self, state):
        board_state = state['board_state']
        metadata = state['metadata']

        if board_state.dim() == 3:
            board_state = board_state.unsqueeze(0)
        if metadata.dim() == 1:
            metadata = metadata.unsqueeze(0)

        # Keep an encoder skip at the same spatial size as the residual blocks.
        skip = self.cnn[:2](board_state)
        pooled_skip = self.pool(self.pool(skip))
        cnn_out = self.cnn[2:](pooled_skip)
        cnn_out = self.skip_relu(cnn_out + pooled_skip)
        cnn_out = torch.flatten(cnn_out, 1)
        board_emb = self.cnn_fc(cnn_out)

        meta_emb = self.meta_fc(metadata)

        fused = torch.cat((board_emb, meta_emb), dim=1)
        return self.head(fused)

    def save(self, file_name='model.pth', metadata=None):
        model_folder_path = './model'
        os.makedirs(model_folder_path, exist_ok=True)
        file_name = os.path.join(model_folder_path, file_name)
        default_metadata = {'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        if metadata:
            default_metadata.update(metadata)
        torch.save({'model_state_dict': self.state_dict(), 'metadata': default_metadata}, file_name)


class MLP_QNet(nn.Module):
    """Flatten the board and combine it with metadata before an MLP."""

    def __init__(self, board_height, board_width, metadata_size, output_size,
                 hidden_sizes=(128, 128), state_spec=None):
        super().__init__()
        state_spec = state_spec or {
            'has_board': True, 'board_channels': 3,
            'board_height': board_height, 'board_width': board_width,
            'metadata_size': metadata_size,
        }
        self.has_board = state_spec['has_board']
        self.metadata_size = state_spec['metadata_size']
        board_input_size = (
            state_spec['board_channels'] * state_spec['board_height'] * state_spec['board_width']
            if self.has_board else 0
        )
        sizes = [board_input_size + self.metadata_size, *hidden_sizes]
        layers = []
        for input_size, hidden_size in zip(sizes, sizes[1:]):
            layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        metadata = state['metadata']
        if metadata.dim() == 1:
            metadata = metadata.unsqueeze(0)
        inputs = []
        if self.has_board:
            board = state['board_state']
            if board.dim() == 3:
                board = board.unsqueeze(0)
            inputs.append(board.flatten(1))
        inputs.append(metadata)
        return self.layers(torch.cat(inputs, dim=1))


# Explicit registry: configuration never imports or evaluates arbitrary code.
ARCHITECTURES = {'residual_cnn': Linear_QNet, 'mlp': MLP_QNet}
ARCHITECTURE_DEFAULTS = {
    'residual_cnn': {
        'channels': 32,
        'residual_blocks': 3,
        'board_embedding_size': 128,
        'metadata_embedding_size': 64,
        'fusion_hidden_size': 128,
    },
    'mlp': {'hidden_sizes': [128, 128]},
}
ARCHITECTURE_CAPABILITIES = {
    'residual_cnn': {'requires_board': True},
    'mlp': {'requires_board': False},
}


def resolve_device(device='auto'):
    if device == 'auto':
        device = ('cuda' if torch.cuda.is_available()
                  else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if device not in ('cpu', 'cuda', 'mps'):
        raise ValueError('device must be auto, cpu, cuda, or mps')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu or auto')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable; use --device cpu or auto')
    return torch.device(device)


class QTrainer:
    """Standard DQN trainer with TD learning."""

    def __init__(
        self,
        model,
        lr,
        gamma,
        max_grad_norm=5.0,
        lr_step_size=500,
        lr_decay=0.95,
        min_lr=1e-5,
        device='auto',
        compile_model=True,
    ):
        self.lr = lr
        self.gamma = gamma
        self.min_lr = min_lr
        self.device = resolve_device(device)
        print(f'Using {self.device} to train')
        self.model = model.to(self.device)
        self.max_grad_norm = max_grad_norm
        # One fused kernel replaces several launches per parameter tensor.
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, fused=True)
        self.optimizer.register_load_state_dict_post_hook(self._keep_fused_adam)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            # Replay updates happen every four environment moves by default.
            # 500 optimizer updates retain the prior ~2000-move decay cadence.
            step_size=lr_step_size,
            gamma=lr_decay,
        )
        self.criterion = nn.MSELoss()
        self.optimizer_steps = 0
        # Compiling the loss fuses the forward and backward kernels, which is
        # most of each update on CUDA and MPS. On CPU it measured no faster and
        # adds seconds of warmup. Shapes are static: replay batches never vary.
        self._loss = (torch.compile(self._td_loss, dynamic=False)
                      if compile_model and self.device.type != 'cpu' else self._td_loss)

    def current_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def apply_settings(self, settings):
        """Reapply configured learning-rate settings after restoring saved state.

        Loading optimizer and scheduler state restores the old base rate and
        decay, so recompute the rate this schedule gives at the restored step.
        """
        self.lr = settings['lr']
        self.scheduler.gamma = settings['lr_decay']
        self.scheduler.base_lrs = [settings['lr'] for _ in self.optimizer.param_groups]
        decays = self.scheduler.last_epoch // self.scheduler.step_size
        lr = settings['lr'] * settings['lr_decay'] ** decays
        if self.min_lr is not None:
            lr = max(lr, self.min_lr)
        for group in self.optimizer.param_groups:
            group['initial_lr'] = settings['lr']
            group['lr'] = lr

    @staticmethod
    def _keep_fused_adam(optimizer):
        # Loading replaces the live param-group flags with the saved ones, so
        # checkpoints from the unfused optimizer would otherwise disable fusion.
        for group in optimizer.param_groups:
            group['fused'] = True
            group['foreach'] = None
            for parameter in group['params']:
                state = optimizer.state.get(parameter, {})
                if 'step' in state:
                    state['step'] = state['step'].to(device=parameter.device, dtype=torch.float32)

    @staticmethod
    def build_td_target(predictions, actions, rewards, next_q_values, dones, gamma):
        """Build DQN targets while changing only the action index selected per row."""
        done_mask = dones.to(predictions.dtype)
        max_next_q = torch.max(next_q_values, dim=1).values
        target_q = rewards + (1.0 - done_mask) * gamma * max_next_q

        target = predictions.detach().clone()
        target.scatter_(1, actions.unsqueeze(1), target_q.unsqueeze(1))
        return target

    def _td_loss(self, state, action, reward, next_state, done):
        # Current Q-values
        pred = self.model(state)

        # TD target: r + γ·max Q(s', a')
        with torch.no_grad():
            next_q = self.model(next_state)
            target = self.build_td_target(
                pred, action, reward, next_q, done, self.gamma,
            )
        return self.criterion(pred, target)

    def _to_device(self, values, dtype=np.float32):
        return torch.from_numpy(np.asarray(values, dtype=dtype)).to(self.device)

    def train_step(self, state, action, reward, next_state, done):
        """Train one transition or a sequence of transitions with one-hot actions."""
        if isinstance(state, dict):
            state, action, reward, next_state, done = (
                [state], [action], [reward], [next_state], [done],
            )

        def stack(states):
            return {key: self._to_device(np.stack([s[key] for s in states])) for key in states[0]}

        self.train_batch(
            stack(state), self._to_device(np.argmax(np.asarray(action), axis=1), np.int64),
            self._to_device(reward), stack(next_state), self._to_device(done),
        )

    def train_batch(self, state, action, reward, next_state, done):
        """Standard TD learning: Q(s,a) ← r + γ·max_a' Q(s',a')

        Inputs are batched tensors on the training device; actions are indices.
        """
        self.optimizer.zero_grad()
        loss = self._loss(state, action, reward, next_state, done)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer_steps += 1

        if self.min_lr is not None:
            for group in self.optimizer.param_groups:
                if group['lr'] < self.min_lr:
                    group['lr'] = self.min_lr
