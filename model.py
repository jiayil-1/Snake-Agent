import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime


class Linear_QNet(nn.Module):
    """Simple DQN: 2-layer CNN + metadata MLP → 2 fusion linear layers → Q-values.

    Board input: 3 channels (head, food, body), binary 0/1, native grid size.
    CNN: Conv(3→16) → ReLU → MaxPool → Conv(16→32) → ReLU → MaxPool → flatten → Linear → 128
    Metadata: Linear → 64
    Fusion: Linear(192→128) → ReLU → Linear(128→output_size)
    """

    def __init__(self, board_height, board_width, metadata_size, output_size):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        self.metadata_size = metadata_size
        self.output_size = output_size

        # -- CNN: 2 conv layers with 16 and 32 output channels -------------
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → H/2 × W/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → H/4 × W/4
        )

        cnn_h = board_height // 4
        cnn_w = board_width // 4
        cnn_flat = 32 * max(cnn_h, 1) * max(cnn_w, 1)

        # Downsample to 128 features
        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_flat, 128),
            nn.ReLU(),
        )

        # -- Metadata branch → 64 features --------------------------------
        self.meta_fc = nn.Sequential(
            nn.Linear(metadata_size, 64),
            nn.ReLU(),
        )

        # -- Fusion: two linear layers → Q-values -------------------------
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, state):
        board_state = state['board_state']
        metadata = state['metadata']

        if board_state.dim() == 3:
            board_state = board_state.unsqueeze(0)
        if metadata.dim() == 1:
            metadata = metadata.unsqueeze(0)

        cnn_out = self.cnn(board_state)
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


class QTrainer:
    """Standard DQN trainer with TD learning."""

    def __init__(
        self,
        model,
        lr,
        gamma,
        max_grad_norm=5.0,
        lr_step_size=2000,
        lr_decay=0.95,
        min_lr=1e-5,
    ):
        self.lr = lr
        self.gamma = gamma
        self.min_lr = min_lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using {self.device} to train')
        self.model = model.to(self.device)
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_step_size,
            gamma=lr_decay,
        )
        self.criterion = nn.MSELoss()

    def current_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def _to_state_tensor(self, state):
        if isinstance(state, dict):
            return {
                'metadata': torch.tensor(np.asarray(state['metadata'], dtype=np.float32),
                                         dtype=torch.float).unsqueeze(0).to(self.device),
                'board_state': torch.tensor(np.asarray(state['board_state'], dtype=np.float32),
                                            dtype=torch.float).unsqueeze(0).to(self.device),
            }
        return {
            'metadata': torch.tensor(np.asarray([s['metadata'] for s in state], dtype=np.float32),
                                     dtype=torch.float).to(self.device),
            'board_state': torch.tensor(np.asarray([s['board_state'] for s in state], dtype=np.float32),
                                        dtype=torch.float).to(self.device),
        }

    def train_step(self, state, action, reward, next_state, done):
        """Standard TD learning: Q(s,a) ← r + γ·max_a' Q(s',a')"""
        state = self._to_state_tensor(state)
        next_state = self._to_state_tensor(next_state)
        action = torch.tensor(np.asarray(action), dtype=torch.long).to(self.device)
        reward = torch.tensor(np.asarray(reward), dtype=torch.float).to(self.device)

        if action.dim() == 1:
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # Current Q-values
        pred = self.model(state)

        # TD target: r + γ·max Q(s', a')
        with torch.no_grad():
            next_q = self.model(next_state)
            max_next_q = torch.max(next_q, dim=1).values
            done_mask = torch.tensor(done, dtype=torch.float, device=self.device)
            target_q = reward + (1.0 - done_mask) * self.gamma * max_next_q

        # Only update the Q-value for the action taken
        target = pred.clone().detach()
        action_indices = torch.argmax(action, dim=1)
        for idx in range(len(done)):
            target[idx][action_indices[idx]] = target_q[idx]

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        if self.min_lr is not None:
            for group in self.optimizer.param_groups:
                if group['lr'] < self.min_lr:
                    group['lr'] = self.min_lr
