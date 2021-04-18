import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQCNN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingQCNN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(state_size[0], 16, (4, 4), (2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (3, 3), (2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.advantage = nn.Sequential(
            nn.Linear(6400, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )

        self.state_value = nn.Sequential(
            nn.Linear(6400, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        x = self.feature_extractor(state)
        x = x.view(x.size(0), -1)
        a = self.advantage(x)
        v = self.state_value(x).expand(state.size(0), self.action_size)
        return v + a - a.mean(dim=1).unsqueeze(1).expand(state.size(0), self.action_size)
