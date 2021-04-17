import torch
import torch.nn as nn


class DuelingQNet(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingQNet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        self.state_value = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        x = self.feature_extractor(state)
        a = self.advantage(x)
        v = self.state_value(x).expand(state.size(0), self.action_size)
        return v + a - a.mean(dim=1).unsqueeze(1).expand(state.size(0), self.action_size)
