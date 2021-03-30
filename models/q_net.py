import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNet, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.lin1 = nn.Linear(state_size, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
