import torch
import torch.nn as nn
import torch.nn.functional as F


class QCNN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QCNN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.seed = torch.manual_seed(seed)
        self.feature_extractor = nn.Sequential(
            # 2, 32, 32, 3
            nn.Conv3d(state_size[0], 10, (5, 5, 1), (1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1)),
            # 10, 13, 13, 3
            nn.Conv3d(10, 32, (5, 5, 1), (1, 1, 1)),
            nn.ReLU(),
            # 32, 9, 9, 3
            nn.MaxPool3d((3, 3, 1), stride=(3, 3, 1)),
            # 16, 3, 3, 3
        )
        self.lin1 = nn.Linear(864, action_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        return x
