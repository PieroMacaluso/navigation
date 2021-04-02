import os
from datetime import datetime

from agents.dqn_agent import DQNAgent
from models.dueling_q_net import DuelingQNet


class DuelingDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, seed, device,
                 buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=1e-3, lr=5e-4, update_every=4, double_qn=True):
        super().__init__(state_size, action_size, seed, device, buffer_size, batch_size, gamma, tau, lr, update_every,
                         double_qn)
        # Q-Network
        self.qnetwork_local = DuelingQNet(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNet(state_size, action_size, seed).to(device)
        self.checkpoint_dir = f"./checkpoints/Dueling{'Double' if self.double_qn else ''}DQNet_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"

    def create_dirs(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
