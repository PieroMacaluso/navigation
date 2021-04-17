import os
import random
from datetime import datetime

import torch
from torch import optim
import torch.nn.functional as F
from memories.replay_buffer import ReplayBuffer
from models.q_cnn import QCNN
from models.q_net import QNet
import numpy as np


class DQNAgent:
    def __init__(self, config):
        # Env Data
        self.config = config
        self.device = config.device()
        self.t_step = 0

        # Seed for reproducibility of results
        self.seed = config.seed
        random.seed(self.seed)

        # Q-Network
        self.q_online = config.q_online(self.config.state_size, self.config.action_size, self.config.seed).to(self.device)
        self.q_target = config.q_target(self.config.state_size, self.config.action_size, self.config.seed).to(self.device)
        self.optimizer = config.optimizer(self.q_online.parameters(), lr=self.config.lr)
        # Replay memory
        self.memory = config.memory()
        self.double_qn = config.double_qn
        self.checkpoint_dir = config.checkpoint_dir

    def create_dirs(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def step(self, state, action, reward, next_state, done):
        # Save an experience in the replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learning phase (every `self.update_every` steps)
        self.t_step = (self.t_step + 1) % self.config.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.config.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.config.gamma)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.

        Parameters
        ----------
        state (array_like): current state
        eps (float): epsilon, for epsilon-greedy action selection

        Returns
        -------
        action (array_like): action for given state as per current policy

        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_online.eval()
        with torch.no_grad():
            action_values = self.q_online(state)
        self.q_online.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.config.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Parameters
        ----------
        experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor

        """
        states, actions, rewards, next_states, dones = experiences
        if self.double_qn:
            best_action_q = self.q_online(next_states).detach().max(1)[1]
            targets_next = self.q_target(next_states).detach()[
                np.arange(self.config.batch_size), best_action_q].unsqueeze(1)
        else:
            targets_next = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)
        targets = rewards + (gamma * targets_next * (1 - dones))
        expected = self.q_online(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(expected, targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        # for param in self.q_online.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        DQNAgent.soft_update(self.q_online, self.q_target, self.config.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def load_weights(self, pth_path):
        self.q_online.load_state_dict(torch.load(pth_path))
        self.q_target.load_state_dict(torch.load(pth_path))

    def save_weights(self, i_episode):
        torch.save(self.q_online.state_dict(), self.checkpoint_dir + f'checkpoint_{i_episode}.pth')
