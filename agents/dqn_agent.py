import random

import torch
from torch import optim
import torch.nn.functional as F
from agents.agent import Agent
from memories.replay_buffer import ReplayBuffer
from models.q_net import QNet
import numpy as np


class DQNAgent(Agent):
    def __init__(self, state_size, action_size, seed, device,
                 buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=1e-3, lr=5e-4, update_every=4, double_qn=True):
        super().__init__(state_size, action_size, seed, device, buffer_size, batch_size, gamma, tau, lr, update_every)
        # Q-Network
        self.q_online = QNet(state_size, action_size, seed).to(device)
        self.q_target: QNet = QNet(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.q_online.parameters(), lr=self.lr)
        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed, self.device)
        self.double_qn = double_qn

    def step(self, state, action, reward, next_state, done):
        # Save an experience in the replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learning phase (every `self.update_every` steps)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

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
            return random.choice(np.arange(self.action_size))

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
                np.arange(self.batch_size), best_action_q].unsqueeze(1)
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
        for param in self.q_online.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        DQNAgent.soft_update(self.q_online, self.q_target, self.tau)

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
