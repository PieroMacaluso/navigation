import random

import torch

from memories.replay_buffer import ReplayBuffer
import numpy as np


class Agent:
    def __init__(self, state_size, action_size, seed, device,
                 buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=1e-3, lr=5e-4, update_every=4):
        # Env Data
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.t_step = 0

        # Seed for reproducibility of results
        self.seed = seed
        random.seed(self.seed)

        # Hyper-parameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every

    def step(self, state, action, reward, next_state, done):
        """
        Make a step in the environment

        Parameters
        ----------
        state
        action
        reward
        next_state
        done

        """
        pass

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
        pass

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Parameters
        ----------
        experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor

        """
        pass
