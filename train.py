from collections import deque

import torch
from unityagents import UnityEnvironment
import numpy as np

from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt

from test import test

if __name__ == '__main__':
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    n_episodes = 2000
    eps_start = 1.0
    eps_decay = 0.99
    eps_min = 0.01
    seed = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_solved = 13.0
    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    scores = []
    test_scores = []
    test_scores_i = []
    avg_scores = []
    scores_window = deque(maxlen=100)
    agent = DQNAgent(state_size, action_size, seed, device)

    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        # Reset the environment and the score
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        avg_scores.append(np.mean(scores_window))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(range(len(scores)), scores, label="Score")
        ax1.plot(range(len(avg_scores)), avg_scores, label="Avg Score")
        ax1.plot(test_scores_i, test_scores, label="Test Score")
        plt.legend()
        plt.show()
        eps = max(eps_min, eps_decay * eps)
        print('\rEpisode {}\tEps {:.2f}\tLast Score: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, eps, score,
                                                                                           np.mean(scores_window)),
              end="")
        if i_episode % 100 == 0:
            test_scores.append(test(env, agent, i_episode))
            test_scores_i.append(i_episode)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= min_solved:
            min_solved = np.mean(scores_window)
            print('\nNew best in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent.q_online.state_dict(), f'./checkpoints/checkpoint_{i_episode}.pth')

    env.close()
