from collections import deque

import torch
from unityagents import UnityEnvironment
import numpy as np

from agents.dqn_agent import DQNAgent

if __name__ == '__main__':
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    n_episodes = 2000
    eps_start = 1.0
    eps_decay = 0.995
    eps_min = 0.01
    seed = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    scores = []
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
        eps = max(eps_min, eps_decay * eps)
        print('\rEpisode {}\tEps {:.2f}\tLast Score: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, eps, score,
                                                                                           np.mean(scores_window)),
              end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    env.close()
