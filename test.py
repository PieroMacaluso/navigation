from collections import deque
from time import sleep

import numpy as np
import torch
from unityagents import UnityEnvironment

from agents.dqn_agent import DQNAgent


def test(env, agent, n_ep_train, n_episodes=10, sleep_t=0.0):
    # Get the default brain
    brain_name = env.brain_names[0]

    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        # Reset the environment and the score
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            sleep(sleep_t)
            next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        print('\rTest Episode {}\tLast Score: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score,
                                                                                    np.mean(scores_window)),
              end="")
    print('\rTest after {} episode mean {:.2f}'.format(n_ep_train, np.mean(scores_window)))
    return np.mean(scores_window)


if __name__ == '__main__':
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])
    seed = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(state_size, action_size, seed, device)
    agent.load_weights("./checkpoint.pth")
    print(test(env, agent, 0, n_episodes=100, sleep_t=0))
