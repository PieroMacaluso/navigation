from collections import deque

import torch
from torch.autograd import Variable
from torchvision.transforms import Resize, ToTensor
from unityagents import UnityEnvironment
import numpy as np

from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt

from test_visual import test
import torch.nn.functional as F

from utils.config import generate_configuration_qnet_visual
from utils.image import process_state

if __name__ == '__main__':
    env = UnityEnvironment(file_name="./VisualBanana_Linux/Banana.x86_64")
    state_len = 2
    min_solved = 13.0
    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = (state_len,) + process_state(np.zeros(env_info.visual_observations[0].squeeze().shape)).shape

    scores = []
    test_scores = []
    test_scores_i = []
    avg_scores = []
    scores_window = deque(maxlen=100)
    config = generate_configuration_qnet_visual(action_size, state_size)
    agent = DQNAgent(config)
    agent.create_dirs()
    state_window = deque(maxlen=state_len)
    for i in range(state_len):
        state_window.append(process_state(np.zeros(env_info.visual_observations[0].squeeze().shape)))
    eps = config.eps_start

    for i_episode in range(1, config.n_episodes + 1):
        # Reset the environment and the score
        env_info = env.reset(train_mode=True)[brain_name]
        state_raw = process_state(env_info.visual_observations[0])
        state_window.append(state_raw)
        state = np.vstack([np.expand_dims(np.array(s), 0) for s in state_window])
        score = 0
        while True:
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state_raw, reward, done = process_state(env_info.visual_observations[0]), env_info.rewards[
                0], \
                                           env_info.local_done[0]
            state_window.append(next_state_raw)
            next_state = np.vstack([np.expand_dims(np.array(s), 0) for s in state_window])
            agent.step(np.array([state]), action, reward, np.array([next_state]), done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        avg_scores.append(np.mean(scores_window))
        eps = max(config.eps_min, config.eps_decay * eps)
        print('\rEpisode {}\tEps {:.2f}\tLast Score: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, eps, score,
                                                                                           np.mean(scores_window)),
              end="")
        if i_episode % 100 == 0:
            test_scores.append(test(env, agent, i_episode, state_len=state_len))
            test_scores_i.append(i_episode)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(range(len(scores)), scores, label="Score")
            ax1.plot(range(len(avg_scores)), avg_scores, label="Avg Score")
            ax1.plot(test_scores_i, test_scores, label="Test Score")
            plt.legend()
            plt.savefig(agent.checkpoint_dir + f'plot_{i_episode}.png', dpi=300)
            plt.show()
            agent.save_weights(i_episode)
        if np.mean(scores_window) >= min_solved:
            min_solved = np.mean(scores_window)
            print('\nNew best in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                               np.mean(scores_window)))
            agent.save_weights(i_episode)

    env.close()
