from collections import deque
from time import sleep

import numpy as np
import torch
from unityagents import UnityEnvironment

from agents.dqn_agent import DQNAgent
from utils.config import generate_configuration_qnet_visual
from utils.image import process_state


def test(env, agent, n_ep_train, n_episodes=10, sleep_t=0.0, state_len=2):
    # Get the default brain
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]

    scores = []
    scores_window = deque(maxlen=100)
    state_window = deque(maxlen=state_len)
    for i in range(state_len):
        state_window.append(process_state(np.zeros(env_info.visual_observations[0].squeeze().shape)))

    for i_episode in range(1, n_episodes + 1):
        # Reset the environment and the score
        env_info = env.reset(train_mode=False)[brain_name]
        state_raw = process_state(env_info.visual_observations[0])
        state_window.append(state_raw)
        state = np.vstack([np.expand_dims(np.array(s), 0) for s in state_window])
        score = 0
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            sleep(sleep_t)
            next_state, reward, done = process_state(env_info.visual_observations[0]), env_info.rewards[
                0], env_info.local_done[0]
            state_window.append(next_state)
            state = np.vstack([np.expand_dims(np.array(s), 0) for s in state_window])
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
    config = generate_configuration_qnet_visual(action_size, state_size)
    agent = DQNAgent(config)
    agent.load_weights("./checkpoint.pth")
    print(test(env, agent, 0, n_episodes=100, sleep_t=0))
