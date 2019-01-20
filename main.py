import gym
import cv2

import torch
from util import NormalizedActions
from collections import deque
from agent import Agent
import numpy as np

from options import options

options = options()

opts = options.parse()


#env = gym.make('CarRacing-v0')

env = gym.make('CarRacing-v0')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#action_space
print("Maximum Values for Actions: ", env.action_space.high)
print("Minimum Values for Actions: ", env.action_space.low)

#observation_space
print("Observation Space: ", env.observation_space)


from IPython.display import clear_output
import matplotlib.pyplot as plt

policy = Agent(env)


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('Episode %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

rewards     = []
ImageStack = 3

def resize_rollaxis_state(state):

    state = cv2.resize(state, (state.shape[0] // 2, state.shape[1] // 2))
    return np.rollaxis(state, 2, 0)

def image_stack(memory):

    state = np.vstack((memory))
    state = torch.from_numpy(state).float().to(device)

    return state

for eps in range(opts.num_episodes):
    scores_deque = deque(maxlen=100)
    short_term_memory = deque(maxlen=ImageStack)
    scores = []
    state = env.reset()
    state = resize_rollaxis_state(state)

    for i in range(ImageStack + 1):
        short_term_memory.append(state)

    episode_reward = 0

    for step in range(opts.max_steps):

        state = image_stack(short_term_memory)

        if eps % 100 == 0:
            action = policy.act(state.unsqueeze(0), step, False)
            action[0] = np.clip(action[0], -1, 1)
            action[1:2] = np.clip(action[1:2], 0, 1)
        else:
            action = policy.act(state.unsqueeze(0), step)
            action[0] = np.clip(action[0], -1, 1)
            action[1:2] = np.clip(action[1:2], 0, 1)

        next_state, reward, done, _ = env.step(action)
        next_state = resize_rollaxis_state(next_state)

        short_term_memory.append(next_state)
        next_state = image_stack(short_term_memory)

        policy.add_to_memory(state, action, reward, next_state, done)

        if policy.memory.__len__() > opts.batch:
            policy.update(step)

        state = next_state
        episode_reward += reward
        opts.frame_idx += 1

        if done:
            break

    scores_deque.append(episode_reward)
    scores.append(episode_reward)

    if eps % opts.print_every == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(eps, np.mean(scores_deque)))

    if np.mean(scores_deque) >= opts.threshold:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(eps - 100,
                                                                                     np.mean(scores_deque)))

        break

    rewards.append(episode_reward)