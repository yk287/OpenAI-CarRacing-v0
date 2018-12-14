import gym
import cv2

env = gym.make('CarRacing-v0')

state = env.reset()
print(state.shape)

#reduce the size of image to half
state = cv2.resize(state, (state.shape[0] // 2, state.shape[1] //2))
print(state.shape)

env.render()

env.step([.5,.5,.5])
env.render()



import gym
import torch
from util import NormalizedActions
from collections import deque
from agent import Agent
import numpy as np

#env = gym.make('CarRacing-v0')

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

num_episodes  = 5000
max_steps   = 1000
frame_idx   = 0
rewards     = []
batch_size  = 128
PRINT_EVERY = 5

for eps in range(num_episodes):
    scores_deque = deque(maxlen=100)
    scores = []
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):

        if eps % 100 == 0:
            action = policy.act(state, step, False)
        else:
            action = policy.act(state, step)

        next_state, reward, done, _ = env.step(action)
        policy.add_to_memory(state, action, reward, next_state, done)

        if policy.memory.__len__() > batch_size:
            policy.update(step)

        state = next_state
        episode_reward += reward
        frame_idx += 1

        #if frame_idx % 1000 == 0:
            #plot(eps, rewards)

        if done:
            break

    scores_deque.append(episode_reward)
    scores.append(episode_reward)

    if eps % PRINT_EVERY == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(eps, np.mean(scores_deque)))

    if np.mean(scores_deque) >= 195:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(eps - 100,
                                                                                     np.mean(scores_deque)))

        break

    rewards.append(episode_reward)