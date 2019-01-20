import gym
import torch.nn as nn
import numpy as np

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class Flatten(nn.Module):
    """
    Given a tensor of Batch * Color * Height * Width, flatten it and make it 1D.
    Used for Linear GANs

    Usable in nn.Sequential
    """
    def forward(self, x):

        B, C, H, W = x.size()

        return x.view(B, -1) #returns a vector that is B * (C * H * W)
