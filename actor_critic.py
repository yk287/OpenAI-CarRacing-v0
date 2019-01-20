import torch
import torch.nn as nn
import torch.nn.functional as F

from util import Flatten

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):

    def __init__(self, state_size, action_size, hidden_layer=256, hidden_layer1=256, w_init=3e-3):
        super(Actor, self).__init__()


        self.cnn = nn.Sequential(
            nn.Conv2d(9, 32, 4, stride=2, padding=1),  # batch_size * 32 * 46 * 46
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
            Flatten()
        )

        self.linear  = nn.Linear(256 * 3 * 3, hidden_layer)
        self.linear1 = nn.Linear(hidden_layer, hidden_layer1)
        self.linear2 = nn.Linear(hidden_layer1, action_size)

        self.linear2.weight.data.uniform_(-w_init, w_init)
        self.linear2.bias.data.uniform_(-w_init, w_init)

    def forward(self, state):

        action = F.relu(self.linear(self.cnn(state)))
        action = F.relu(self.linear1(action))
        action = torch.tanh(self.linear2(action))

        return action

    def get_action(self, state):
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


class Critic(nn.Module):

    def __init__(self, state_size, action_size, hidden_layer=256, hidden_layer1=256, w_init=3e-3):
        super(Critic, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(9, 32, 4, stride=2, padding=1),  # batch_size * 32 * 46 * 46
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
            Flatten()
        )

        self.linear = nn.Linear(256 * 3 * 3 + action_size, hidden_layer)
        self.linear1 = nn.Linear(hidden_layer, hidden_layer1)
        self.linear2 = nn.Linear(hidden_layer1, action_size)

        self.linear2.weight.data.uniform_(-w_init, w_init)
        self.linear2.bias.data.uniform_(-w_init, w_init)

    def forward(self, state, action):

        state = self.cnn(state)
        value   = torch.cat([state, action], 1)
        value   = F.relu(self.linear(value))
        value   = F.relu(self.linear1(value))
        value   = self.linear2(value)

        return value

