import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDeepQNetwork(nn.Module):
    def __init__(self, dim, input_depth, nb_actions):
        super(SimpleDeepQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_depth, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(dim[0] * dim[1] * 32, nb_actions)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))

        x = self.fc1(torch.flatten(x, start_dim=1))
        return x


class SimpleDeepQNetworkGenerator:
    def __init__(self, dim, input_depth, nb_actions):
        self.dim = dim
        self.input_depth = input_depth
        self.nb_actions = nb_actions

    def generate_network(self):
        return SimpleDeepQNetwork(self.dim, self.input_depth, self.nb_actions)