import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork3(nn.Module):
    def __init__(self, dim, input_depth, nb_actions):
        super(DeepQNetwork3, self).__init__()

        assert(dim[0] > 12)
        assert(dim[1] > 12)

        self.conv1 = nn.Conv2d(input_depth, 5, kernel_size=5, padding=0)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=5, padding=0)

        self.fc1 = nn.Linear(20 * (dim[0] - 12) * (dim[1] - 12), nb_actions)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = self.fc1(torch.flatten(x, start_dim=1))
        return x


class DeepQNetworkGenerator3:
    def __init__(self, dim, input_depth, nb_actions, device):
        self.dim = dim
        self.input_depth = input_depth
        self.nb_actions = nb_actions
        self.device = device

    def generate_network(self):
        model = DeepQNetwork3(self.dim, self.input_depth, self.nb_actions)
        model.to(self.device)
        return model