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


class SimpleDeepQNetwork2(nn.Module):
    def __init__(self, dim, input_depth, nb_actions):
        super(SimpleDeepQNetwork2, self).__init__()

        self.conv1 = nn.Conv2d(input_depth, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(64 * dim[0] * dim[1], nb_actions)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = self.fc1(torch.flatten(x, start_dim=1))
        return x


class SimpleDeepQNetworkGenerator2:
    def __init__(self, dim, input_depth, nb_actions):
        self.dim = dim
        self.input_depth = input_depth
        self.nb_actions = nb_actions

    def generate_network(self):
        return SimpleDeepQNetwork2(self.dim, self.input_depth, self.nb_actions)


if __name__ == "__main__":

    dim = 8

    generator1 = SimpleDeepQNetworkGenerator(
        dim=(dim, dim),
        input_depth=3,
        nb_actions=4
    )
    network1 = generator1.generate_network()
    print(sum(p.numel() for p in network1.parameters() if p.requires_grad))

    test_tens = torch.rand((32, 3, dim, dim))
    output = network1(test_tens)
    print(output.shape)

    generator2 = SimpleDeepQNetworkGenerator2(
        dim=(dim, dim),
        input_depth=3,
        nb_actions=4
    )
    network2 = generator2.generate_network()
    print(sum(p.numel() for p in network2.parameters() if p.requires_grad))

    test_tens = torch.rand((32, 3, dim, dim))
    output = network2(test_tens)
    print(output.shape)