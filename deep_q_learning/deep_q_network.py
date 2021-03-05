import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):
    def __init__(self, width, height, input_depth=3, nb_actions=4):
        super(DeepQNetwork, self).__init__()
        self.width = width
        self.height = height
        self.input_depth = input_depth
        self.nb_actions = nb_actions

        self.conv1 = nn.Conv2d(input_depth, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(width * height * 32, nb_actions)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))

        x = self.fc1(torch.flatten(x, start_dim=1))

        return x


class NetworkGenerator:
    def __init__(self):
        self.width = 16
        self.height = 16

    def generate_network(self):
        return DeepQNetwork(self.width, self.height)


if __name__ == "__main__":
    network = DeepQNetwork(16, 16, 3, 4)

    test_input = torch.rand(20, 3, 16, 16)
    print(test_input.shape)

    output = network(test_input)
    print(output.shape)

    nb_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(nb_parameters)


