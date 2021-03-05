import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import math

from deep_q_learning.deep_q_network import NetworkGenerator


QUE_LENGTH = 5000
GAMMA = 0.9
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 2000
UPDATE_RATE = 1000
TARGET_UPDATE = 1000


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class DeepQAgent:
    def __init__(self, network_generator, batch_size=32):
        self.policy_net = network_generator.generate_network()
        self.target_net = network_generator.generate_network()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.replay_buffer = deque(maxlen=QUE_LENGTH)
        self.epsilon = EPSILON_START

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.batch_size = batch_size

        self.step_counter = 1

    def select_action(self, state_tensor):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_eval = self.policy_net(torch.unsqueeze(state_tensor, 0))[0]
                return torch.argmax(state_eval).item()

        return random.randrange(self.policy_net.nb_actions)

    def update_epsilon(self):
        diff = EPSILON_START - EPSILON_END
        self.epsilon = EPSILON_END + diff * math.exp(-1 * self.step_counter / EPSILON_DECAY)

    def save(self, path, episode_nb):
        torch.save(self.policy_net.state_dict(), path + f"deep_q_agent_{episode_nb}.pt")

    def load(self, path, episode_nb):
        self.policy_net.load_state_dict(torch.load(path + f"deep_q_agent_{episode_nb}.pt"))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def observe_transition(self, transition):
        self.replay_buffer.append(transition)

        if len(self.replay_buffer) <= self.batch_size * 10:
            return

        transitions = random.sample(self.replay_buffer, self.batch_size)
        mini_batch = Transition(*zip(*transitions))

        state_batch = torch.stack(mini_batch.state)
        action_batch = torch.stack(mini_batch.action)
        action_batch = action_batch.unsqueeze(1)
        reward_batch = torch.stack(mini_batch.reward)
        next_state_batch = torch.stack(mini_batch.next_state)
        non_final_mask = ~torch.stack(mini_batch.done)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = torch.max(self.target_net(next_state_batch), dim=1)[0][non_final_mask]
        expected_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_counter % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.update_epsilon()
        self.step_counter += 1

    def evaluate(self):
        self.epsilon = 0.0


if __name__ == "__main__":
    agent = DeepQAgent(NetworkGenerator(), batch_size=32)

    state1 = torch.rand(3, 16, 16)
    state2 = torch.rand(3, 16, 16)

    print(torch.stack([state1, state2]).shape)
    print(torch.cat([state1, state2]).shape)

    state_values = torch.rand(20, 4)
    action_values = torch.randint(4, (20, 1))

    print(state_values.shape)
    print(action_values.shape)

    state_action_values = state_values.gather(1, action_values)

    for i in range(20):
        state = torch.rand(3, 16, 16)
        print(agent.select_action(state))