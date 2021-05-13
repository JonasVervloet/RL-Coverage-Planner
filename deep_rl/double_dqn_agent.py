import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from collections import deque
import random
import math

from deep_rl.trainer import Transition as Transition


class DoubleDeepQAgent:

    QUEUE_LENGTH = 5000
    GAMMA = 0.9
    EPSILON_START = 0.9
    EPSILON_END = 0.05
    EPSILON_DECAY = 2000
    TARGET_UPDATE = 1000

    BATCH_SIZE = 32

    def __init__(self, network_generator, optim_class, nb_actions):
        self.policy_net = network_generator.generate_network()
        self.target_net = network_generator.generate_network()

        self.nb_actions = nb_actions

        self.replay_buffer = deque(maxlen=DoubleDeepQAgent.QUEUE_LENGTH)
        self.epsilon = DoubleDeepQAgent.EPSILON_START
        self.optimizer = optim_class(self.policy_net.parameters())
        self.batch_size = DoubleDeepQAgent.BATCH_SIZE

        self.step_counter = 1

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state_tensor, soft_max=False):
        if not soft_max:
            if random.random() > self.epsilon:
                with torch.no_grad():
                    state_eval = self.policy_net(torch.unsqueeze(state_tensor, 0))[0]
                    return torch.argmax(state_eval).item()

            return random.randrange(self.nb_actions)

        with torch.no_grad():
            state_eval = self.policy_net(torch.unsqueeze(state_tensor, 0))[0]
            probs = F.softmax(state_eval, dim=0)
            distribution = Categorical(probs=probs)
            return distribution.sample().item()

    def update_epsilon(self, episode_nb):
        diff = DoubleDeepQAgent.EPSILON_START - DoubleDeepQAgent.EPSILON_END
        self.epsilon = DoubleDeepQAgent.EPSILON_END + diff * math.exp(-1 * episode_nb / DoubleDeepQAgent.EPSILON_DECAY)

    def save(self, path, episode_nb):
        torch.save(self.policy_net.state_dict(), path + f"deep_q_agent_{episode_nb}.pt")

    def load(self, path, episode_nb):
        self.policy_net.load_state_dict(torch.load(path + f"deep_q_agent_{episode_nb}.pt"))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def evaluate(self):
        self.epsilon = 0.0

    def observe_transition(self, transition, device):
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
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_actions = torch.argmax(self.policy_net(next_state_batch), dim=1)
        next_state_values[non_final_mask] = self.target_net(next_state_batch)[
            torch.arange(self.batch_size), next_state_actions
        ][non_final_mask]
        expected_values = (next_state_values * DoubleDeepQAgent.GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_counter % DoubleDeepQAgent.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step_counter += 1