import numpy as np

class QLearner:
    def __init__(self, size, epsilon, learning_rate, discount):
        self.size = size
        self.epsilon = epsilon
        self.lr = learning_rate
        self.discount = discount

        self.q_table = None
        self.init_q_table()

    def init_q_table(self):
        shape = (2*self.size - 1, )*4 + (4,)
        self.q_table = np.random.uniform(-5, 0, shape)
        self.q_table[0, 0] = 0
        self.q_table[:, :, 0, 0] = 0

    def state_to_index(self, state):
        return tuple(state + np.array((self.size-1,)*4))

    def decay_epsilon(self, decay):
        self.epsilon = decay * self.epsilon

    def select_action(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.q_table[self.state_to_index(state)])
        else:
            return np.random.randint(0, self.q_table.shape[-1])

    def update_q_value(self, state, action, reward, n_state):
        index = self.state_to_index(state) + (action,)
        n_index = self.state_to_index(n_state)
        update_target = reward + self.discount * np.max(self.q_table[n_index])
        curr_value = self.q_table[index]
        self.q_table[index] = curr_value + self.lr * (update_target - curr_value)
