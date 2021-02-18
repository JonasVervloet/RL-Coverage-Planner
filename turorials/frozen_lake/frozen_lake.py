import gym
import numpy as np
import matplotlib.pyplot as plt

from turorials.frozen_lake.coverage_env import FrozenLakeCoverage

NB_EPISODES = 50000
PRINT_EVERY = 1000

MAX_STEPS = 100
ALPHA = 0.1
DISCOUNT = 0.9
EPSILON_START = 0.99
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.000025

# env = gym.make("FrozenLake-v0")
# env = gym.make("FrozenLake-v0", is_slippery=False)
env = FrozenLakeCoverage(slippery=False)


def select_action(table, s, eps):
    if np.random.random() > eps:
        return np.argmax(table[s])
    else:
        return np.random.randint(0, env.action_size)


def update_q_value(table, s, a, r, s_n):
    max_next = np.max(table[s_n])
    curr_value = table[s][a]
    table[s][a] = curr_value + ALPHA * (r + DISCOUNT * max_next - curr_value)
    return table


def decay_epsilon(eps):
    return max(MIN_EPSILON, eps-EPSILON_DECAY)


q_table = np.zeros(env.observation_size + (env.action_size, ))
print(q_table.shape)
epsilon = EPSILON_START
nbs_steps = []
avg_steps = []
avg_nb_success = []
nb_successes = 0

for episode in range(NB_EPISODES):

    current_state = env.reset()
    nb_steps = 0
    done = False

    while not done:

        action = select_action(q_table, current_state, epsilon)
        new_state, reward, done, _ = env.step(action)
        nb_steps += 1

        q_table = update_q_value(
            q_table, current_state, action,
            reward, new_state
        )

        current_state = new_state

    if reward >= 1.0:
        nbs_steps.append(nb_steps)
        avg_steps.append(sum(nbs_steps) / len(nbs_steps))
        nb_successes += 1
    avg_nb_success.append(nb_successes/(episode + 1))

    epsilon = decay_epsilon(epsilon)

    if episode % PRINT_EVERY == 0:
        print()
        print(f"EPISODE {episode}")
        print(f"nb successes: {nb_successes}")
        print(f"max nb steps: {max(nbs_steps) if len(nbs_steps) != 0 else None}")
        print(f"min nb steps: {min(nbs_steps) if len(nbs_steps) != 0 else None}")
        print(f"epsilon: {epsilon}")
        print(f"max visits: {env.max_visits}")

env.close()

plt.plot(range(len(nbs_steps)), nbs_steps)
plt.plot(range(len(avg_steps)), avg_steps)
plt.show()

plt.plot(range(NB_EPISODES), avg_nb_success)
plt.show()

current_state = env.reset()
env.render()
done = False
while not done:
    action = np.argmax(q_table[current_state])
    new_state, _, done, _ = env.step(action)
    env.render()

    current_state = new_state
