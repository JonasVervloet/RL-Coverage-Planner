import gym
import numpy as np
import matplotlib.pyplot as plt

NB_EPISODES = 10000
PRINT_EVERY = 1000

ALPHA = 0.1
DISCOUNT = 0.9
EPSILON_START = 0.99
MIN_EPSILON = 0.01
EPSILON_DECAY = 1.0/NB_EPISODES

env = gym.make("Taxi-v3")
OBS_SPACE = env.observation_space.n
ACT_SPACE = env.action_space.n
print(OBS_SPACE)
print(ACT_SPACE)


def select_action(table, s, eps):
    if np.random.random() > eps:
        return np.argmax(table[s])
    else:
        return np.random.randint(0, ACT_SPACE)

def update_q_value(table, s, a, r, s_n):
    max_next = np.max(table[s_n])
    curr_value = table[s][a]
    table[s][a] = curr_value + ALPHA * (r + DISCOUNT * max_next - curr_value)
    return table

def decay_epsilon(eps):
    return max(MIN_EPSILON, eps - EPSILON_DECAY)


q_table = np.zeros((OBS_SPACE, ACT_SPACE))
epsilon = EPSILON_START
nbs_steps = []
avg_nbs_steps = []
total_rewards = []
avg_total_rewards = []
print(q_table.shape)

for episode in range(NB_EPISODES):
    current_state = env.reset()
    nb_steps = 0
    total_reward = 0
    done = False

    while not done:

        action = select_action(q_table, current_state, epsilon)
        new_state, reward, done, _ = env.step(action)
        nb_steps += 1
        total_reward += reward

        q_table = update_q_value(
            q_table,
            current_state, action,
            reward, new_state
        )

        current_state = new_state

    nbs_steps.append(nb_steps)
    avg_nbs_steps.append(np.average(nbs_steps))
    total_rewards.append(total_reward)
    avg_total_rewards.append(np.average(total_rewards))

    epsilon = decay_epsilon(epsilon)

    if episode % PRINT_EVERY == 0:
        print()
        print(f"EPISODE {episode}")
        print(f"average reward {avg_nbs_steps[episode]}")
        print(f"avg nb steps {avg_total_rewards[episode]}")

plt.plot(range(NB_EPISODES), total_rewards)
plt.plot(range(NB_EPISODES), avg_total_rewards)
plt.show()
plt.plot(range(NB_EPISODES), nbs_steps)
plt.plot(range(NB_EPISODES), avg_nbs_steps)
plt.show()

current_state = env.reset()
env.render()
done = False
while not done:
    action = np.argmax(q_table[current_state])
    new_state, reward, done, _ = env.step(action)
    env.render()
    print(reward)

    current_state = new_state
