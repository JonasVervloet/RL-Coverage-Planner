import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1

DISCOUNT = 0.95
NB_EPISODES = 400
PRINT_EVERY = 200
SHOW_EVERY = 2500

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

EPSILON = 1
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = NB_EPISODES
epsilon_decay_value = EPSILON / (END_EPSILON_DECAY - START_EPSILON_DECAY)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.size)
print(q_table.shape)
print()

# statistics
nb_successes = 0
ep_rewards = []
aggr_ep_rewards = {
    'ep': [],
    'avg': [],
    'max': [],
    'min': []
}


def get_discrete_state(state):
    relative_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(relative_state.astype(np.int64))


# Epsilon Greedy Action Selection
def select_action(table, state, epsilon):
    if np.random.random() > epsilon:
        return np.argmax(table[state])
    else:
        return np.random.randint(0, env.action_space.n)


def update_q_value(table, state, a, r, n_state, finished, g_reached):
    index = state + (a,)
    if not finished:
        max_next_value = np.max(table[n_state])
        curr_value = table[index]
        new_value = curr_value + LEARNING_RATE * (r + DISCOUNT * max_next_value - curr_value)
        table[index] = new_value
    elif g_reached:
        table[index] = 0

    return table


for episode in range(NB_EPISODES):

    current_state = get_discrete_state(env.reset())
    total_reward = 0

    done = False

    if episode % PRINT_EVERY == 0:
        print("Episode {}".format(episode))
        print("# successes: {}".format(nb_successes))

    while not done:

        action = select_action(q_table, current_state, EPSILON)

        new_state, reward, done, _ = env.step(action)
        goal_reached = done and new_state[0] >= env.goal_position
        if goal_reached:
            nb_successes += 1

        new_state = get_discrete_state(new_state)
        total_reward += reward

        q_table = update_q_value(
            q_table,
            current_state, action,
            reward, new_state,
            done, goal_reached
        )

        if episode % SHOW_EVERY == 0 and episode != 0:
            env.render()

        current_state = new_state

    ep_rewards.append(total_reward)
    if episode % PRINT_EVERY == 0:
        recent_rewards = ep_rewards[-PRINT_EVERY:]
        average_reward = sum(recent_rewards) / PRINT_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(recent_rewards))
        aggr_ep_rewards['min'].append(min(recent_rewards))
        print(f'average reward: {average_reward}')
        print(f'max reward: {max(recent_rewards)}')
        print(f'current epsilon: {EPSILON}')
        print()

    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
        EPSILON -= epsilon_decay_value

env.close()

path = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/"
np.save(f"{path}{NB_EPISODES}-qtable.npy", q_table)

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='average rewards')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max rewards')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min rewards')
plt.legend(loc=4)
plt.show()
