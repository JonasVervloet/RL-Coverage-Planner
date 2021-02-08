import numpy as np
import matplotlib.pyplot as plt


def initialize_action_values(nb_arms):
    return np.zeros(nb_arms)


def update_action_values(old_values):
    random_walk = 0.01 * np.random.standard_normal(old_values.shape)
    return old_values + random_walk


def sample_reward(action_values, index):
    return np.random.randn() + action_values[index]


def greedy_action_selection(estimated_values, epsilon):
    if np.random.random() >= epsilon:
        return np.argmax(estimated_values)
    else:
        return np.random.randint(estimated_values.size)


def update_value(values, index, new_value, step_size):
    old_value = values[index]
    values[index] = old_value + step_size * (new_value - old_value)
    return values


def update_value_estimate(estimated_values, nb_visits, index, reward):
    nb_visits[index] = nb_visits[index] + 1
    new_values = update_value(
        estimated_values, index, reward,
        (1/nb_visits[index])
    )

    return nb_visits, new_values


def update_avg_rewards(avg_rewards, index, reward, run_nb):
    return update_value(avg_rewards, index, reward, (1/(run_nb + 1)))


def is_optimal_action(action_values, index):
    return np.argmax(action_values) == index


def update_optimal_selections(optimal_selections, index, optimal, run_nb):
    new_value = 1 if optimal else 0
    return update_value(optimal_selections, index, new_value, (1/(run_nb + 1)))


NB_RUNS = 500
NB_ARMS = 10
NB_STEPS = 10000
EPSILON = 0.1
ALPHA = 0.1

avg_rewards_avg_st = np.zeros(NB_STEPS)
optimal_selections_avg_st = np.zeros(NB_STEPS)

avg_rewards_cte_st = np.zeros(NB_STEPS)
optimal_selections_cte_st = np.zeros(NB_STEPS)

for run_nb in range(NB_RUNS):

    if run_nb % 10 == 0:
        print("#### RUN {} ####".format(run_nb))

    action_values = initialize_action_values(NB_ARMS)

    estimated_values_avg_st = np.zeros(NB_ARMS)
    nb_visits_avg_st = np.zeros(NB_ARMS)

    estimated_values_cte_st = np.zeros(NB_ARMS)

    for step in range(NB_STEPS):
        # SAMPLE AVERAGES
        action_avg_st = greedy_action_selection(estimated_values_avg_st, EPSILON)
        reward_avg_st = sample_reward(action_values, action_avg_st)

        estimated_values_avg_st, nb_visits_avg_st = update_value_estimate(
            estimated_values_avg_st, nb_visits_avg_st, action_avg_st, reward_avg_st
        )

        avg_rewards_avg_st = update_avg_rewards(
            avg_rewards_avg_st, step, reward_avg_st, run_nb
        )
        optimal_action = is_optimal_action(action_values, action_avg_st)
        optimal_selections_avg_st = update_optimal_selections(
            optimal_selections_avg_st, step, optimal_action, run_nb
        )

        # CONSTANT STEP SIZE
        action_cte_st = greedy_action_selection(estimated_values_cte_st, EPSILON)
        reward_cte_st = sample_reward(action_values, action_cte_st)

        estimated_values_cte_st = update_value(
            estimated_values_cte_st, action_cte_st, reward_cte_st, ALPHA
        )

        avg_rewards_cte_st = update_avg_rewards(
            avg_rewards_cte_st, step, reward_cte_st, run_nb
        )
        optimal_action = is_optimal_action(action_values, action_cte_st)
        optimal_selections_cte_st = update_optimal_selections(
            optimal_selections_cte_st, step, optimal_action, run_nb
        )

        action_values = update_action_values(action_values)

# PLOT RESULTS
x_values = range(NB_STEPS)
plt.plot(x_values, avg_rewards_avg_st,
         linewidth=0.2, label="sample averages")
plt.plot(x_values, avg_rewards_cte_st,
         linewidth=0.2, label="constant step size")
plt.xlabel("steps")
plt.ylabel("average reward")
plt.show()
plt.clf()
plt.plot(x_values, optimal_selections_avg_st,
         linewidth=0.2, label="sample averages")
plt.plot(x_values, optimal_selections_cte_st,
         linewidth=0.2, label="constant step size")
plt.xlabel("steps")
plt.ylabel("optimal action selection")
plt.show()