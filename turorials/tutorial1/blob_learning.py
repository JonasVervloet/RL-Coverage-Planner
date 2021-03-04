import matplotlib.pyplot as plt
import numpy as np

from turorials.tutorial1.environment import BlobEnv
from turorials.tutorial1.q_learner import QLearner

SIZE = 5

NB_EPISODES = 25000
START_EPS = 0.9
EPS_DECAY = 0.9999
SHOW_EVERY = 1000
PRINT_EVERY = 1000

LEARNING_RATE = 0.1
DISCOUNT = 0.95

env = BlobEnv(SIZE)

q_learner = QLearner(SIZE, START_EPS, LEARNING_RATE, DISCOUNT)

nb_successes = 0
rewards = []
avg_rewards = []

for episode in range(NB_EPISODES):
    current_state = env.reset()
    total_reward = 0

    done = False

    while not done:
        action = q_learner.select_action(current_state)

        new_state, reward, done, g_reached = env.step(action)

        if g_reached:
            nb_successes += 1

        total_reward += reward
        q_learner.update_q_value(current_state, action, reward, new_state)

    rewards.append(total_reward)
    q_learner.decay_epsilon(EPS_DECAY)
    avg_rewards.append(sum(rewards)/len(rewards))
    if episode%PRINT_EVERY == 0:
        print(f"Episode {episode}")
        print(f"total_reward: {total_reward}")
        print(f"nb successes:  {nb_successes}")
        print(f"epsilon: {q_learner.epsilon}")
        print()

moving_avg = np.convolve(rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
plt.plot(range(len(moving_avg)), moving_avg, label='avg rewards')
plt.legend()
plt.show()