import numpy as np
import matplotlib.pyplot as plt
import torch

from environments.simple_environment import SimpleEnvironment
from deep_q_learning.deep_q_agent import DeepQAgent, Transition
from deep_q_learning.deep_q_network import NetworkGenerator

NB_EPISODES = 1000
INFO_EVERY = 50

SAVE_PATH = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/"

name = "test_grid.npy"
obstacle_grid = np.load(SAVE_PATH + name)
env = SimpleEnvironment(obstacle_grid)
network_generator = NetworkGenerator()
agent = DeepQAgent(network_generator, batch_size=32)

total_rewards = []
avg_total_rewards = []
nbs_steps = []
avg_nbs_steps = []
tiles_visited = []
avg_tiles_visited = []

for episode_nb in range(NB_EPISODES + 1):

    current_state = torch.tensor(env.reset(), dtype=torch.float)
    done = False
    total_reward = 0
    nb_steps = 0

    while not done:
        action = agent.select_action(current_state)
        n_state, reward, done = env.step(action)
        action = torch.tensor(action, dtype=torch.int64)
        n_state = torch.tensor(n_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)

        agent.observe_transition(Transition(
            current_state, action, n_state, reward, done
        ))

        current_state = n_state
        total_reward += reward
        nb_steps += 1

    if np.sum(env.visited_tiles) == 120:
        print("Full coverage!!!")
    if np.sum(env.visited_tiles) > 120:
        print("TOO MUCH TILES VISITED!!!")
        print(f"tiles visited: {np.sum(tiles_visited)}")

    total_rewards.append(total_reward)
    avg_total_rewards.append(np.average(total_rewards))
    nbs_steps.append(nb_steps)
    avg_nbs_steps.append(np.average(nbs_steps))
    tiles_visited.append(np.sum(env.visited_tiles))
    avg_tiles_visited.append(np.average(tiles_visited))

    if episode_nb % INFO_EVERY == 0:
        print(f"Episode {episode_nb}")
        print(f"total reward: {total_reward}")
        print(f"nb steps: {nb_steps}")
        print(f"tiles visited: {tiles_visited[episode_nb]}")
        print(f"total nb steps: {agent.step_counter}")
        print(f"epsilon: {agent.epsilon}")
        print()
        
        plt.clf()
        x = range(episode_nb + 1)
        plt.plot(x, total_rewards, x, avg_total_rewards)
        plt.legend(['total rewards', 'average total rewards'])
        plt.title('Total reward for every episode')
        plt.savefig(SAVE_PATH + f"reward_episode{episode_nb}.png")

        plt.clf()
        plt.plot(x, nbs_steps, x, avg_nbs_steps)
        plt.legend(['nb steps', 'average nb steps'])
        plt.title('Number of steps for every episode')
        plt.savefig(SAVE_PATH + f"nb_steps_episode{episode_nb}.png")

        plt.clf()
        plt.plot(x, tiles_visited, x, avg_tiles_visited)
        plt.legend(['nb visited tiles', 'average nb visited tiles'])
        plt.title('Number of visited tiles for every episode')
        plt.savefig(SAVE_PATH + f"tiles_visited_episode{episode_nb}.png")

agent.save(SAVE_PATH, NB_EPISODES)
