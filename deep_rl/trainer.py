import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class DeepRLTrainer:

    NB_EPISODES = 3000
    SAVE_EVERY = 500
    INFO_EVERY = 50

    def __init__(self, environment, agent, save_path):
        self.env = environment
        self.agent = agent

        self.save_path = save_path

        self.total_rewards = []
        self.avg_rewards = []

        self.tiles_visited = []
        self.avg_tiles_visited = []

        self.nb_steps = []
        self.avg_nb_steps = []

        self.cc_counter = 0
        self.nb_complete_cov = []

    def train(self):
        for i in range(DeepRLTrainer.NB_EPISODES):

            current_state = torch.tensor(self.env.reset(), dtype=torch.float)
            done = False
            total_reward = 0
            nb_steps = 0

            self.agent.update_epsilon(i)

            while not done:
                action = self.agent.select_action(current_state)
                n_state, reward, done = self.env.step(action)

                total_reward += reward
                nb_steps += 1

                action = torch.tensor(action, dtype=torch.int64)
                n_state = torch.tensor(n_state, dtype=torch.float)
                reward = torch.tensor(reward, dtype=torch.float)
                done = torch.tensor(done, dtype=torch.bool)

                self.agent.observe_transition(Transition(
                    current_state, action, n_state, reward, done
                ))

                current_state = n_state

            if self.env.complete_coverage():
                self.cc_counter += 1
                print(f"COMPLETE COVERAGE: {self.cc_counter}")

            self.total_rewards.append(total_reward)
            self.nb_steps.append(nb_steps)
            self.tiles_visited.append(self.env.get_nb_visited_tiles())
            self.nb_complete_cov.append(self.cc_counter)

            if i < DeepRLTrainer.SAVE_EVERY:
                self.avg_rewards.append(np.average(self.total_rewards))
                self.avg_tiles_visited.append(np.average(self.tiles_visited))
                self.avg_nb_steps.append(np.average(self.nb_steps))
            else:
                self.avg_rewards.append(np.average(self.total_rewards[-DeepRLTrainer.SAVE_EVERY:]))
                self.avg_tiles_visited.append(np.average(self.tiles_visited[-DeepRLTrainer.SAVE_EVERY:]))
                self.avg_nb_steps.append(np.average(self.nb_steps[-DeepRLTrainer.SAVE_EVERY:]))

            episode_nb = i + 1
            if episode_nb % DeepRLTrainer.INFO_EVERY == 0:
                print(f"Episode {episode_nb}")
                print(f"total reward: {total_reward}")
                print(f"nb steps: {nb_steps}")
                print(f"tiles visited: {self.tiles_visited[i]}")
                print(f"epsilon: {self.agent.epsilon}")
                print()

            if episode_nb % DeepRLTrainer.SAVE_EVERY == 0:
                x = range(episode_nb)

                plt.clf()
                plt.plot(x, self.total_rewards, x, self.avg_rewards)
                plt.legend(['total rewards', 'average total rewards'])
                plt.title('Total reward for every episode')
                plt.savefig(self.save_path + f"reward_episode{episode_nb}.png")
                np.save(self.save_path + f"rewards_episode{episode_nb}.npy", self.total_rewards)
                np.save(self.save_path + f"avg_rewards_episode{episode_nb}.npy", self.avg_rewards)

                plt.clf()
                plt.plot(x, self.tiles_visited, x, self.avg_tiles_visited)
                plt.legend(['nb tiles visited', 'average nb tile visited'])
                plt.title('Number of tiles visited for every episode')
                plt.savefig(self.save_path + f"tiles_visited_episode{episode_nb}.png")
                np.save(self.save_path + f"tiles_visited_episode{episode_nb}.npy", self.tiles_visited)
                np.save(self.save_path + f"avg_tiles_visited_episode{episode_nb}.npy", self.avg_tiles_visited)

                plt.clf()
                plt.plot(x, self.nb_steps, x, self.avg_nb_steps)
                plt.legend(['nb steps', 'average nb steps'])
                plt.title('Number of steps for every episode')
                plt.savefig(self.save_path + f"nb_steps_episode{episode_nb}.png")
                np.save(self.save_path + f"nb_steps_episode{episode_nb}.npy", self.nb_steps)
                np.save(self.save_path + f"avg_nb_steps_episode{episode_nb}.npy", self.avg_nb_steps)

                plt.clf()
                plt.plot(x, self.nb_complete_cov)
                plt.legend(['nb complete coverage runs'])
                plt.title('Nb of complete coverage runs')
                plt.savefig(self.save_path + f"nb_complete_cov_episode{episode_nb}.png")
                np.save(self.save_path + f"nb_complete_cov_episode{episode_nb}.npy", self.nb_complete_cov)

                self.agent.save(self.save_path, episode_nb)


