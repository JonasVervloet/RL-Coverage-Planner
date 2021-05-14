import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class DeepRLTrainer:

    NB_EPISODES = 3000
    SAVE_EVERY = 500
    INFO_EVERY = 50

    SOFT_MAX = False

    DEVICE = 'cpu'

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

        self.terrain_diffs = []
        self.avg_terrain_diffs = []

    def train(self):
        for i in range(DeepRLTrainer.NB_EPISODES):

            current_state = torch.tensor(self.env.reset(), dtype=torch.float,
                                         device=DeepRLTrainer.DEVICE)
            done = False
            info = {}

            self.agent.update_epsilon(i)

            while not done:
                action = self.agent.select_action(
                    current_state, soft_max=DeepRLTrainer.SOFT_MAX
                )
                n_state, reward, done, info = self.env.step(action)

                action = torch.tensor(action, dtype=torch.int64,
                                      device=DeepRLTrainer.DEVICE)
                n_state = torch.tensor(n_state, dtype=torch.float,
                                       device=DeepRLTrainer.DEVICE)
                reward = torch.tensor(reward, dtype=torch.float,
                                      device=DeepRLTrainer.DEVICE)
                done = torch.tensor(done, dtype=torch.bool,
                                    device=DeepRLTrainer.DEVICE)

                self.agent.observe_transition(Transition(
                    current_state, action, n_state, reward, done
                ), device=DeepRLTrainer.DEVICE)

                current_state = n_state

            if info["full_cc"]:
                self.cc_counter += 1
                print(f"COMPLETE COVERAGE: {self.cc_counter}")

            self.total_rewards.append(info["total_reward"])
            self.nb_steps.append(info["nb_steps"])
            self.tiles_visited.append(info["total_covered_tiles"])
            self.nb_complete_cov.append(self.cc_counter)
            self.terrain_diffs.append(info["total_pos_terr_diff"])

            avg_start = 0 if i  < DeepRLTrainer.SAVE_EVERY else -DeepRLTrainer.SAVE_EVERY
            self.avg_rewards.append(np.average(self.total_rewards[avg_start:]))
            self.avg_tiles_visited.append(np.average(self.tiles_visited[avg_start:]))
            self.avg_nb_steps.append(np.average(self.nb_steps[avg_start:]))
            self.avg_terrain_diffs.append(np.average(self.terrain_diffs[avg_start:]))

            episode_nb = i + 1
            if episode_nb % DeepRLTrainer.INFO_EVERY == 0:
                print(f"Episode {episode_nb}")
                print(f"average total reward: {self.avg_rewards[-1]}")
                print(f"average nb steps: {self.avg_nb_steps[-1]}")
                print(f"average nb tiles visited: {self.avg_tiles_visited[-1]}")
                print(f"average positive terrain diff: {self.avg_terrain_diffs[-1]}")
                print(f"epsilon: {self.agent.epsilon}")
                print()

            if episode_nb % DeepRLTrainer.SAVE_EVERY == 0:
                x = range(episode_nb)

                plt.clf()
                plt.plot(x, self.total_rewards, x, self.avg_rewards)
                plt.legend(['total rewards', 'average total rewards'])
                plt.title('Total reward for every episode')
                plt.savefig(self.save_path + f"rewards.png")
                np.save(self.save_path + f"rewards.npy", self.total_rewards)
                np.save(self.save_path + f"avg_rewards.npy", self.avg_rewards)

                plt.clf()
                plt.plot(x, self.tiles_visited, x, self.avg_tiles_visited)
                plt.legend(['nb tiles visited', 'average nb tile visited'])
                plt.title('Number of tiles visited for every episode')
                plt.savefig(self.save_path + f"tiles_visited.png")
                np.save(self.save_path + f"tiles_visited.npy", self.tiles_visited)
                np.save(self.save_path + f"avg_tiles_visited.npy", self.avg_tiles_visited)

                plt.clf()
                plt.plot(x, self.nb_steps, x, self.avg_nb_steps)
                plt.legend(['nb steps', 'average nb steps'])
                plt.title('Number of steps for every episode')
                plt.savefig(self.save_path + f"nb_steps.png")
                np.save(self.save_path + f"nb_steps.npy", self.nb_steps)
                np.save(self.save_path + f"avg_nb_steps.npy", self.avg_nb_steps)

                plt.clf()
                plt.plot(x, self.nb_complete_cov)
                plt.legend(['nb complete coverage runs'])
                plt.title('Nb of complete coverage runs')
                plt.savefig(self.save_path + f"nb_complete_covs.png")
                np.save(self.save_path + f"nb_complete_covs.npy", self.nb_complete_cov)

                plt.clf()
                plt.plot(x, self.terrain_diffs, x, self.avg_terrain_diffs)
                plt.legend(['terrain differences', 'average terrain differences'])
                plt.title('Total terrain differences for every episode')
                plt.savefig(self.save_path + f"terrain_diffs.png")
                np.save(self.save_path + f"terrain_diffs.npy", self.terrain_diffs)
                np.save(self.save_path + f"avg_terrain_diffs.npy", self.avg_terrain_diffs)

                self.agent.save(self.save_path, episode_nb)


