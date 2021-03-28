import random
import numpy as np
import matplotlib.pyplot as plt

from environments.env_generation import SingleEnvironmentGenerator
from environments.env_representation import EnvironmentRepresentation


class Environment:

    MOVE_PUNISHMENT = 0.05
    TERRAIN_PUNISHMENT = 0.05
    OBSTACLE_PUNISHMENT = 0.5

    DISCOVER_REWARD = 1.0
    COVERAGE_REWARD = 50.0

    MAX_STEP_MULTIPLIER = 2

    def __init__(self, generator):

        self.generator = generator
        self.env_info = None

        self.current_pos = None
        self.visited_tiles = None
        self.done = True
        self.nb_steps = 0
        self.total_reward = 0.0
        self.total_terrain_diff = 0.0

        self.reset()

    def get_dimension(self):
        return self.generator.get_dimension()

    def gives_terrain_info(self):
        return self.env_info.has_terrain_info()

    def get_nb_visited_tiles(self):
        return np.sum(self.visited_tiles)

    def get_input_depth(self):
        self.reset()
        return 4 if self.env_info.has_terrain_info() else 3

    def get_nb_actions(self):
        return 4

    def reset(self):
        self.done = False
        self.env_info = self.generator.generate_environment()
        self.current_pos = random.sample(self.env_info.start_positions, 1)[0]
        self.visited_tiles = np.zeros_like(self.env_info.obstacle_map)
        self.visited_tiles[self.current_pos] = 1
        self.nb_steps = 0
        self.total_reward = 0
        self.total_terrain_diff = 0

        return self.get_state()

    def get_state(self):
        current_pos_grid = np.zeros_like(self.env_info.obstacle_map)
        current_pos_grid[self.current_pos] = 1

        if self.env_info.has_terrain_info():
            return np.stack([current_pos_grid, self.visited_tiles, self.env_info.obstacle_map, self.env_info.terrain_map])
        else:
            return np.stack([current_pos_grid, self.visited_tiles, self.env_info.obstacle_map])

    def complete_coverage(self):
        return self.env_info.nb_free_tiles == np.sum(self.visited_tiles)

    def has_complete_coverage(self, n_pos):
        if self.env_info.obstacle_map[n_pos] == 1:
            return False

        nb_visited_tiles = np.sum(self.visited_tiles)
        if self.visited_tiles[n_pos] == 0:
            nb_visited_tiles += 1

        return nb_visited_tiles == self.env_info.nb_free_tiles

    def get_reward(self, n_pos):
        reward = -self.MOVE_PUNISHMENT
        if self.env_info.has_terrain_info():
            diff = abs(self.env_info.terrain_map[self.current_pos] - self.env_info.terrain_map[n_pos])
            reward += diff * -self.TERRAIN_PUNISHMENT

        if self.env_info.obstacle_map[n_pos] == 1:
            reward -= self.OBSTACLE_PUNISHMENT
        else:
            if self.visited_tiles[n_pos] == 0:
                reward += self.DISCOVER_REWARD

            if self.has_complete_coverage(n_pos):
                reward += self.COVERAGE_REWARD

        return reward

    def is_done(self, n_pos):
        if self.env_info.obstacle_map[n_pos] == 1:
            self.done = True
            return True

        if self.has_complete_coverage(n_pos):
            self.done = True
            return True

        if self.nb_steps > self.MAX_STEP_MULTIPLIER * self.env_info.nb_free_tiles:
            self.done = True
            return True

        return False

    def step(self, action):
        if self.done:
            raise Exception("ENVIRONMENT: should reset first...")

        if action == 0:
            n_position = (max(0, self.current_pos[0] - 1), self.current_pos[1])
        elif action == 1:
            n_position = (min(self.env_info.obstacle_map.shape[0] - 1, self.current_pos[0] + 1), self.current_pos[1])
        elif action == 2:
            n_position = (self.current_pos[0], max(0, self.current_pos[1] - 1))
        elif action == 3:
            n_position = (self.current_pos[0], min(self.env_info.obstacle_map.shape[1] - 1, self.current_pos[1] + 1))
        else:
            raise Exception("ACTION NOT SUPPORTED!")

        self.nb_steps += 1

        reward = self.get_reward(n_position)
        self.total_reward += reward
        done = self.is_done(n_position)

        if self.env_info.obstacle_map[n_position] == 0 and self.visited_tiles[n_position] == 0:
            self.visited_tiles[n_position] = 1

        info = {
            "reward": reward,
            "total reward": self.total_reward,
            "nb steps": self.nb_steps,
            "nb visited tiles": self.get_nb_visited_tiles(),
            "complete coverage": self.complete_coverage(),
        }
        if self.env_info.has_terrain_info():
            terrain_diff = abs(
                self.env_info.terrain_map[self.current_pos] - self.env_info.terrain_map[n_position]
            )
            self.total_terrain_diff += terrain_diff
            info["terrain diff"] = terrain_diff
            info["total terrain diff"] = self.total_terrain_diff

        self.current_pos = n_position

        return self.get_state(), reward, done, info


if __name__ == "__main__":
    save_path = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/"

    generator = SingleEnvironmentGenerator([save_path, "env_8x_terrain"])
    env = Environment(generator)

    state = env.reset()
    print(state.shape)
    print(env.current_pos)
    print(np.sum(env.visited_tiles))
    print(env.nb_steps)
    print()

    plt.imshow(np.moveaxis(state, 0, -1))
    plt.show()

    done = False
    last_pos = env.current_pos
    while not done:
        print("Enter action: ")
        action = int(input())
        print()
        state, reward, done, info = env.step(action)
        print(state.shape)
        print(f"reward: {info['reward']}")
        print(f"total reward: {info['total reward']}")
        print(f"done: {done}")
        print(f"current position: {env.current_pos}")
        print(f"nb steps: {info['nb steps']}")
        print(f"visited tiles: {info['nb visited tiles']}")
        if (env.env_info.terrain_map is not None):
            print(env.env_info.terrain_map[last_pos])
            print(env.env_info.terrain_map[env.current_pos])
            print(f"terrain diff: {info['terrain diff']}")
            print(f"total terrain diff:  {info['total terrain diff']}")
        print()

        last_pos = env.current_pos

        plt.imshow(np.moveaxis(state, 0, -1))
        plt.show()