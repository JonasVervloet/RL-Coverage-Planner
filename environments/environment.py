import random
import numpy as np
import matplotlib.pyplot as plt

from environments.env_representation import EnvironmentRepresentation

class Environment:

    MOVE_PUNISHMENT = 0.05
    OBSTACLE_PUNISHMENT = 0.5

    DISCOVER_REWARD = 1.0
    COVERAGE_REWARD = 50.0

    MAX_STEP_MULTIPLIER = 2

    def __init__(self, generator=None):
        self.generator = generator

        self.env_info = None
        self.current_pos = None
        self.visited_tiles = None
        self.nb_steps = 0
        self.done = True

        self.single_env = False

    def set_environment_representation(self, representation):
        self.env_info = representation
        self.single_env = True

    def get_nb_visited_tiles(self):
        return np.sum(self.visited_tiles)

    def get_input_depth(self):
        return 3

    def get_nb_actions(self):
        return 4

    def reset(self):
        self.done = False
        if not self.single_env:
            self.env_info = self.generator.generate_environment()
        self.current_pos = random.sample(self.env_info.start_positions, 1)[0]
        self.visited_tiles = np.zeros_like(self.env_info.obstacle_map)
        self.visited_tiles[self.current_pos] = 1
        self.nb_steps = 0

        return self.get_state()

    def get_state(self):
        if self.env_info.terrain_map is not None:
            raise Exception("NOT SUPPORTED YET")
        else:
            current_pos_grid = np.zeros_like(self.env_info.obstacle_map)
            current_pos_grid[self.current_pos] = 1

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
        done = self.is_done(n_position)

        if self.env_info.obstacle_map[n_position] == 0 and self.visited_tiles[n_position] == 0:
            self.visited_tiles[n_position] = 1

        self.current_pos = n_position

        return self.get_state(), reward, done


if __name__ == "__main__":
    save_path = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/"

    env_representation = EnvironmentRepresentation()
    env_representation.load(save_path, "env_8x")

    env = Environment()
    env.set_environment_representation(env_representation)

    state = env.reset()
    print(state.shape)
    print(env.current_pos)
    print(np.sum(env.visited_tiles))
    print(env.nb_steps)
    print()

    plt.imshow(np.moveaxis(state, 0, -1))
    plt.show()

    done = False
    while not done:
        print("Enter action: ")
        action = int(input())
        print()
        state, reward, done = env.step(action)
        print(state.shape)
        print(reward)
        print(done)
        print(env.current_pos)
        print(env.nb_steps)
        print(np.sum(env.visited_tiles))
        print()

        plt.imshow(np.moveaxis(state, 0, -1))
        plt.show()