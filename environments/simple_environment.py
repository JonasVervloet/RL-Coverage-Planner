import numpy as np
import matplotlib.pyplot as plt
import random

from turorials.perlin_noise.obstacle_generation import flood_grid

OBSTACLE_PUNISHMENT = 0.5
DISCOVER_REWARD = 1.0
MOVE_PUNISHMENT = 0.05
COVERAGE_REWARD = 50.0
MAX_NB_STEPS = 300

class SimpleEnvironment:
    def __init__(self, obstacles):
        self.obstacles = obstacles
        self.start_positions = self.get_start_positions()
        self.visited_tiles = np.zeros_like(self.obstacles)
        self.current_pos = random.sample(self.start_positions, 1)[0]
        self.nb_steps = 0

    def get_start_positions(self):
        regions = flood_grid(self.obstacles)
        if len(regions) > 2:
            raise Exception("More than two regions not supported!!")

        print("NB TILES TO BE COVERED")
        if regions[0][0] == 0:
            print(len(regions[0][1]) + len(regions[0][2]))
            return regions[0][1]

        if regions[1][0] == 0:
            print(len(regions[1][1]) + len(regions[1][2]))
            return regions[1][1]

        raise Exception("Only obstacle regions given!!")

    def render(self):
        render = np.copy(self.obstacles)
        for tile in self.start_positions:
            render[tile] = 0.5

        return render

    def get_state(self):
        current_pos_grid = np.zeros_like(self.obstacles)
        current_pos_grid[self.current_pos] = 1
        return np.stack([current_pos_grid, self.visited_tiles, self.obstacles])

    def reset(self):
        self.visited_tiles = np.zeros_like(self.obstacles)
        self.current_pos = random.sample(self.start_positions, 1)[0]
        self.visited_tiles[self.current_pos] = 1
        self.nb_steps = 0
        return self.get_state()

    def get_reward(self, n_position):
        if self.obstacles[n_position] == 1:
            return -OBSTACLE_PUNISHMENT
        if self.visited_tiles[n_position] == 0:
            return DISCOVER_REWARD

        check = np.ones_like(self.obstacles)[n_position] = 0
        if np.sum(self.visited_tiles + self.obstacles - check) == 0:
            return COVERAGE_REWARD

        return -MOVE_PUNISHMENT

    def is_done(self, n_position):
        if self.obstacles[n_position] == 1:
            return True
        check = np.ones_like(self.obstacles)[n_position] = 0
        if np.sum(self.visited_tiles  + self.obstacles - check) == 0:
            return True

        return False

    def step(self, action):
        if action == 0:
            n_position = (max(0, self.current_pos[0] - 1), self.current_pos[1])
        elif action == 1:
            n_position = (min(self.obstacles.shape[0] - 1, self.current_pos[0] + 1), self.current_pos[1])
        elif action == 2:
            n_position = (self.current_pos[0], max(0, self.current_pos[1] - 1))
        elif action == 3:
            n_position = (self.current_pos[0], min(self.obstacles.shape[1] - 1, self.current_pos[1] + 1))
        else:
            raise Exception("ACTION NOT SUPPORTED!!")

        reward = self.get_reward(n_position)
        done = self.is_done(n_position)
        self.nb_steps += 1

        self.visited_tiles[n_position] = 1
        self.current_pos = n_position

        return self.get_state(), reward, done


if __name__ == "__main__":
    save_path = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/"
    name = "test_grid.npy"
    obstacle_grid = np.load(save_path + name)

    plt.imshow(obstacle_grid, cmap='gray')
    plt.show()

    env = SimpleEnvironment(obstacle_grid)
    print(env.current_pos)
    print(env.reset().shape)
