import numpy as np
import random
import matplotlib.pyplot as plt

from turorials.perlin_noise.obstacle_generation import flood_grid

OBSTACLE_PUNISHMENT = -1
DISCOVER_REWARD = 1
MOVE_PUNISHMENT = 0.2
COVERAGE_REWARD = 10

class Environment:
    def __init__(self, obstacles, start_positions):
        self.obstacles = obstacles
        self.start_positions = start_positions
        self.visited_tiles = np.zeros_like(self.obstacles)
        self.current_pos = random.sample(self.start_positions, 1)[0]

    def get_state(self):
        return self.current_pos, self.visited_tiles, self.obstacles

    def reset(self):
        self.visited_tiles = np.zeros_like(self.obstacles)
        self.current_pos = self.start_positions[np.random.randint(len(self.start_positions))]
        return self.get_state()

    def get_reward(self):
        if self.obstacles[self.current_pos] == 1:
            return OBSTACLE_PUNISHMENT
        if self.visited_tiles[self.current_pos] == 0:
            return DISCOVER_REWARD

        if np.sum(self.visited_tiles  + self.obstacles - np.ones_like(self.obstacles)) == 0:
            return COVERAGE_REWARD

        return MOVE_PUNISHMENT

    def is_done(self):
        if self.obstacles[self.current_pos] == 1:
            return True
        if np.sum(self.visited_tiles  + self.obstacles - np.ones_like(self.obstacles)) == 0:
            return True

        return False

    def step(self, action):
        self.visited_tiles[self.current_pos] = 1

        if action == 0:
            self.current_pos = (max(0, self.current_pos[0] - 1), self.current_pos[1])
        elif action == 1:
            self.current_pos = (min(self.obstacles.shape[0] - 1, self.current_pos[0] + 1), self.current_pos[1])
        elif action == 2:
            self.current_pos = (self.current_pos[0], max(0, self.current_pos[1] - 1))
        elif action == 3:
            self.current_pos = (self.current_pos[0], min(self.obstacles.shape[1] - 1, self.current_pos[1] + 1))

        return self.get_state(), self.get_reward(), self.is_done()


if __name__ == "__main__":
    save_path = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/"
    name = "test_grid.npy"
    obstacle_grid = np.load(save_path + name)

    regions = flood_grid(obstacle_grid)

    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(obstacle_grid, cmap="gray")

    masked_grid = np.copy(obstacle_grid)
    for region in regions:
        region_value = region[0]
        middle_tiles = region[2]

        middle_color = 0.25 if region_value else 0.75
        for tile in middle_tiles:
            masked_grid[tile] = middle_color

    axs[1].imshow(masked_grid, cmap="gray")

    plt.show()

