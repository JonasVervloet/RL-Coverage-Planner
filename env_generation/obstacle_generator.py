import numpy as np
import matplotlib.pyplot as plt

from env_generation.height_generator import Height_Generator

BOUNDARY_SIZE = 1
SMOOTH_SIZE = 1
RULE_THRESHOLD = 4


class ObstacleGenerator:
    def __init__(self):
        self.dim = (16, 16)
        self.fill_ratio = 0.35
        self.nb_smoothing = 2

    def generate_grid(self):
        simple_grid = self.generate_simple_grid()
        return simple_grid

    def generate_simple_grid(self):
        random_grid = np.random.rand(self.dim[0], self.dim[1])
        clipped_grid = np.zeros(self.dim)
        clipped_grid[random_grid < self.fill_ratio] = 1

        clipped_grid[:BOUNDARY_SIZE, :] = 1
        clipped_grid[:, :BOUNDARY_SIZE] = 1
        clipped_grid[-BOUNDARY_SIZE:, :] = 1
        clipped_grid[:, -BOUNDARY_SIZE:] = 1

        for i in range(self.nb_smoothing):
            self.smooth_grid(clipped_grid)

        return clipped_grid

    def smooth_grid(self, grid):
        bound_x_low = BOUNDARY_SIZE
        bound_y_low = BOUNDARY_SIZE
        bound_x_high = self.dim[0] - BOUNDARY_SIZE
        bound_y_high = self.dim[1] - BOUNDARY_SIZE

        neighbor_sum = np.zeros((self.dim[0] - 2 * BOUNDARY_SIZE, self.dim[1] - 2 * BOUNDARY_SIZE))
        for delta_x in range(-SMOOTH_SIZE, SMOOTH_SIZE + 1):
            for delta_y in range(-SMOOTH_SIZE, SMOOTH_SIZE + 1):
                if delta_x == 0 and delta_y == 0:
                    continue

                neighbor_sum += grid[
                                bound_x_low + delta_x: bound_x_high + delta_x,
                                bound_y_low + delta_y: bound_y_high + delta_y
                                ]

        grid_middle = grid[bound_x_low:bound_x_high, bound_y_low:bound_y_high]
        grid_middle[neighbor_sum > RULE_THRESHOLD] = 1
        grid_middle[neighbor_sum < RULE_THRESHOLD] = 0
        grid[bound_x_low:bound_x_high, bound_y_low:bound_y_high] = grid_middle


class ObstacleGenerator2:
    def __init__(self):
        self.dim = (16, 16)
        self.fill_ratio = 0.35
        self.noise_gen = Height_Generator()
        self.boundary_grid = self.generate_boundary_grid()
        self.boundaries = False

    def generate_boundary_grid(self):
        grid = np.zeros(self.dim)
        x = np.linspace(-1.0, 1.0, num=self.dim[0])
        y = np.linspace(-1.0, 1.0, num=self.dim[1])

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                grid[i, j] = (x[i]**2 + y[j]**2)**4

        return grid

    def set_dim(self, n_dim):
        self.noise_gen.dim = n_dim
        self.dim = n_dim
        self.boundary_grid = self.generate_boundary_grid()


    def set_freq(self, n_freq):
        self.noise_gen.res = n_freq

    def generate_grid(self):
        noise_grid = self.noise_gen.generate_grid()
        noise_grid += np.ones_like(noise_grid)
        noise_grid /= 2.0

        if self.boundaries:
            grid = noise_grid + self.boundary_grid
            obstacle_grid = np.zeros_like(grid)
            obstacle_grid[grid > (1 - self.fill_ratio)] = 1.0

            return obstacle_grid

        else:
            noise_grid[0, :] = 1.0
            noise_grid[:, 0] = 1.0
            noise_grid[self.dim[0] - 1, :] = 1.0
            noise_grid[:, self.dim[1] - 1] = 1.0

            obstacle_grid = np.zeros_like(noise_grid)
            obstacle_grid[noise_grid > (1-self.fill_ratio)] = 1.0

            return obstacle_grid




if __name__ == "__main__":
    gen = ObstacleGenerator2()
    grid = gen.generate_boundary_grid()

    mask = np.zeros_like(grid)
    mask[grid > 1.0] = 1.0

    plt.imshow(mask, cmap='gray')
    plt.show()

    grid2 = gen.generate_grid()
    plt.imshow(grid2, cmap='gray')
    plt.show()
