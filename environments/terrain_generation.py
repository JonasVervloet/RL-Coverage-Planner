import numpy as np
import matplotlib.pyplot as plt

from environments.noise_generation import NoiseGenerator


class TerrainGenerator:
    def __init__(self):
        self.noise_generator = NoiseGenerator()
        self.noise_generator.dim = (16, 16)

    def set_dimension(self, n_dim):
        self.noise_generator.dim = n_dim

    def set_frequency(self, n_freq):
        self.noise_generator.res = n_freq

    def generate_terrain_map(self):
        noise_map = self.noise_generator.generate_noise_map()

        min_value = np.min(noise_map)
        max_value = np.max(noise_map)

        terrain_map = (noise_map - min_value) / (max_value - min_value)

        return terrain_map


if __name__ == "__main__":
    terrain_generator = TerrainGenerator()
    terrain_generator.set_dimension((8, 8))
    terrain_generator.set_frequency((1,1))

    for i in range(10):
        terrain_map = terrain_generator.generate_terrain_map()
        print(np.max(terrain_map))
        print(np.min(terrain_map))

        plt.imshow(terrain_map, cmap='gray')
        plt.show()