import numpy as np
import matplotlib.pyplot as plt

from environments.obstacle_generation import ObstacleMapGenerator
from environments.terrain_generation import TerrainGenerator
from environments.env_representation import EnvironmentRepresentation


class EnvironmentGenerator:

    MIN_AREA = 15

    def __init__(self, requires_height_map=False):

        self.dim = (8, 8)

        self.obstacle_generator = ObstacleMapGenerator()

        self.requires_height_map = requires_height_map
        self.terrain_generator = TerrainGenerator()

    def get_dimension(self):
        return self.dim

    def set_dimension(self, n_dim):
        self.dim = n_dim
        self.obstacle_generator.set_dimension(n_dim)
        self.terrain_generator.set_dimension(n_dim)

    def set_height_frequency(self, n_freq):
        self.terrain_generator.set_frequency(n_freq)

    def set_obstacle_frequency(self, n_freq):
        self.obstacle_generator.set_frequency(n_freq)

    def set_fill_ration(self, n_ratio):
        self.obstacle_generator.fill_ratio = n_ratio

    def generate_environment(self, extra_spacing=False):
        env_representation = EnvironmentRepresentation()
        env_representation.set_dimension(self.dim)
        extra_x = 0
        extra_y = 0
        if extra_spacing:
            extra_x = (self.dim[0] - 1) // 5
            extra_y = (self.dim[1] - 1) // 5
        extra_dim_x = self.dim[0] + 2 * extra_x
        extra_dim_y = self.dim[1] + 2 * extra_y
        env_representation.set_extra_spacing((extra_x, extra_y))

        area = 0
        while area < self.MIN_AREA:
            obstacle_map, nb_tiles, start_positions = self.obstacle_generator.generate_obstacle_map()
            extra_obstacle_map = np.ones((extra_dim_x, extra_dim_y))
            extra_obstacle_map[extra_x:extra_dim_x-extra_x, extra_y:extra_dim_y-extra_y] = obstacle_map

            env_representation.obstacle_map = extra_obstacle_map
            env_representation.nb_free_tiles = nb_tiles
            env_representation.start_positions = start_positions

            area = nb_tiles

        if self.requires_height_map:
            self.terrain_generator.set_dimension((extra_dim_x, extra_dim_y))
            env_representation.terrain_map = self.terrain_generator.generate_terrain_map()

        return env_representation


class SingleEnvironmentGenerator:
    def __init__(self, load_info):
        self.env_representation = EnvironmentRepresentation()
        self.env_representation.load(load_info[0], load_info[1])

    def generate_environment(self):
        return self.env_representation

    def get_dimension(self):
        return self.env_representation.get_dimension()


if __name__ == "__main__":
    env_generator = EnvironmentGenerator(requires_height_map=True)
    env_generator.set_dimension((64, 64))
    env_generator.set_height_frequency((8, 2))
    env_generator.set_fill_ration(0.25)

    fig, axs = plt.subplots(2, 2)

    env_repr = env_generator.generate_environment()
    print(env_repr.nb_free_tiles)
    axs[0][0].imshow(env_repr.terrain_map)
    axs[0][1].imshow(env_repr.obstacle_map)

    print(np.max(env_repr.terrain_map))
    print(np.min(env_repr.terrain_map))

    env_repr = env_generator.generate_environment()
    print(env_repr.nb_free_tiles)
    axs[1][0].imshow(env_repr.terrain_map)
    axs[1][1].imshow(env_repr.obstacle_map)

    print(np.max(env_repr.terrain_map))
    print(np.min(env_repr.terrain_map))

    plt.show()

