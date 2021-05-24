import numpy as np
import matplotlib.pyplot as plt

from environments.obstacle_generation import ObstacleMapGenerator
from environments.terrain_generation import TerrainGenerator
from environments.env_representation import EnvironmentRepresentation, GeneralEnvironmentRepresentation


class EnvironmentGenerator:

    MIN_AREA = 15

    def __init__(self, requires_height_map=False):

        self.dim = (8, 8)

        self.obstacle_generator = ObstacleMapGenerator()

        self.requires_height_map = requires_height_map
        self.terrain_generator = TerrainGenerator()

        self.agent_radius = 1

    def get_dimension(self):
        return self.dim

    def set_agent_radius(self, n_radius):
        self.agent_radius = n_radius

    def set_dimension(self, n_dim):
        self.dim = n_dim
        self.obstacle_generator.set_dimension(n_dim)
        self.terrain_generator.set_dimension(n_dim)

    def set_height_frequency(self, n_freq):
        self.terrain_generator.set_frequency(n_freq)

    def set_obstacle_frequency(self, n_freq):
        self.obstacle_generator.set_frequency(n_freq)

    def set_fill_ratio(self, n_ratio):
        self.obstacle_generator.set_fill_ratio(n_ratio)

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
            obstacle_map, nb_tiles, start_positions = self.obstacle_generator.generate_obstacle_map(
                agent_radius=self.agent_radius
            )
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


class GeneralEnvironmentGenerator:
    def __init__(self, n_dim):
        self.dim = (8, 8)
        self.set_dimension(n_dim)

        self.obstacle_generator = ObstacleMapGenerator()
        self.terrain_generator = TerrainGenerator()

        self.loaded_representation = None

    def get_dimension(self):
        return self.dim

    def set_dimension(self, n_dim):
        assert (n_dim[0] == n_dim[1])

        self.dim = n_dim

    def set_obstacle_frequency(self, n_freq):
        self.obstacle_generator.set_frequency(n_freq)

    def set_fill_ratio(self, n_ratio):
        self.obstacle_generator.fill_ratio = n_ratio

    def set_height_frequency(self, n_freq):
        self.terrain_generator.set_frequency(n_freq)

    def load_env_representation(self, env_repr):
        self.loaded_representation = env_repr
        dim_x, dim_y = env_repr.get_obstacle_map().shape
        self.set_dimension((dim_x, dim_y))

    def get_area_minimum(self):
        return self.dim[0] / 4

    def get_interval_and_count(self, extra_spacing):
        freq_x, freq_y = self.terrain_generator.get_frequency()
        min_freq = min(freq_x, freq_y)

        interval = self.dim[0] // min_freq

        count = (2 * extra_spacing) // interval
        if not (2 * extra_spacing) % interval == 0:
            count += 1

        return interval, count

    def get_final_extra_space(self, extra_spacing):
        interval, count = self.get_interval_and_count(extra_spacing)

        return (count * interval) // 2

    def get_final_frequency_terrain(self, extra_spacing):
        interval, count = self.get_interval_and_count(extra_spacing)
        freq_x, freq_y = self.terrain_generator.get_frequency()

        interval_x = self.dim[0] // freq_x
        multiplier_x = interval // interval_x

        interval_y = self.dim[0] // freq_y
        multiplier_y = interval // interval_y

        n_freq = freq_x + count * multiplier_x, freq_y + count * multiplier_y

        return freq_x + count * multiplier_x, freq_y + count * multiplier_y

    def generate_environment(self, extra_spacing=0, agent_radius=1):
        # loaded representation
        if self.loaded_representation is not None:
            return self.loaded_representation

        # compute extra spacing so that the terrain frequency holds
        x_dim, y_dim = self.dim
        final_extra_spacing = self.get_final_extra_space(extra_spacing)

        # create an obstacle map with enough extra spacing
        obstacle_map = None
        nb_tiles = 0
        start_positions = []
        self.obstacle_generator.set_dimension(self.dim)

        while nb_tiles < self.get_area_minimum():
            obstacle_map, nb_tiles, start_positions = self.obstacle_generator.generate_obstacle_map(agent_radius)

        extra_dim_x, extra_dim_y = x_dim + 2 * final_extra_spacing, y_dim + 2 * final_extra_spacing
        extra_map = np.ones((extra_dim_x, extra_dim_y))
        extra_map[
            final_extra_spacing : extra_dim_x - final_extra_spacing,
            final_extra_spacing : extra_dim_y - final_extra_spacing
        ] = obstacle_map

        # create a terrain map with enough extra spacing
        t_freq = self.terrain_generator.get_frequency()
        self.terrain_generator.set_frequency(self.get_final_frequency_terrain(extra_spacing))

        self.terrain_generator.set_dimension((x_dim + 2*final_extra_spacing, y_dim + 2*final_extra_spacing))
        terrain_map = self.terrain_generator.generate_terrain_map()

        self.terrain_generator.set_frequency(t_freq)

        # create environment representation
        env_representation = GeneralEnvironmentRepresentation(
            extra_map,
            nb_tiles,
            start_positions,
            terrain_map,
            final_extra_spacing
        )

        return env_representation


class SingleEnvironmentGenerator:
    def __init__(self, load_info):
        self.env_representation = EnvironmentRepresentation()
        self.env_representation.load(load_info[0], load_info[1])

    def generate_environment(self, extra_spacing=False):
        return self.env_representation

    def get_dimension(self):
        return self.env_representation.get_dimension()


if __name__ == "__main__":
    env_generator = EnvironmentGenerator(requires_height_map=True)
    env_generator.set_dimension((64, 64))
    env_generator.set_height_frequency((8, 2))
    env_generator.set_fill_ratio(0.25)

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

