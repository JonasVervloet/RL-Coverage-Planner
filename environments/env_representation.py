import numpy as np
import json

from turorials.perlin_noise.obstacle_generation import flood_grid


class EnvironmentRepresentation:
    def __init__(self):
        self.obstacle_map = None
        self.terrain_map = None
        self.start_positions = None
        self.nb_free_tiles = 0

        self.dim = (8, 8)
        self.extra_spacing = (0, 0)

    def set_dimension(self, n_dim):
        self.dim = n_dim

    def set_extra_spacing(self, n_spacing):
        self.extra_spacing = n_spacing

    def get_dimension(self):
        return self.dim

    def get_obstacle_map(self, extra_spacing=False):
        if not extra_spacing:
            x_tot, y_tot = self.obstacle_map.shape
            return self.obstacle_map[
                   self.extra_spacing[0]:x_tot-self.extra_spacing[0],
                   self.extra_spacing[1]:y_tot-self.extra_spacing[1]
            ]
        else:
            return self.obstacle_map

    def get_terrain_map(self, extra_spacing=False):
        if not extra_spacing:
            x_tot, y_tot = self.obstacle_map.shape
            return self.terrain_map[
                   self.extra_spacing[0]:x_tot-self.extra_spacing[0],
                   self.extra_spacing[1]:y_tot-self.extra_spacing[1]
            ]
        else:
            return self.terrain_map

    def has_terrain_info(self):
        return self.terrain_map is not None

    def save(self, path, name):
        json_to_save = {}

        obstacle_path = f"{path}{name}_obstacle_grid.npy"
        np.save(obstacle_path, self.obstacle_map)
        json_to_save['obstacle_grid'] = obstacle_path

        json_to_save['terrain_grid'] = None
        if self.terrain_map is not None:
            terrain_path = f"{path}{name}_terrain_grid.npy"
            np.save(terrain_path, self.terrain_map)
            json_to_save['terrain_grid'] = terrain_path

        json_to_save['start_positions'] = self.start_positions
        json_to_save['nb_free_tiles'] = self.nb_free_tiles

        with open(f'{path}{name}.txt', 'w') as output_file:
            json.dump(json_to_save, output_file)

    def load(self, path, name):
        with open(f'{path}{name}.txt') as input_file:
            input_data = json.load(input_file)
            obstacle_path = input_data['obstacle_grid']
            self.obstacle_map = np.load(obstacle_path)

            terrain_path = input_data['terrain_grid']
            if terrain_path is not None:
                self.terrain_map = np.load(terrain_path)

            start_positions_array = np.array(input_data['start_positions'])
            self.start_positions = [pos for pos in zip(start_positions_array[:, 0], start_positions_array[:, 1])]
            self.nb_free_tiles = input_data['nb_free_tiles']


class GeneralEnvironmentRepresentation:
    def __init__(self, n_obstacle_map, nb_free_tiles, stat_positions,
                 n_terrain_map, extra_spacing=0):
        assert(n_obstacle_map.shape == n_terrain_map.shape)

        self.extra_spacing = extra_spacing

        self.obstacle_map = n_obstacle_map
        self.nb_free_tiles = nb_free_tiles
        self.start_positions = stat_positions

        self.terrain_map = n_terrain_map

    def get_nb_free_tiles(self):
        return self.nb_free_tiles

    def get_start_positions(self):
        return self.start_positions

    def get_obstacle_map(self, extra_spacing=0):
        assert(extra_spacing <= self.extra_spacing)

        offset = self.extra_spacing - extra_spacing
        x_tot, y_tot = self.obstacle_map.shape

        return self.obstacle_map[
               offset:x_tot - offset,
               offset:y_tot - offset
        ]

    def get_terrain_map(self, extra_spacing=0):
        assert (extra_spacing <= self.extra_spacing)

        offset = self.extra_spacing - extra_spacing
        x_tot, y_tot = self.terrain_map.shape

        return self.terrain_map[
               offset:x_tot-offset,
               offset:y_tot-offset
        ]


if __name__ == "__main__":
    save_path = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/"
    name = "test_grid.npy"
    obstacle_grid = np.load(save_path + name)

    env_repr = EnvironmentRepresentation()
    env_repr.obstacle_map = obstacle_grid

    regions = flood_grid(obstacle_grid)
    if regions[0][0] == 0:
        env_repr.start_positions = regions[0][1]
        env_repr.nb_free_tiles = len(regions[0][1]) + len(regions[0][2])
        print(regions[0][1])

    if regions[1][0] == 0:
        env_repr.start_positions = regions[1][1]
        env_repr.nb_free_tiles = len(regions[1][1]) + len(regions[1][2])
        print(regions[1][1])

    env_repr.save(save_path, "test_representation")

    env_repr2 = EnvironmentRepresentation()
    env_repr2.load(save_path, "test_representation")
    print(env_repr2.nb_free_tiles)
    print(env_repr2.start_positions)