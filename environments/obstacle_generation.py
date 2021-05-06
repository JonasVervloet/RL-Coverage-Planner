import numpy as np
import matplotlib.pyplot as plt

from environments.noise_generation import NoiseGenerator


class ObstacleMapGenerator:

    CORNERS_INCL = [
        True,
        False,
        False,
        True,
        False,
        False
    ]

    def __init__(self):
        self.dim = (16, 16)
        self.fill_ratio = 0.14

        self.noise_generator = NoiseGenerator()
        self.boundary_grid = self.generate_boundary_map()

        self.noise_generator.dim = (16, 16)

    def set_dimension(self, n_dim):
        self.dim = n_dim
        self.noise_generator.dim = n_dim
        self.boundary_grid = self.generate_boundary_map()

    def set_frequency(self, n_freq):
        self.noise_generator.res = n_freq

    def set_fill_ratio(self, n_ratio):
        self.fill_ratio = n_ratio

    def generate_boundary_map(self):
        grid = np.zeros(self.dim)
        x = np.linspace(-1.0, 1.0, num=self.dim[0])
        y = np.linspace(-1.0, 1.0, num=self.dim[1])

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                grid[i, j] = (x[i]**2 + y[j]**2)**4

        return grid

    def generate_obstacle_map(self, agent_radius=1):
        noise_map = self.noise_generator.generate_noise_map()
        noise_map += np.ones_like(noise_map)
        noise_map /= 2.0
        noise_map += self.boundary_grid

        obstacle_map = np.zeros_like(noise_map)
        obstacle_map[noise_map > (1 - self.fill_ratio)] = 1.0

        obstacle_regions, cover_regions = self.flood_map(obstacle_map)

        max_size = 0
        max_index = 0
        for idx, region in enumerate(cover_regions):
            if region[1] > max_size:
                max_size = region[1]
                max_index = idx

        max_region = cover_regions.pop(max_index)
        for region in cover_regions:
            self.change_region_value(obstacle_map, region, 1)

        obstacle_map, start_positions, nb_free_tiles = ObstacleMapGenerator.adapt_obstacle_map(
            obstacle_map, max_region[1], agent_radius
        )

        return obstacle_map, nb_free_tiles, start_positions

    @staticmethod
    def flood_map(map, corners_incl=False):
        visited_tiles = np.zeros_like(map)
        obstacle_regions = []
        cover_regions = []

        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                if visited_tiles[(i, j)] == 1:
                    continue

                region = ObstacleMapGenerator.flood_tile(map, visited_tiles, (i, j), corners_incl)

                region_value = region[0]
                if region_value == 0:
                    cover_regions.append(region)
                else:
                    obstacle_regions.append(region)

        return obstacle_regions, cover_regions

    @staticmethod
    def flood_tile(map, visited_tiles, tile, corners_incl=False):
        tile_value = map[tile]
        middle_tiles = []
        border_tiles = []
        queue = []

        border = ObstacleMapGenerator.grow_queue(map, visited_tiles, queue, tile, tile_value, corners_incl)
        if border:
            border_tiles.append(tile)
        else:
            middle_tiles.append(tile)
        visited_tiles[tile] = 1

        while len(queue) > 0:
            n_tile = queue.pop(0)
            border = ObstacleMapGenerator.grow_queue(map, visited_tiles, queue, n_tile, tile_value, corners_incl)
            if border:
                border_tiles.append(n_tile)
            else:
                middle_tiles.append(n_tile)
            visited_tiles[n_tile] = 1

        region_length = len(middle_tiles) + len(border_tiles)

        return tile_value, region_length, border_tiles, middle_tiles

    @staticmethod
    def grow_queue(map, visited_tiles, queue, tile, tile_value, corners_incl=False):
        border_tile = False

        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if not corners_incl and (i == -1 or i == 1) and not j == 0:
                    continue

                n_tile_x = tile[0] + i
                n_tile_y = tile[1] + j

                if n_tile_x < 0 or n_tile_y < 0:
                    continue

                if n_tile_x == map.shape[0] or n_tile_y == map.shape[1]:
                    continue

                if map[n_tile_x, n_tile_y] == tile_value:
                    if visited_tiles[(n_tile_x, n_tile_y)] == 1:
                        continue

                    if (n_tile_x, n_tile_y) in queue:
                        continue

                    queue.append((n_tile_x, n_tile_y))

                else:
                    border_tile = True

        return border_tile

    @staticmethod
    def change_region_value(map, region, n_value, border_only=False):
        for border_tile in region[2]:
            map[border_tile] = n_value

        if not border_only:
            for middle_tile in region[3]:
                map[middle_tile] = n_value

    @staticmethod
    def adapt_obstacle_map(map, nb_free_tiles, agent_radius=1):
        new_map = np.copy(map)

        start_positions, start_positions_list = ObstacleMapGenerator.get_start_positions(map, agent_radius)

        x = np.arange(0, map.shape[0])
        y = np.arange(0, map.shape[1])

        xx, yy = np.meshgrid(x, y)
        xx, yy = np.transpose(xx), np.transpose(yy)

        mask = ObstacleMapGenerator.get_radius_map(agent_radius)
        offset  = agent_radius - 1

        for i, positions in enumerate(start_positions_list):
            position_set = set(positions)

            for position in start_positions:
                local_xx = xx[
                           position[0] - offset: position[0] + offset + 1,
                           position[1] - offset: position[1] + offset + 1
                ]
                local_xx = local_xx[mask]
                local_yy = yy[
                           position[0] - offset: position[0] + offset + 1,
                           position[1] - offset: position[1] + offset + 1
                ]
                local_yy = local_yy[mask]
                point_set = set(zip(local_xx, local_yy))

                position_set -= point_set

            if len(position_set) == 0:
                break
            else:
                nb_free_tiles -= len(position_set)
                for position in position_set:
                    new_map[position] = 1

        return new_map, start_positions, nb_free_tiles

    @staticmethod
    def get_start_positions(map, agent_radius=1):
        obstacle_map_copy = np.copy(map)
        start_positions_list = []
        start_positions = []
        for i in range(agent_radius):
            positions = []
            corners_incl = ObstacleMapGenerator.CORNERS_INCL[i] if not i == agent_radius - 1 else False
            obstacle_regions_copy, cover_regions_copy = ObstacleMapGenerator.flood_map(
                obstacle_map_copy, corners_incl
            )
            for region in cover_regions_copy:
                ObstacleMapGenerator.change_region_value(
                    obstacle_map_copy, region, 1.0, border_only=True
                )
                positions += region[2]

            start_positions_list.append(positions)
            start_positions = positions

        return start_positions, start_positions_list

    @staticmethod
    def get_radius_map(agent_radius):
        agent_size = 1 + (agent_radius - 1) * 2
        x = np.linspace(0.5, agent_size - 0.5, agent_size) - (agent_size/2)
        yy, xx = np.meshgrid(x, x)

        dists = np.sqrt(xx**2 + yy**2)
        mask = dists <= (agent_size/2)

        return mask


if __name__ == "__main__":
    obstacle_generator = ObstacleMapGenerator()
    obstacle_generator.set_dimension((32, 32))
    obstacle_generator.fill_ratio = 0.22

    obst_map, nb_tiles, start_positions = obstacle_generator.generate_obstacle_map(agent_radius=5)

    xs, ys = zip(*start_positions)
    map = np.copy(obst_map)
    map[xs, ys] = 0.5
    plt.subplot(131)
    plt.imshow(map)

    map1 = np.copy(obst_map)
    for radius in range(1, 7):
        positions, _ = ObstacleMapGenerator.get_start_positions(
            obst_map, radius
        )
        print(f"radius {radius}: {len(positions)} start positions")
        if len(positions) > 0:
            xs, ys = zip(*positions)
            map1[xs, ys] = 1 - radius * (1 / 7)

    print()

    plt.subplot(132)
    plt.imshow(map1)

    map2 = np.copy(obst_map)
    for radius in range(6, 0, -1):
        positions, _ = ObstacleMapGenerator.get_start_positions(
            obst_map, radius
        )
        print(f"radius {radius}: {len(positions)} start positions")
        if len(positions) > 0:
            xs, ys = zip(*positions)
            map2[xs, ys] = 1 - radius * (1 / 7)

    plt.subplot(133)
    plt.imshow(map2)

    plt.show()

