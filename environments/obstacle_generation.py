import numpy as np
import matplotlib.pyplot as plt

from environments.noise_generation import NoiseGenerator


class ObstacleMapGenerator:
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

    def generate_boundary_map(self):
        grid = np.zeros(self.dim)
        x = np.linspace(-1.0, 1.0, num=self.dim[0])
        y = np.linspace(-1.0, 1.0, num=self.dim[1])

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                grid[i, j] = (x[i]**2 + y[j]**2)**4

        return grid

    def generate_obstacle_map(self):
        noise_map = self.noise_generator.generate_noise_map()
        noise_map += np.ones_like(noise_map)
        noise_map /= 2.0
        noise_map += self.boundary_grid

        obstacle_map = np.zeros_like(noise_map)
        obstacle_map[noise_map > (1 - self.fill_ratio)] = 1.0

        obstacle_regions, cover_regions = self.flood_map(obstacle_map)

        max_size = 0
        max_index = 0
        for i in range(len(cover_regions)):
            region = cover_regions[i]
            if (region[1] > max_size):
                max_size = region[1]
                max_index = i

        for i in range(len(cover_regions)):
            if i == max_index:
                continue

            self.change_region_value(obstacle_map, cover_regions[i], 1)

        return obstacle_map, cover_regions[i][1], cover_regions[i][2]

    @staticmethod
    def helper_functions(region):
        print(region)
        return region[1]

    @staticmethod
    def flood_map(map):
        visited_tiles = np.zeros_like(map)
        obstacle_regions = []
        cover_regions = []

        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                if visited_tiles[(i, j)] == 1:
                    continue

                region = ObstacleMapGenerator.flood_tile(map, visited_tiles, (i, j))

                region_value = region[0]
                if region_value == 0:
                    cover_regions.append(region)
                else:
                    obstacle_regions.append(region)

        return obstacle_regions, cover_regions

    @staticmethod
    def flood_tile(map, visited_tiles, tile):
        tile_value = map[tile]
        middle_tiles = []
        border_tiles = []
        queue = []

        border = ObstacleMapGenerator.grow_queue(map, visited_tiles, queue, tile, tile_value)
        if border:
            border_tiles.append(tile)
        else:
            middle_tiles.append(tile)
        visited_tiles[tile] = 1

        while len(queue) > 0:
            n_tile = queue.pop(0)
            border = ObstacleMapGenerator.grow_queue(map, visited_tiles, queue, n_tile, tile_value)
            if border:
                border_tiles.append(n_tile)
            else:
                middle_tiles.append(n_tile)
            visited_tiles[n_tile] = 1

        region_length = len(middle_tiles) + len(border_tiles)

        return tile_value, region_length, border_tiles, middle_tiles

    @staticmethod
    def grow_queue(map, visited_tiles, queue, tile, tile_value):
        border_tile = False

        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if (i == -1 or i == 1) and not j == 0:
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
    def change_region_value(map, region, n_value):
        for border_tile in region[2]:
            map[border_tile] = n_value

        for middle_tile in region[3]:
            map[middle_tile] = n_value


if __name__ == "__main__":
    obstacle_generator = ObstacleMapGenerator()
    obstacle_generator.set_dimension((32, 32))
    obstacle_generator.fill_ratio = 0.24

    # for i in range(20):
    #     map, nb_tiles, _ = obstacle_generator.generate_obstacle_map()
    #
    #     print(f"nb tiles: {nb_tiles}")
    #     plt.clf()
    #     plt.imshow(map)
    #     plt.show()

    obstacle_generator.set_dimension((8, 8))
    obstacle_generator.fill_ratio = 0.18
    obstacle_generator.set_frequency((2, 2))

    count = 0
    for i in range(20000):
        if i % 1000 == 0:
            print(i)

        _, nb_tiles, _ = obstacle_generator.generate_obstacle_map()
        if nb_tiles < 15:
            count += 1
            print(f"Small map: {nb_tiles} tiles")
            print(f"total small map count: {count}")

