import matplotlib.pyplot as plt
import numpy as np

from turorials.perlin_noise.open_simplex import generate_noise_map

# DIM = (8, 8)
DIM = (16, 16)
# DIM = (32, 32)

RES = (5, 5)

# FILL_RATIO = 0.30
FILL_RATIO = 0.35
# FILL_RATIO = 0.40

BOUNDARY_SIZE = 1

# NB_SMOOTHING = 2
NB_SMOOTHING = 2
# NB_SMOOTHING = 3

SMOOTH_SIZE = 1
RULE_THRESHOLD = 4

NB_GRIDS = 10


def generate_obstacle_grid():
    random_grid = np.random.rand(DIM[0], DIM[1])
    clipped_grid = np.zeros(DIM)
    clipped_grid[random_grid < FILL_RATIO] = 1

    clipped_grid[:BOUNDARY_SIZE, :] = 1
    clipped_grid[:, :BOUNDARY_SIZE] = 1
    clipped_grid[-BOUNDARY_SIZE:, :] = 1
    clipped_grid[:, -BOUNDARY_SIZE:] = 1

    for i in range(NB_SMOOTHING):
        bound_x_low = BOUNDARY_SIZE
        bound_y_low = BOUNDARY_SIZE
        bound_x_high = DIM[0] - BOUNDARY_SIZE
        bound_y_high = DIM[1] - BOUNDARY_SIZE

        neighbor_sum = np.zeros((DIM[0] - 2 * BOUNDARY_SIZE, DIM[1] - 2 * BOUNDARY_SIZE))
        for delta_x in range(-SMOOTH_SIZE, SMOOTH_SIZE + 1):
            for delta_y in range(-SMOOTH_SIZE, SMOOTH_SIZE + 1):
                if delta_x == 0 and delta_y == 0:
                    continue

                neighbor_sum += clipped_grid[
                                bound_x_low + delta_x: bound_x_high + delta_x,
                                bound_y_low + delta_y: bound_y_high + delta_y
                                ]

        clipped_grid_middle = clipped_grid[bound_x_low:bound_x_high, bound_y_low:bound_y_high]
        clipped_grid_middle[neighbor_sum > RULE_THRESHOLD] = 1
        clipped_grid_middle[neighbor_sum < RULE_THRESHOLD] = 0
        clipped_grid[bound_x_low:bound_x_high, bound_y_low:bound_y_high] = clipped_grid_middle

    return clipped_grid


def flood_grid(grid):
    visited_tiles = np.zeros(grid.shape)
    regions = []

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if visited_tiles[(i, j)] == 1:
                continue

            region = flood_tile(grid, visited_tiles, (i, j))
            print(region)
            regions.append(region)

    return regions


def flood_tile(grid, visited_tiles, tile):
    tile_value = grid[tile]
    middle_tiles = []
    border_tiles = []
    queue = []

    border = grow_queue(grid, visited_tiles, queue, tile, tile_value)
    if border:
        border_tiles.append(tile)
    else:
        middle_tiles.append(tile)
    visited_tiles[tile] = 1

    while len(queue) != 0:
        n_tile = queue.pop(0)
        border = grow_queue(grid, visited_tiles, queue, n_tile, tile_value)
        if border:
            border_tiles.append(n_tile)
        else:
            middle_tiles.append(n_tile)
        visited_tiles[n_tile] = 1

    return [tile_value, border_tiles, middle_tiles]


def grow_queue(grid, visited_tiles, queue, tile, tile_value):
    border_tile = False
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue

            n_tile_x = tile[0] + i
            n_tile_y = tile[1] + j

            if n_tile_x < 0 or n_tile_y < 0:
                continue

            if n_tile_x == grid.shape[0] or n_tile_y == grid.shape[1]:
                continue

            if grid[n_tile_x, n_tile_y] == tile_value:
                if visited_tiles[(n_tile_x, n_tile_y)] == 1:
                    continue

                if (n_tile_x, n_tile_y) in queue:
                    continue

                queue.append((n_tile_x, n_tile_y))
            else:
                border_tile = True

    return border_tile





if __name__ == "__main__":
    for i in range(NB_GRIDS):
        print(f"GRID NB {i}")
        grid = generate_obstacle_grid()

        plt.imshow(grid, cmap="gray")
        plt.show()

    regions = flood_grid(grid)
    print(regions)

    masked_grid = np.copy(grid)

    for region in regions:
        region_value = region[0]
        middle_tiles = region[2]

        middle_color = 0.25 if region_value == 0 else 0.75
        for tile in middle_tiles:
            masked_grid[tile] = middle_color


    fig, axs = plt.subplots(2)

    axs[0].imshow(grid, cmap='gray')
    axs[1].imshow(masked_grid, cmap='gray')
    plt.show()