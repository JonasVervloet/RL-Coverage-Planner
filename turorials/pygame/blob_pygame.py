import pygame
import numpy as np

from turorials.pygame.environment import Environment

FPS = 2

NB_TILES = 4
TILE_WIDTH = 100
TILE_BORDER = 3
EXTRA_SPACING = 100

FIELD_OF_VIEW = 1

DIMENSION = NB_TILES * (TILE_WIDTH + TILE_BORDER) + TILE_BORDER + EXTRA_SPACING

BORDER_COLOR = "tan"
TILE_COLOR = "moon_glow"
VISITED_COLOR = "forest_green"
CURRENT_COLOR = "white"
OBSTACLE_COLOR = "coffee_brown"
VIEW_COLOR = "magenta"

OBSTACLES = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
])

START_POS = [
    (0, 0), (0, 1), (0, 2)
]

current_pos = (1, 2)

env = Environment(OBSTACLES, START_POS)


COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "maroon4": (139, 28, 98, 255),
    "magenta": (255,0,230),
    "forest_green": (0,50,0),
    "tan": (230,220,170),
    "coffee_brown": (200,190,140),
    "moon_glow": (235,245,255)
}

pygame.init()
screen = pygame.display.set_mode((DIMENSION, DIMENSION))
screen.fill(COLORS["white"])
clock = pygame.time.Clock()


def draw_tiles(surface, current_pos, visited_tiles, obstacles):
    offset = EXTRA_SPACING / 2
    for i in range(NB_TILES):
        x = offset + i * (TILE_WIDTH + TILE_BORDER)

        for j in range(NB_TILES):
            y = offset + j * (TILE_WIDTH + TILE_BORDER)

            color = COLORS[TILE_COLOR]
            if (i, j) == current_pos:
                color = COLORS[CURRENT_COLOR]
            elif visited_tiles[i, j] == 1:
                color = COLORS[VISITED_COLOR]
            elif obstacles[i, j] == 1:
                color = COLORS[OBSTACLE_COLOR]

            tile_square = pygame.Rect(y + TILE_BORDER, x + TILE_BORDER, TILE_WIDTH, TILE_WIDTH)
            pygame.draw.rect(surface, color, tile_square)


def draw_fov(surface, current_pos, fov):
    fov_x = max(offset + (TILE_WIDTH + TILE_BORDER) * (current_pos[0] - fov), offset)
    fov_y = max(offset + (TILE_WIDTH + TILE_BORDER) * (current_pos[1] - fov), offset)
    width = (fov * 2 + 1) * (TILE_WIDTH + TILE_BORDER) + TILE_BORDER
    height = (fov * 2 + 1) * (TILE_WIDTH + TILE_BORDER) + TILE_BORDER
    fov_square = pygame.Rect(fov_y, fov_x, width, height)
    pygame.draw.rect(surface, COLORS[VIEW_COLOR], fov_square)

    border_square = pygame.Rect(fov_y + TILE_BORDER, fov_x + TILE_BORDER, width - 2*TILE_BORDER, height - 2*TILE_BORDER)
    pygame.draw.rect(surface, COLORS[BORDER_COLOR], border_square)


def draw_state(surface, state):
    offset = EXTRA_SPACING / 2
    width = (TILE_WIDTH + TILE_BORDER) * NB_TILES + TILE_BORDER
    height = (TILE_WIDTH + TILE_BORDER) * NB_TILES + TILE_BORDER
    border_square = pygame.Rect(offset, offset, width, height)
    pygame.draw.rect(screen, COLORS[BORDER_COLOR], border_square)

    current_pos, visited_tiles, obstacles = state
    draw_tiles(surface, current_pos, visited_tiles, obstacles)



running = True
done = False

state = env.reset()

while running:
    enter_pressed = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                enter_pressed = True

    if not done:
        action = np.random.randint(4)
        state, _, done = env.step(action)
    else:
        print("waiting for enter")
        if enter_pressed:
            state = env.reset()
            done = False

    draw_state(screen, state)

    pygame.display.update()
    clock.tick(FPS)

pygame.quit()