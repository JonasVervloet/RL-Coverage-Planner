import numpy as np
import torch
import pygame

from environments.simple_environment import SimpleEnvironment
from deep_q_learning.deep_q_network import NetworkGenerator
from deep_q_learning.deep_q_agent import DeepQAgent

FPS = 2
DIMENSION = 800

EPISODE_NB = 1000
LOAD_PATH = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/"

COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "maroon4": (139, 28, 98, 255),
    "magenta": (255,0,230),
    "forest_green": (0,50,0),
    "tan": (230,220,170),
    "coffee_brown": (200,190,140),
    "moon_glow": (235,245,255),
    "red": (255, 0, 0)
}

env_name = "test_grid.npy"
obstacle_grid = np.load(LOAD_PATH + env_name)
env = SimpleEnvironment(obstacle_grid)

network_generator = NetworkGenerator()
agent = DeepQAgent(network_generator, batch_size=32)
agent.load(LOAD_PATH, EPISODE_NB)
agent.evaluate()

pygame.init()
screen = pygame.display.set_mode((DIMENSION, DIMENSION))
screen.fill(COLORS["white"])
clock = pygame.time.Clock()
font = pygame.font.Font("freesansbold.ttf", 32)
running = True

done = False
current_state = env.reset()
total_reward = 0.0
reward = 0.0


def state_to_image(state):
    _, m, n = state.shape

    unscaled_img = np.zeros((m, n, 3))

    obstacles = state[2].astype(bool)
    visited_tiles = state[1].astype(bool)
    current_position = state[0].astype(bool)

    unscaled_img[obstacles] = np.array(COLORS['coffee_brown'])
    unscaled_img[visited_tiles] = np.array(COLORS['forest_green'])
    unscaled_img[current_position] = np.array(COLORS['white'])
    unscaled_img[np.multiply(current_position, obstacles)] =np.array(COLORS['red'])

    scaled_img = np.repeat(unscaled_img, 45, axis=0)
    scaled_img = np.repeat(scaled_img, 45, axis=1)

    return scaled_img


while running:

    screen.fill(COLORS["white"])

    enter_pressed = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                enter_pressed = True

    img = state_to_image(current_state)
    state_surface = pygame.surfarray.make_surface(img)
    screen.blit(state_surface, (40, 40))

    text = font.render(f"REWARD: {reward} --- TOTAL REWARD: {round(total_reward, 2)}",
                       True, COLORS['black'], COLORS['white'])
    screen.blit(text, (100, 25))

    if enter_pressed:
        n_reward = 0.0
        n_state = env.reset()
        total_reward = 0.0
        done = False

    elif not done:
        action = agent.select_action(torch.tensor(current_state, dtype=torch.float))
        n_state, n_reward, done = env.step(action)

    else:
        print("waiting for enter")
        n_reward = 0.0
        n_state = current_state

    current_state = n_state
    reward = n_reward
    total_reward += reward


    pygame.display.update()
    clock.tick(FPS)

pygame.quit()





