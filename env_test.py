import numpy as np
import math
import pygame

from environments.general_environment import GeneralEnvironment
from environments.env_generation import GeneralEnvironmentGenerator


def state_to_surface(maps, info, nb_repeats):
    dim_x, dim_y = maps["obstacle_map"].shape

    unscaled_img = np.zeros((dim_x, dim_y, 3))
    unscaled_img[maps["obstacle_map"].astype(bool)] = np.array((200, 190, 140))
    unscaled_img[maps["coverage_map"].astype(bool)] = np.array((0, 50, 0))

    curr_x, curr_y = info["current_position"]

    if "agent_size" in info:
        agent_size = info["agent_size"]
        mask = GeneralEnvironment.get_radius_map(agent_size)
        offset = agent_size // 2
        local_img = unscaled_img[
                    curr_x - offset: curr_x + offset + 1,
                    curr_y - offset: curr_y + offset + 1
        ]
        local_img[mask] = local_img[mask] * 0.5 + 0.5 * np.array((205, 92, 92))

    if maps["obstacle_map"][info["current_position"]]:
        unscaled_img[info["current_position"]] = np.array((255, 0, 0))
    else:
        unscaled_img[info["current_position"]] = np.array((255, 255, 255))

    # test points to verify orientation
    unscaled_img[5, 0, :] = np.array((255, 0, 0))
    unscaled_img[0, 5, :] = np.array((0, 255, 0))

    scaled_img = np.repeat(unscaled_img, nb_repeats[0], axis=0)
    scaled_img = np.repeat(scaled_img, nb_repeats[1], axis=1)

    surface = pygame.surfarray.make_surface(scaled_img)

    if "fov" in info:
        offset = info["fov"] // 2
        min_x, max_x = (curr_x - offset) * nb_repeats[0], (curr_x + offset + 1) * nb_repeats[0]
        min_y, max_y = (curr_y - offset) * nb_repeats[1], (curr_y + offset + 1) * nb_repeats[1]

        points = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])

        if "angle" in info:
            angle = info["angle"]

            center_x = (curr_x + 0.5) * nb_repeats[0]
            center_y = (curr_y + 0.5) * nb_repeats[1]

            rel_points_x = points[:, 0] - center_x
            rel_points_y = points[:, 1] - center_y

            points_x = math.cos(angle) * rel_points_x - math.sin(angle) * rel_points_y + center_x
            points_y = math.sin(angle) * rel_points_x + math.cos(angle) * rel_points_y + center_y

            points = np.transpose(
                np.stack([points_x, points_y])
            )

        pygame.draw.lines(surface, color=(235, 245, 255),
                           closed=True, points=points)

    if "angle" in info:
        angle = info["angle"]

        point_1 = ((curr_x + 0.5) * nb_repeats[0],
                   (curr_y + 0.5) * nb_repeats[1])
        point_2 = (point_1[0] + math.cos(angle) * nb_repeats[0],
                   point_1[1] + math.sin(angle) * nb_repeats[1])
        pygame.draw.line(surface, color=(0, 245, 255),
                         start_pos=point_1, end_pos=point_2)

    surface = pygame.transform.flip(surface, False, True)

    return surface


# General Environment Generator
env_dim = (32, 32)
obstacle_freq = (2, 2)
fill_ratio = 0.35

generator = GeneralEnvironmentGenerator(env_dim)
generator.set_obstacle_frequency(obstacle_freq)
generator.set_fill_ratio(fill_ratio)

# General Environment
agent_size = 7
fov = None
turning = False
terrain_info = False

env = GeneralEnvironment(generator)
env.set_agent_size(agent_size)
env.set_field_of_view(fov)
env.activate_turning(turning)
env.activate_terrain(terrain_info)

# Pygame
vis_dim = (512, 512)
state_size = 128

pygame.init()
screen = pygame.display.set_mode((vis_dim[0], vis_dim[1] + state_size))
screen.fill((50, 50, 50))
nb_repeats = (vis_dim[0] // env_dim[0], vis_dim[1] // env_dim[1])
state_repeats = state_size // env.get_state_shape()[1]
clock = pygame.time.Clock()
font = pygame.font.Font("freesansbold.ttf", 32)

running = True

# Environment setup
done = False
state = env.reset()
total_reward = 0.0
reward = 0.0

info = env.get_info({})
maps = {
    "obstacle_map": env.get_obstacle_map(),
    "coverage_map": env.get_coverage_map()
}

while running:
    screen.fill((50, 50, 50))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            keys_pressed = pygame.key.get_pressed()

            # reset environment if R-key is pressed
            if keys_pressed[pygame.K_r]:
                state = env.reset()
                maps["obstacle_map"] = env.get_obstacle_map()
                maps["coverage_map"] = env.get_coverage_map()
                info = env.get_info({})
                done = False

                print()
                print()
                print("NEW EPISODE")

            # transform key-presses to actions
            action = None
            if keys_pressed[pygame.K_8] or keys_pressed[pygame.K_KP8]:
                if not turning:
                    action = 3
            if keys_pressed[pygame.K_4] or keys_pressed[pygame.K_KP4]:
                action = 0
                if turning:
                    action = 0
            if keys_pressed[pygame.K_6] or keys_pressed[pygame.K_KP6]:
                action = 1
                if turning:
                    action = 2
            if keys_pressed[pygame.K_5] or keys_pressed[pygame.K_KP5]:
                action = 2
                if turning:
                    action = 1

            # execute action
            if action is not None and not done:
                state, reward, done, info = env.step(action)
                maps["obstacle_map"] = env.get_obstacle_map()
                maps["coverage_map"] = env.get_coverage_map()
                total_reward += reward

                print()
                print(f"REWARD: {round(info['reward'], 2)}")
                print(f"TOTAL REWARD: {round(info['total_reward'], 2)}")
                if terrain_info:
                    print(f"TERRAIN DIFF: {info['terr_diff']}")
                    print(f"TOTAL POSS TERR DIFF: {info['total_pos_terr_diff']}")
                if info['done']:
                    print("DONE")
                if (info['collision']):
                    print(f"COLLISION: {info['collision']}")
                if (info['full_cc']):
                    print(f"FULL CC: {info['full_cc']}")


    env_surface = state_to_surface(maps, info, nb_repeats)
    screen.blit(env_surface, (0, 0))

    for i in range(state.shape[0]):
        unscaled = state[i]
        unscaled = np.array(unscaled * 255, dtype=int)
        unscaled = np.stack([unscaled, unscaled, unscaled])
        unscaled = np.moveaxis(unscaled, 0, 2)

        scaled_img = np.repeat(unscaled, state_repeats, axis=0)
        scaled_img = np.repeat(scaled_img, state_repeats, axis=1)

        surface = pygame.surfarray.make_surface(scaled_img)
        surface = pygame.transform.flip(surface, False, True)

        pygame.draw.rect(
            surface,
            (255, 0, 0),
            (0, 0, state_size, state_size),
            1
        )

        screen.blit(surface, (i * state_size, vis_dim[1]))

    pygame.display.update()
    clock.tick(60)

pygame.quit()


