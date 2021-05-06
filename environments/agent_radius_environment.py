import numpy as np
import pygame

from environments.environment import Environment
from environments.env_generation import EnvironmentGenerator


class AgentRadiusEnvironment(Environment):
    def __init__(self, generator):
        self.agent_radius = 1
        super().__init__(generator)

    def reset(self):
        state = super().reset()
        self.cover_tiles(self.current_pos)
        return state

    def set_agent_radius(self, n_radius):
        self.agent_radius = n_radius
        self.generator.set_agent_radius(n_radius)

    def step(self, action):
        if self.done:
            raise Exception("ENVIRONMENT: should reset first...")

        if action == 0:
            n_position = (max(0, self.current_pos[0] - 1), self.current_pos[1])
        elif action == 1:
            n_position = (min(self.env_info.obstacle_map.shape[0] - 1, self.current_pos[0] + 1), self.current_pos[1])
        elif action == 2:
            n_position = (self.current_pos[0], max(0, self.current_pos[1] - 1))
        elif action == 3:
            n_position = (self.current_pos[0], min(self.env_info.obstacle_map.shape[1] - 1, self.current_pos[1] + 1))
        else:
            raise Exception("ACTION NOT SUPPORTED!")

        self.nb_steps += 1

        reward = self.get_reward(n_position)
        self.total_reward += reward
        done = self.is_done(n_position)

        if self.env_info.get_obstacle_map()[n_position] == 0:
            self.cover_tiles(n_position)

        self.current_pos = n_position
        info = {
            "reward": reward,
            "total reward": self.total_reward,
            "nb steps": self.nb_steps,
            "nb visited tiles": self.get_nb_visited_tiles(),
            "complete coverage": self.complete_coverage(),
        }
        info.update(self.get_extra_info())

        return self.get_state(), reward, done, info

    def get_extra_info(self):
        extra = super().get_extra_info()
        extra.update({
            'radius': self.agent_radius
        })

        return extra

    def cover_tiles(self, n_position):
        n_x = n_position[0] + 0.5
        n_y = n_position[1] + 0.5

        dim_x, dim_y = self.get_dimension()
        x = np.linspace(0.5, dim_x - 0.5, dim_x)
        y = np.linspace(0.5, dim_y - 0.5, dim_y)

        yy, xx = np.meshgrid(x, y)

        dists = np.sqrt(
            (xx - n_x)**2 + (yy - n_y) ** 2
        )
        mask = dists <= self.agent_radius + 0.5

        self.visited_tiles[mask] = 1


if __name__ == "__main__":
    dim = (32, 32)
    vis_dim = (800, 800)
    agent_radius = 4

    generator = EnvironmentGenerator()
    generator.set_dimension(dim)
    generator.set_obstacle_frequency((2, 2))
    generator.set_fill_ratio(0.25)
    generator.set_agent_radius(agent_radius)

    env = AgentRadiusEnvironment(generator)
    env.set_agent_radius(agent_radius)
    state = env.reset()

    pygame.init()
    screen = pygame.display.set_mode(vis_dim)
    screen.fill((255, 255, 255))
    nb_repeats = (vis_dim[0] // dim[0], vis_dim[1] // dim[1])
    clock = pygame.time.Clock()
    font = pygame.font.Font("freesansbold.ttf", 32)

    running = True

    done = False
    total_reward = 0.0
    reward = 0.0

    curr_state = env.reset()
    info = env.get_extra_info()
    maps = {
        "obstacle_map": env.get_obstacle_map(),
        "coverage_map": env.get_coverage_map()
    }

    while running:
        screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                keys_pressed = pygame.key.get_pressed()
                action = None
                if keys_pressed[pygame.K_4] or keys_pressed[pygame.K_KP4]:
                    action = 0
                if keys_pressed[pygame.K_6] or keys_pressed[pygame.K_KP6]:
                    action = 1
                if keys_pressed[pygame.K_5] or keys_pressed[pygame.K_KP5]:
                    action = 2
                if keys_pressed[pygame.K_8] or keys_pressed[pygame.K_KP8]:
                    action = 3

                if action is not None and not done:
                    curr_state, reward, done, info = env.step(action)
                    if done:
                        print("DONE")
                    maps['obstacle_map'] = env.get_obstacle_map()
                    maps['coverage_map'] = env.get_coverage_map()
                    total_reward += reward

                if keys_pressed[pygame.K_r]:
                    env.reset()
                    maps['obstacle_map'] = env.get_obstacle_map()
                    maps['coverage_map'] = env.get_coverage_map()
                    info = env.get_extra_info()
                    done = False

        m, n = maps["obstacle_map"].shape

        unscaled_img = np.zeros((m, n, 3))
        unscaled_img[maps['obstacle_map'].astype(bool)] = np.array((200, 190, 140))
        unscaled_img[maps['coverage_map'].astype(bool)] = np.array((0, 50, 0))

        unscaled_img[info['position']] = np.array((255, 255, 255))

        unscaled_img[5, 0, :] = np.array((255, 0, 0))
        unscaled_img[0, 5, :] = np.array((0, 255, 0))

        scaled_img = np.repeat(unscaled_img, nb_repeats[0], axis=0)
        scaled_img = np.repeat(scaled_img, nb_repeats[1], axis=1)

        surface = pygame.surfarray.make_surface(scaled_img)

        if 'radius' in info:
            radius = (info['radius'] + 0.5) * nb_repeats[0]
            curr_pos = info['position']
            pygame.draw.circle(
                surface,
                color=(0, 245, 255),
                center=((curr_pos[0] + 0.5) * nb_repeats[0], (curr_pos[1] + 0.5) * nb_repeats[1]),
                radius=radius,
                width=2
            )

        surface = pygame.transform.flip(surface, False, True)
        screen.blit(surface, (0, 0))

        pygame.display.update()
        clock.tick(60)

    pygame.quit()
