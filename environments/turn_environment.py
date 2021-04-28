import matplotlib.pyplot as plt
import random
import numpy as np
import math

from environments.environment import Environment
from environments.env_generation import SingleEnvironmentGenerator


class SimpleTurnEnvironment(Environment):
    '''
    Environment where the agent takes on a single pixel.
    The environment has three possible actions each turn.
        - turn left
        - go forward
        - turn right
    The agent will turn with 45 degree intervals. When the agent
    is turned, the agent will go to the diagonal adjacent pixel.
    The state that the agent perceives is turned according to its
    rotation.
    '''
    def __init__(self, generator):
        self.angle_count = 0
        super().__init__(generator)

    def get_nb_actions(self):
        return 3

    def reset(self):
        self.done = False
        self.env_info = self.generator.generate_environment(extra_spacing=True)
        self.current_pos = random.sample(self.env_info.start_positions, 1)[0]
        self.visited_tiles = np.zeros_like(self.env_info.get_obstacle_map())
        self.visited_tiles[self.current_pos] = 1
        self.nb_steps = 0
        self.total_reward = 0
        self.total_terrain_diff = 0

        self.angle_count = self.get_random_angle_count()

        return self.get_state()

    def get_state(self):
        dim_x, dim_y = self.get_dimension()
        x = np.linspace(0.5, dim_x - 0.5, dim_x) - (dim_x / 2)
        y = np.linspace(0.5, dim_y - 0.5, dim_y) - (dim_y / 2)

        yy, xx = np.meshgrid(x, y)
        angle = self.angle_count * (math.pi / 4)

        xx_rot = math.cos(angle) * xx - math.sin(angle) * yy + (dim_x / 2)
        yy_rot = math.sin(angle) * xx + math.cos(angle) * yy + (dim_y / 2)

        extra_spacing_x = (dim_x - 1) // 5
        extra_spacing_y = (dim_y - 1) // 5

        xx_idxs = np.floor(xx_rot).astype(int) + extra_spacing_x
        yy_idxs = np.floor(yy_rot).astype(int) + extra_spacing_y

        full_obstacle_map = self.env_info.get_obstacle_map(extra_spacing=True)
        curr_obstacle_map = full_obstacle_map[(xx_idxs, yy_idxs)]

        full_coverage_map = np.zeros((dim_x + 2 * extra_spacing_x, dim_y + 2 * extra_spacing_y))
        full_coverage_map[extra_spacing_x:-extra_spacing_x, extra_spacing_y:-extra_spacing_y] = self.visited_tiles
        curr_coverage_map = full_coverage_map[(xx_idxs, yy_idxs)]

        curr_pos_map = np.zeros((dim_x, dim_y))
        if not self.env_info.get_obstacle_map()[self.current_pos] == 1:
            curr_pos_x = self.current_pos[0] + 0.5 - (dim_x / 2)
            curr_pos_y = self.current_pos[1] + 0.5 - (dim_y / 2)
            rot_pos_x = curr_pos_x * math.cos(angle) + curr_pos_y * math.sin(angle)
            rot_pos_y = -curr_pos_x * math.sin(angle) + curr_pos_y * math.cos(angle)
            curr_pos_map[
                math.floor(rot_pos_x + dim_x / 2),
                math.floor(rot_pos_y + dim_y / 2)
            ] = 1.0

        if self.gives_terrain_info():
            full_terrain_map = self.env_info.get_terrain_map(extra_spacing=True)
            curr_terrain_map = full_terrain_map[(xx_idxs, yy_idxs)]

            return [curr_pos_map, curr_coverage_map, curr_obstacle_map, curr_terrain_map]

        return [curr_pos_map, curr_coverage_map, curr_obstacle_map]

    def get_random_angle_count(self):
        rand_value = np.random.randint(0, 8)
        valid = False

        while not valid:
            n_pos = SimpleTurnEnvironment.get_new_position(self.current_pos, rand_value)
            if self.env_info.get_obstacle_map()[n_pos] == 0:
                valid = True
            else:
                rand_value = np.random.randint(0, 8)

        return rand_value

    @staticmethod
    def get_new_position(curr_pos, angle_cnt):
        n_x = curr_pos[0]
        if 4 > angle_cnt > 0:
            n_x -= 1
        if 4 < angle_cnt < 8:
            n_x += 1

        n_y = curr_pos[1]
        turned_angle = (angle_cnt + 2) % 8
        if 4 > turned_angle > 0:
            n_y += 1
        if 4 < turned_angle < 8:
            n_y -= 1

        return (n_x, n_y)

    def get_extra_info(self):
        dim_x, dim_y = self.get_dimension()
        angle = self.angle_count * (math.pi / 4)

        curr_pos_x = self.current_pos[0] + 0.5 - (dim_x / 2)
        curr_pos_y = self.current_pos[1] + 0.5 - (dim_y / 2)

        rot_pos_x = curr_pos_x * math.cos(angle) + curr_pos_y * math.sin(angle)
        rot_pos_y = -curr_pos_x * math.sin(angle) + curr_pos_y * math.cos(angle)

        extra = super().get_extra_info()
        extra.update({
            'rotation': self.angle_count * math.pi / 4,
            'rot_pos': (math.floor(rot_pos_x + (dim_x / 2)),
                        math.floor(rot_pos_y + (dim_y / 2)))
        })

        return extra

    def step(self, action):
        n_position = self.current_pos
        if action == 0:
            self.angle_count = (self.angle_count + 1) % 8

        elif action == 1:
            n_position = self.get_new_position(self.current_pos, self.angle_count)

        elif action == 2:
            self.angle_count = (self.angle_count - 1) % 8

        self.nb_steps += 1

        reward = self.get_reward(n_position)
        self.total_reward += reward
        done = self.is_done(n_position)

        if self.env_info.get_obstacle_map()[n_position] == 0 and self.visited_tiles[n_position] == 0:
            self.visited_tiles[n_position] = 1

        self.current_pos = n_position

        info = {
            "reward": reward,
            "total reward": self.total_reward,
            "nb steps": self.nb_steps,
            "nb visited tiles": self.get_nb_visited_tiles(),
            "complete coverage": self.complete_coverage(),
        }
        if self.env_info.has_terrain_info():
            terrain_diff = abs(
                self.env_info.terrain_map[self.current_pos] - self.env_info.terrain_map[n_position]
            )
            self.total_terrain_diff += terrain_diff
            info["terrain diff"] = terrain_diff
            info["total terrain diff"] = self.total_terrain_diff
        info.update(self.get_extra_info())

        return self.get_state(), reward, done, info


if __name__ == "__main__":
    pass