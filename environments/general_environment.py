import numpy as np
import math
import random


class GeneralEnvironment:

    MAX_STEP_MULTIPLIER = 2

    MOVE_PUNISH = 0.05
    OBSTACLE_PUNISH = 0.5
    TERR_PUNISH = 0.05

    DISC_REWARD = 1.0
    CC_REWARD = 50.0

    def __init__(self, generator):
        self.agent_size = 1
        self.fov = None
        self.turning = False
        self.terrain_info = False

        self.generator = generator
        self.env_repr = None

        self.done = False
        self.nb_steps = 0
        self.total_covered_tiles = 0
        self.total_reward = 0.0
        self.total_terr_diff = 0.0

        self.current_position = (0, 0)
        self.visited_tiles = []

        self.angle_count = 0

    def set_agent_size(self, n_size):
        assert(n_size >= 1)

        self.agent_size = n_size

    def set_field_of_view(self, n_fov):
        assert(n_fov is None or (n_fov >= 1 and n_fov % 2 == 1))

        self.fov = n_fov

    def activate_turning(self, active):
        self.turning = active

    def activate_terrain(self, active):
        self.terrain_info = active

    def get_nb_actions(self):
        return 3 if self.turning else 4

    def get_state_shape(self):
        dim_x, dim_y = self.generator.get_dimension()
        if self.fov is not None:
            dim_x, dim_y = self.fov, self.fov

        nb_channels = 3
        if self.fov is not None:
            nb_channels -= 1
        if self.terrain_info:
            nb_channels += 1

        return [nb_channels, dim_x, dim_y]

    def get_extra_spacing(self):
        extra_spacing = self.agent_size // 2
        if self.fov is not None:
            extra_spacing = max(
                extra_spacing, self.fov // 2
            )
        if self.turning:
            extra_spacing = max(
                extra_spacing, (self.generator.get_dimension()[0] - 1) // 5
            )
        return extra_spacing

    def get_obstacle_map(self):
        return np.copy(
            self.env_repr.get_obstacle_map()
        )

    def get_coverage_map(self):
        return np.copy(
            self.visited_tiles
        )

    def reset(self):
        self.env_repr = self.generator.generate_environment(
            self.get_extra_spacing(),
            int((self.agent_size + 1) / 2)
        )

        self.done = False
        self.nb_steps = 0
        self.total_covered_tiles = 0
        self.total_reward = 0.0
        self.total_terr_diff = 0.0

        self.current_position = random.sample(
            self.env_repr.get_start_positions(), 1
        )[0]
        self.visited_tiles = np.zeros_like(self.env_repr.get_obstacle_map())

        # cover tiles at initial position
        mask = GeneralEnvironment.get_radius_map(self.agent_size)
        xx, yy = self.get_local_map(
            self.agent_size,
            self.current_position,
            "indices"
        )
        xx_select = xx[mask]
        yy_select = yy[mask]
        self.visited_tiles[(xx_select, yy_select)] = 1

        if self.turning:
            self.set_random_angle_count()

        return self.get_state()

    def set_random_angle_count(self):
        assert(self.turning)

        self.angle_count = np.random.randint(0, 8)
        collision = True

        while collision:
            self.angle_count = np.random.randint(0, 8)
            n_pos = self.get_new_position(1)
            collision = self.verify_collision(n_pos)

    def step(self, action):
        n_position = self.get_new_position(action)

        collision = self.verify_collision(n_position)
        nb_covered_tiles = self.get_nb_covered_tiles(n_position)
        terr_diff = self.get_terrain_difference(n_position)

        full_cc = self.complete_coverage(nb_covered_tiles)

        done = self.get_done_status(collision, full_cc)
        reward = self.get_reward(collision, nb_covered_tiles,
                                 full_cc, terr_diff)

        self.done = done
        self.nb_steps += 1
        self.total_covered_tiles += nb_covered_tiles
        self.total_terr_diff += terr_diff
        self.total_reward += reward

        self.update_environment(action, n_position)

        state = self.get_state()
        info = self.get_info({
            "collision": collision,
            "nb_covered_tiles": nb_covered_tiles,
            "terr_diff": terr_diff,
            "full_cc": full_cc,
            "done": done,
            "reward": reward
        })

        return state, reward, done, info

    def get_new_position(self, action):
        assert(action < self.get_nb_actions())

        n_position = self.current_position
        if not self.turning:
            if action == 0:
                n_position = (
                    max(0, self.current_position[0] - 1),
                    self.current_position[1]
                )
            elif action == 1:
                max_x = self.generator.get_dimension()[0] - 1
                n_position = (
                    min(max_x, self.current_position[0] + 1),
                    self.current_position[1]
                )
            elif action == 2:
                n_position = (
                    self.current_position[0],
                    max(0, self.current_position[1] - 1)
                )
            elif action == 3:
                max_y = self.generator.get_dimension()[1] - 1
                n_position = (
                    self.current_position[0],
                    min(max_y, self.current_position[1] + 1)
                )
        else:
            if action == 1:
                n_x, n_y = self.current_position

                if 4 > self.angle_count > 0:
                    n_y += 1
                elif 4 < self.angle_count < 8:
                    n_y -= 1

                turned_angle = (self.angle_count + 2) % 8
                if 4 > turned_angle > 0:
                    n_x += 1
                elif 4 < turned_angle < 8:
                    n_x -= 1

                n_position = (n_x, n_y)

        return n_position

    def get_local_map(self, size, position, type, copy=True):
        assert(type in ["obstacle", "terrain", "coverage", "indices"])

        offset = size // 2
        extra = 1 if size % 2 == 1 else 0

        if type == "indices":
            x = np.linspace(
                position[0] - offset, position[0] + offset,
                size, endpoint=True, dtype=int
            )
            y = np.linspace(
                position[1] - offset, position[1] + offset,
                size, endpoint=True, dtype=int
            )
            xx, yy = np.meshgrid(x, y)

            return np.transpose(xx), np.transpose(yy)

        x_pos = position[0] + offset
        y_pos = position[1] + offset

        map = None
        if type == "obstacle":
            map = self.env_repr.get_obstacle_map(offset)

        elif type == "terrain":
            map = self.env_repr.get_terrain_map(offset)

        elif type == "coverage":
            dim_x, dim_y = self.generator.get_dimension()
            n_dim_x, n_dim_y = (dim_x + 2 * offset, dim_y + 2 * offset)
            map = np.zeros((n_dim_x, n_dim_y))
            map[offset:n_dim_x-offset, offset:n_dim_y-offset] = self.visited_tiles

        selection = map[
                    x_pos - offset: x_pos + offset + extra,
                    y_pos - offset: y_pos + offset + extra
        ]
        if copy:
            return np.copy(selection)

        return selection

    def get_turned_local_map(self, size, center, type, angle):
        assert (type in ["obstacle", "terrain", "coverage"])

        dim_x, dim_y = self.generator.get_dimension()
        x = np.linspace(0.5, dim_x - 0.5, dim_x) - (dim_x / 2)
        y = np.linspace(0.5, dim_x - 0.5, dim_x) - (dim_x / 2)

        offset = size // 2
        extra = 1 if size % 2 == 1 else 0
        x = x[center[0] - offset: center[0] + offset + extra]
        y = y[center[1] - offset: center[1] + offset + extra]

        xx, yy = np.meshgrid(x, y)
        xx, yy = np.transpose(xx), np.transpose(yy)

        xx_rot = math.cos(angle) * xx - math.sin(angle) * yy + (dim_x / 2)
        yy_rot = math.sin(angle) * xx + math.cos(angle) * yy + (dim_y / 2)

        xx_idxs = np.floor(xx_rot).astype(int) + self.get_extra_spacing()
        yy_idxs = np.floor(yy_rot).astype(int) + self.get_extra_spacing()

        if type == "obstacle":
            obstacle_map = self.env_repr.get_obstacle_map(self.get_extra_spacing())
            return np.copy(obstacle_map[(xx_idxs, yy_idxs)])

        elif type == "terrain":
            terrain_map = self.env_repr.get_terrain_map(self.get_extra_spacing())
            return np.copy(terrain_map[(xx_idxs, yy_idxs)])

        elif type == "coverage":
            extra_space = self.get_extra_spacing()
            n_dim_x, n_dim_y = (dim_x + 2 * extra_space,
                                dim_y + 2 * extra_space)
            coverage_map = np.zeros((n_dim_x, n_dim_y))
            coverage_map[
                extra_space:n_dim_x - extra_space,
                extra_space:n_dim_y - extra_space
            ] = self.visited_tiles
            return np.copy(coverage_map[(xx_idxs, yy_idxs)])

    def verify_collision(self, n_position):
        local_obstacle_map = self.get_local_map(
            self.agent_size,
            n_position,
            "obstacle"
        )
        mask = GeneralEnvironment.get_radius_map(self.agent_size)
        if np.sum(local_obstacle_map[mask]) > 0:
            return True

        return False

    def get_nb_covered_tiles(self, n_position):
        local_coverage_map = self.get_local_map(
            self.agent_size,
            n_position,
            "coverage"
        )
        mask = GeneralEnvironment.get_radius_map(self.agent_size)

        nb_covered_tiles = np.sum(mask) - np.sum(local_coverage_map[mask])

        return nb_covered_tiles

    def complete_coverage(self, nb_covered_tiles):
        nb_free_tiles = self.env_repr.get_nb_free_tiles()
        if np.sum(self.visited_tiles) + nb_covered_tiles == nb_free_tiles:
            return True

        return False

    def get_terrain_difference(self, n_position):
        curr_height = self.env_repr.get_terrain_map()[self.current_position]
        n_height = self.env_repr.get_terrain_map()[n_position]
        return n_height - curr_height

    def get_done_status(self, collision, full_cc):
        # Terminate on collision with obstacle
        if collision:
            return True

        # Terminate when the agent has covered the full environment
        if full_cc:
            return True

        # Terminate when the agent has done too many steps
        nb_free_tiles = self.env_repr.get_nb_free_tiles()
        if self.nb_steps >= GeneralEnvironment.MAX_STEP_MULTIPLIER * nb_free_tiles:
            return True

        return False

    def get_reward(self, collision, nb_covered_tiles, full_cc, terr_diff):
        # standard punishment every step
        reward = -GeneralEnvironment.MOVE_PUNISH

        # extra punishment on collision with obstacle
        if collision:
            reward -= GeneralEnvironment.OBSTACLE_PUNISH
            return reward

        # reward for every newly covered tile.
        reward += GeneralEnvironment.DISC_REWARD * nb_covered_tiles

        # reward for covering the whole area
        if full_cc:
            reward += GeneralEnvironment.CC_REWARD

        # TODO: always add terrain difference punishment??
        if self.turning:
            reward -= GeneralEnvironment.TERR_PUNISH * terr_diff

        return reward

    def update_environment(self, action, n_position):
        # change turning angle if turning is selected
        if self.turning:
            if action == 0:
                self.angle_count = (self.angle_count + 1) % 8
            elif action == 2:
                self.angle_count = (self.angle_count - 1) % 8

        # update coverage map
        mask = GeneralEnvironment.get_radius_map(self.agent_size)
        local_obstacle_map = self.get_local_map(
            self.agent_size,
            n_position,
            "obstacle"
        )
        mask[local_obstacle_map == 1] = False

        xx, yy = self.get_local_map(
            self.agent_size,
            n_position,
            "indices"
        )
        xx_select, yy_select = xx[mask], yy[mask]

        self.visited_tiles[(xx_select, yy_select)] = 1

        # update position
        self.current_position = n_position

    def get_state(self):
        # both FOV and turning are not active
        if self.fov is None and not self.turning:
            curr_pos_map = np.zeros_like(self.env_repr.get_obstacle_map())
            curr_pos_map[self.current_position] = 1

            state = np.stack([curr_pos_map, self.visited_tiles,
                              self.env_repr.get_obstacle_map()])
            if self.terrain_info:
                state = np.concatenate([state, [self.env_repr.get_terrain_map()]])

            return state

        # FOV is active, but turning is not
        elif self.fov is not None and not self.turning:
            local_coverage_map = self.get_local_map(
                self.fov,
                self.current_position,
                "coverage"
            )
            local_obstacle_map = self.get_local_map(
                self.fov,
                self.current_position,
                "obstacle"
            )
            state = np.stack([local_coverage_map, local_obstacle_map])

            if self.terrain_info:
                local_terrain_map = self.get_local_map(
                    self.fov,
                    self.current_position,
                    ""
                )
                state = np.concatenate([state, [local_terrain_map]])

            return state

        # turning is active, but FOV is not
        elif self.turning and self.fov is None:
            angle = ((self.angle_count - 2) % 8) * (math.pi / 4)
            dim_x, dim_y = self.generator.get_dimension()

            # coverage and obstacle map
            coverage_map = self.get_turned_local_map(
                dim_x,
                (dim_x // 2, dim_y // 2),
                "coverage",
                angle
            )
            obstacle_map = self.get_turned_local_map(
                dim_x,
                (dim_x // 2, dim_y // 2),
                "obstacle",
                angle
            )

            # current position map
            curr_pos_map = np.zeros((dim_x, dim_y))

            rel_pos_x = self.current_position[0] + 0.5 - (dim_x / 2)
            rel_pos_y = self.current_position[1] + 0.5 - (dim_y / 2)

            rot_pos_x = rel_pos_x * math.cos(angle) + rel_pos_y * math.sin(angle)
            rot_pos_y = -rel_pos_x * math.sin(angle) + rel_pos_y * math.cos(angle)

            curr_pos_map[
                math.floor(rot_pos_x + dim_x / 2),
                math.floor(rot_pos_y + dim_x / 2)
            ] = 1.0

            state = np.stack([curr_pos_map, coverage_map, obstacle_map])

            if self.terrain_info:
                local_terrain_map = self.get_turned_local_map(
                    dim_x,
                    (dim_x // 2, dim_y // 2),
                    "terrain",
                    angle
                )
                state = np.concatenate([state, [local_terrain_map]])
            return state

        # both FOV and turning are active
        else:
            angle = ((self.angle_count - 2) % 8) * (math.pi / 4)
            local_coverage_map = self.get_turned_local_map(
                self.fov,
                self.current_position,
                "coverage",
                angle
            )
            local_obstacle_map = self.get_turned_local_map(
                self.fov,
                self.current_position,
                "obstacle",
                angle
            )
            state = np.stack([local_coverage_map, local_obstacle_map])

            if self.terrain_info:
                local_terrain_map = self.get_turned_local_map(
                    self.fov,
                    self.current_position,
                    "terrain",
                    angle
                )
                state = np.concatenate([state, [local_terrain_map]])

            return state

    def get_info(self, info):
        info.update({
            "current_position": self.current_position,
            "total_reward": self.total_reward,
            "total_terr_diff": self.total_terr_diff,
            "total_covered_tiles": self.total_covered_tiles,
            "nb_steps": self.nb_steps,
            "agent_size": self.agent_size
        })
        if self.turning:
            info.update({
                "angle": self.angle_count * (math.pi / 4)
            })
        if self.fov is not None:
            info.update({
                "fov": self.fov
            })
        return info

    @staticmethod
    def get_radius_map(agent_size):
        x = np.linspace(0.5, agent_size - 0.5, agent_size) - (agent_size/2)
        yy, xx = np.meshgrid(x, x)

        dists = np.sqrt(xx**2 + yy**2)
        mask = dists <= (agent_size/2)

        return mask