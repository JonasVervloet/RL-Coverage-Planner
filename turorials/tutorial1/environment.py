import numpy as np

MAX_STEPS = 200
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

type_to_color = {
    "player": (255, 175, 0),
    "food": (0, 255, 0),
    "enemy": (0, 0, 255)
}

action_to_dx_dy = {
    0: (1, 1),
    1: (-1, -1),
    2: (-1, 1),
    3: (1, -1)
}

class Blob:
    def __init__(self, size, type):
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)
        self.type = type

    def get_color(self):
        return type_to_color[self.type]

    def move(self, size, dx=None, dy=None):
        if dx is None:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += dx
        self.x = np.clip(self.x, 0, size - 1)

        if dy is None:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += dy
        self.y = np.clip(self.y, 0, size - 1)

    def take_action(self, action, size):
        (dx, dy) = action_to_dx_dy[action]
        self.move(size, dx, dy)

    def get_position(self):
        return [self.x, self.y]

    def __str__(self):
        return f"{self.type}: {self.x}, {self.y}"

    def __sub__(self, other):
        return [self.x - other.x, self.y - other.y]

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class BlobEnv:
    def __init__(self, size, static=True):
        self.size = size
        self.static = static

        self.player = None
        self.food = None
        self.enemy = None
        self.steps = 0

        self.reset()

    def reset(self):
        self.player = Blob(self.size, "player")
        self.food = Blob(self.size, "food")
        self.enemy = Blob(self.size, "enemy")
        self.steps = 0
        return self.get_state()

    def get_state(self):
        return np.array((self.player - self.food) + (self.player - self.enemy))

    def get_full_state(self):
        return np.array([
            self.player.get_position(),
            self.food.get_position(),
            self.enemy.get_position()
        ])

    def get_reward(self):
        if self.player == self.enemy:
            return -ENEMY_PENALTY
        elif self.player == self.food:
            return FOOD_REWARD
        else:
            return -MOVE_PENALTY

    def is_done(self):
        return (self.player == self.enemy
                or self.player == self.food
                or self.steps == MAX_STEPS)

    def goal_reached(self):
        return self.player == self.food

    def step(self, action):
        self.player.take_action(action, self.size)
        if not self.static:
            self.food.move(self.size)
            self.enemy.move(self.size)
        self.steps += 1
        return self.get_state(), self.get_reward(), self.is_done(), self.goal_reached()

    def __str__(self):
        return "\n".join([
            f"{self.player}",
            f"{self.food}",
            f"{self.enemy}"
        ])