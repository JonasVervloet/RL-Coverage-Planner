import gym
from gym.spaces import Tuple, Discrete

HOLE_POSITIONS = [5, 7, 11, 12]
TILES = 12
OPTIMAL_STEPS = 14

CUSTOM_MAP = [
    'SFFF',
    'FHFH',
    'FFFH',
    'HFFF'
]

class FrozenLakeCoverage():
    def __init__(self, slippery=False):
        self.env = gym.make("FrozenLake-v0", is_slippery=slippery, desc=CUSTOM_MAP)
        self.visited_states = []
        self.observation_size = (self.env.observation_space.n, TILES)
        self.action_size = self.env.action_space.n
        self.max_visits = 0
        self.start_state = self.env.reset()

    def reset(self):
        self.start_state = self.env.reset()
        self.visited_states = [self.start_state]
        return self.start_state, self.get_coverage_state()

    def get_coverage_state(self):
        return len(self.visited_states) - 1

    def has_full_coverage(self, state):
        return (len(self.visited_states) == TILES and
                state == self.start_state)

    def step(self, action):
        new_state, reward, done, info = self.env.step(action)

        if new_state not in HOLE_POSITIONS:
            if new_state not in self.visited_states:
                reward = 1.0 / TILES
                self.visited_states.append(new_state)
            else:
                reward = -1.0/OPTIMAL_STEPS
            if self.has_full_coverage(new_state):
                reward = 1.0
                done = True
        else:
            reward = -1.0

        if done:
            self.max_visits = max(self.max_visits, len(self.visited_states))

        return (new_state, self.get_coverage_state()), reward, done, info

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()
