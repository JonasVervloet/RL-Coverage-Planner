import numpy as np


class Height_Generator:
    def __init__(self):
        self.dim = (16, 16)
        self.res = (2, 2)

    def generate_grid(self):
        delta = (self.res[0] / self.dim[0], self.res[1] / self.dim[1])
        d = (self.dim[0] // self.res[0], self.dim[1] // self.res[1])
        grid = np.mgrid[0:self.res[0]:delta[0], 0:self.res[1]:delta[1]].transpose(1, 2, 0) % 1
        # Gradients
        angles = 2 * np.pi * np.random.rand(self.res[0] + 1, self.res[1] + 1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
        # Interpolation
        t = self.smooth_f(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

    @staticmethod
    def smooth_f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3