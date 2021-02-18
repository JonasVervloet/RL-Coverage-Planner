import matplotlib.pyplot as plt
import numpy as np

SHAPE = (256, 256)
RES = (16, 16)


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
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
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


# fig, axs = plt.subplots(2, 2)
#
# image = generate_perlin_noise_2d(SHAPE, (1, 1))
# axs[0][0].imshow(image, cmap='gray')
#
# image = generate_perlin_noise_2d(SHAPE, (2, 2))
# axs[0][1].imshow(image, cmap='gray')
#
# image = generate_perlin_noise_2d(SHAPE, (8, 8))
# axs[1][0].imshow(image, cmap='gray')
#
# image = generate_perlin_noise_2d(SHAPE, (16, 16))
# axs[1][1].imshow(image, cmap='gray')
#
# plt.show()
#
#
# fig, axs = plt.subplots(2, 2)
#
# image = generate_fractal_noise_2d((256, 256), (2, 2), octaves=3, persistence=0.25)
# axs[0][0].imshow(image, cmap='gray')
# print(np.max(image))
#
# image = generate_fractal_noise_2d((256, 256), (4, 4), octaves=3, persistence=0.25)
# axs[0][1].imshow(image, cmap='gray')
# print(np.max(image))
#
# image = generate_fractal_noise_2d((256, 256), (8, 8), octaves=3, persistence=0.25)
# axs[1][0].imshow(image, cmap='gray')
# print(np.max(image))
#
# image = generate_fractal_noise_2d((256, 256), (16, 16), octaves=3, persistence=0.25)
# axs[1][1].imshow(image, cmap='gray')
# print(np.max(image))
#
# plt.show()