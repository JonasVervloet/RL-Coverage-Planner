from opensimplex import OpenSimplex
import matplotlib.pyplot as plt
import numpy as np

MAX_NOISE_SEED = 1024
MAX_OFFSET = 1024.0


def generate_noise_map(dim, res, seed, nb_octaves, persistence=0.5):
    np.random.seed(seed)
    noise_generator = OpenSimplex(np.random.randint(0, MAX_NOISE_SEED))
    image = np.zeros(dim)

    for octave_nb in range(nb_octaves):
        offset = np.random.random((2,)) * MAX_OFFSET

        octave_res = np.array(res) * (2**octave_nb)
        amplitude = 1.0 * (persistence ** octave_nb)
        print(octave_res)
        print(amplitude)
        image += amplitude * generate_simple_noise_map(dim, octave_res, offset, noise_generator)

    return image / np.max(np.abs(image))


def generate_simple_noise_map(dim, res, offset, generator):

    return np.array([[
        generator.noise2d(x + offset[0], y + offset[1])
            for x in np.arange(0, res[0], res[0]/dim[0])]
                for y in np.arange(0, res[1], res[1]/dim[1])]
    )


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2)

    image1 = generate_noise_map((256, 256), (5, 5), 10, 2, 0.25)
    axs[0][0].imshow(image1, cmap='gray')

    image2 = generate_noise_map((256, 256), (5, 5), 10, 2, 0.5)
    axs[0][1].imshow(image2, cmap='gray')

    image3 = generate_noise_map((256, 256), (5, 5), 10, 2, 0.75)
    axs[1][0].imshow(image3, cmap='gray')

    image4 = generate_noise_map((256, 256), (5, 5), 10, 2, 1.0)
    axs[1][1].imshow(image4, cmap='gray')

    plt.show()