import sys, getopt
import matplotlib.pyplot as plt

from environments.env_generation import EnvironmentGenerator

SHORT_OPTIONS = ""
LONG_OPTIONS = [
    "heightRequired",
    "dim=",
    "hFreq=",
    "oFreq=",
    "fillRatio="
]

def main(argv):
    try:
        options, args = getopt.getopt(argv, SHORT_OPTIONS, LONG_OPTIONS)
    except getopt.GetoptError:
        print("badly formatted command line arguments")

    height_required = False
    dimension = (16, 16)
    height_frequency = (2, 2)
    obstacle_frequency = (2, 2)
    fill_ratio = 0.14


    for option, argument in options:
        if option == "--heightRequired":
            height_required = True

        if option == "--dim":
            arg_split = map(int, argument.split(","))
            dimension = tuple(arg_split)

        if option == "--hFreq":
            arg_split = map(int, argument.split(","))
            height_frequency = tuple(arg_split)

        if option == "--oFreq":
            arg_split = map(int, argument.split(","))
            obstacle_frequency = tuple(arg_split)

        if option == "--fillRatio":
            fill_ratio = float(argument)

    env_generator = EnvironmentGenerator(height_required)
    env_generator.set_dimension(dimension)
    env_generator.set_height_frequency(height_frequency)
    env_generator.set_obstacle_frequency(obstacle_frequency)
    env_generator.set_fill_ration(fill_ratio)

    fig, axs = plt.subplots(2, 2)

    env_repr = env_generator.generate_environment()
    axs[0][0].imshow(env_repr.terrain_map)
    axs[0][1].imshow(env_repr.obstacle_map)

    env_repr = env_generator.generate_environment()
    axs[1][0].imshow(env_repr.terrain_map)
    axs[1][1].imshow(env_repr.obstacle_map)

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])