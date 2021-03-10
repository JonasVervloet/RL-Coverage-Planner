import sys, getopt
import matplotlib.pyplot as plt

from environments.env_generation import EnvironmentGenerator
from environments.env_representation import EnvironmentRepresentation
from environments.environment import Environment

SHORT_OPTIONS = ""
LONG_OPTIONS = [
    "heightRequired",
    "dim=",
    "hFreq=",
    "oFreq=",
    "fillRatio=",
    "loadEnv=",
    "movePunish=",
    "obstaclePunish=",
    "discoverReward=",
    "coverageReward=",
    "maxStepMultiplier="
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

    env_repr = None

    move_punishment = 0.05
    obstacle_punishment = 0.5
    discover_reward = 1.0
    coverage_reward = 50.0
    max_step_multiplier = 2

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

        if option == "--loadEnv":
            arg_split = argument.split(",")
            env_repr = EnvironmentRepresentation()
            env_repr.load(arg_split[0], arg_split[1])

        if option == "--movePunish":
            move_punishment = float(argument)

        if option == "--obstaclePunish":
            obstacle_punishment = float(argument)

        if option == "--discoverReward":
            discover_reward = float(argument)

        if option == "--coverageReward":
            coverage_reward = float(argument)

        if option == "--maxStepMultiplier":
            max_step_multiplier = int(argument)

    env_generator = EnvironmentGenerator(height_required)
    env_generator.set_dimension(dimension)
    env_generator.set_height_frequency(height_frequency)
    env_generator.set_obstacle_frequency(obstacle_frequency)
    env_generator.set_fill_ration(fill_ratio)

    env = Environment(env_generator)
    if env_repr is not None:
        env.set_environment_representation(env_repr)

    env.MOVE_PUNISHMENT = move_punishment
    env.OBSTACLE_PUNISHMENT = obstacle_punishment
    env.DISCOVER_REWARD = discover_reward
    env.COVERAGE_REWARD = coverage_reward
    env.MAX_STEP_MULTIPLIER = max_step_multiplier

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