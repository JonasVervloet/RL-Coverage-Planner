import sys, getopt
import numpy as np
import json
import pygame
import torch

from environments.env_generation import EnvironmentGenerator
from environments.env_representation import EnvironmentRepresentation
from environments.environment import Environment

from train import GENERATORS, AGENTS, OPTIMIZERS

SHORT_OPTIONS = ""
LONG_OPTIONS = [
    "loadTrainArgs=",
    "episodeNb=",
    "visDim=",
    "fps=",
    "softmax="
]

COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "maroon4": (139, 28, 98, 255),
    "magenta": (255,0,230),
    "forest_green": (0,50,0),
    "tan": (230,220,170),
    "coffee_brown": (200,190,140),
    "moon_glow": (235,245,255),
    "red": (255, 0, 0)
}


def state_to_image(state, nb_repeats):
    _, m, n = state.shape

    unscaled_img = np.zeros((m, n, 3))
    unscaled_img[state[2].astype(bool)] = np.array(COLORS["coffee_brown"])
    unscaled_img[state[1].astype(bool)] = np.array(COLORS["forest_green"])
    unscaled_img[state[0].astype(bool)] = np.array(COLORS["white"])
    unscaled_img[np.multiply(state[2].astype(bool), state[0].astype(bool))] = np.array(COLORS["red"])

    scaled_img = np.repeat(unscaled_img, nb_repeats[0], axis=0)
    scaled_img = np.repeat(scaled_img, nb_repeats[1], axis=1)

    return scaled_img


def main(argv):
    try:
        options, args = getopt.getopt(argv, SHORT_OPTIONS, LONG_OPTIONS)
    except getopt.GetoptError:
        print("badly formatted command line arguments")

    arguments = {
        "heightRequired": False,
        "visDim": (800, 800),
        "dim": (16, 16),
        "hFreq": (2, 2),
        "oFreq": (2, 2),
        "fillRatio": 0.14,
        "loadEnv": None,
        "obstaclePunish": 0.5,
        "discoverReward": 1.0,
        "coverageReward": 50.0,
        "maxStepMultiplier": 2,
        "networkGen": "simpleQ",
        "rlAgent": "deepQ",
        "optim": "rmsProp",
        "nbEpisodes": 250,
        "savePath": "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/",
        "loadTrainArgs": "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/test/arguments.txt",
        "episodeNb": 250,
        "fps": 2,
        "softmax": False
    }

    for option, argument in options:
        if option == "--loadTrainArgs":
            with open(argument + "arguments.txt") as input_file:
                input_data = json.load(input_file)
                arguments.update(input_data)
                arguments["episodeNb"] = arguments["nbEpisodes"]
                arguments["loadTrainArgs"] = argument

        if option == "--visDim":
            arguments["visDim"] = tuple(map(int, argument.split(",")))

        if option == "--episodeNb":
            arguments["episodeNb"] = int(argument)

        if option == "--fps":
            arguments["fps"] = int(argument)

        if option == "--softmax":
            arguments["softmax"] = bool(argument)

    print(arguments["softmax"])

    env_generator = EnvironmentGenerator(arguments["heightRequired"])
    print(f"dim: {arguments['dim']}")
    env_generator.set_dimension(arguments["dim"])
    env_generator.set_height_frequency(arguments["hFreq"])
    print(f"oFreq: {arguments['oFreq']}")
    env_generator.set_obstacle_frequency(arguments["oFreq"])
    print(f"fill ratio: {arguments['fillRatio']}")
    env_generator.set_fill_ration(arguments["fillRatio"])

    env = Environment(env_generator)
    if arguments["loadEnv"] is not None:
        env_repr = EnvironmentRepresentation()
        env_repr.load(arguments["loadEnv"][0], arguments["loadEnv"][1])
        arguments["dim"] = env_repr.get_dimension()
        env.set_environment_representation(env_repr)

    env.MOVE_PUNISHMENT = arguments["movePunish"]
    env.OBSTACLE_PUNISHMENT = arguments["obstaclePunish"]
    env.DISCOVER_REWARD = arguments["discoverReward"]
    env.COVERAGE_REWARD = arguments["coverageReward"]
    env.MAX_STEP_MULTIPLIER = arguments["maxStepMultiplier"]

    network_gen = GENERATORS[arguments["networkGen"]](
        arguments["dim"],
        env.get_input_depth(),
        env.get_nb_actions()
    )
    agent = AGENTS[arguments["rlAgent"]](
        network_gen,
        OPTIMIZERS[arguments["optim"]],
        env.get_nb_actions()
    )
    agent.load(arguments["loadTrainArgs"], arguments["episodeNb"])
    agent.evaluate()

    pygame.init()
    screen = pygame.display.set_mode(arguments["visDim"])
    nb_repeats = (arguments["visDim"][0] // arguments["dim"][0],
                  arguments["visDim"][1] // arguments["dim"][1])
    screen.fill(COLORS["white"])
    clock = pygame.time.Clock()
    font = pygame.font.Font("freesansbold.ttf", 32)

    running = True

    done = False
    current_state = env.reset()
    total_reward = 0.0
    reward = 0.0

    while running:
        screen.fill(COLORS["white"])

        enter_pressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                enter_pressed = True

        img = state_to_image(current_state, nb_repeats)
        state_surface = pygame.surfarray.make_surface(img)
        screen.blit(state_surface, (40, 40))

        text = font.render(f"REWARD: {reward} --- TOTAL REWARD: {round(total_reward, 2)}",
                           True, COLORS['black'], COLORS['white'])
        screen.blit(text, (100, 25))

        if enter_pressed:
            n_reward = 0.0
            n_state = env.reset()
            total_reward = 0.0
            done = False
        elif not done:
            action = agent.select_action(torch.tensor(current_state, dtype=torch.float), arguments["softmax"])
            n_state, n_reward, done = env.step(action)
        else:
            print("waiting for enter")
            n_reward = 0.0
            n_state = current_state

        current_state = n_state
        reward = n_reward
        total_reward += reward

        pygame.display.update()
        clock.tick(arguments["fps"])

    pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1:])
