import json
import torch
import torch.optim as optim

from environments.env_generation import EnvironmentGenerator, SingleEnvironmentGenerator
from environments.env_generation import GeneralEnvironmentGenerator
from environments.environment import Environment, EnvironmentFOV
from environments.turn_environment import SimpleTurnEnvironment
from environments.general_environment import GeneralEnvironment
from environments.env_representation import GeneralEnvironmentRepresentation

from networks.simple_q_network import SimpleDeepQNetworkGenerator
from networks.simple_q_network import SimpleDeepQNetworkGenerator2

from deep_rl.deep_q_agent import DeepQAgent
from deep_rl.double_dqn_agent import DoubleDeepQAgent
from deep_rl.trainer import DeepRLTrainer

DEFAULT_ARGUMENTS = {
    "cuda": True,

    "dim": (16, 16),
    "hFreq": (2, 2),
    "oFreq": (2, 2),
    "fillRatio": 0.14,
    "loadEnv": None,

    "movePunish": 0.05,
    "terrainPunish": 0.05,
    "obstaclePunish": 0.5,
    "discoverReward": 1.0,
    "coverageReward": 50.0,
    "maxStepMultiplier": 2,
    "agentSize": 1,
    "fov": None,
    "turn": False,
    "terrain": False,

    "networkGen": "simpleQ",
    "rlAgent": "deepQ",
    "inputMatch": True,
    "agentInpDepth": 3,
    "optim": "rmsProp",
    "lr": 0.01,
    "gamma": 0.9,
    "epsilonDecay": 2000,
    "targetUpdate": 1000,
    "queueLength": 5000,

    "loadEpisode": None,
    "loadPath": "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/test/",
    "loadings": [],

    "nbEpisodes": 2000,
    "printEvery": 50,
    "saveEvery": 250,
    "savePath": "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/"
}

GENERATORS = {
    "simpleQ": SimpleDeepQNetworkGenerator,
    "simpleQ2": SimpleDeepQNetworkGenerator2
}

AGENTS = {
    "deepQ": DeepQAgent,
    "doubleDQ": DoubleDeepQAgent
}

OPTIMIZERS = {
    "rmsProp": optim.RMSprop
}


def default_arguments():
    return DEFAULT_ARGUMENTS.copy()


def load_arguments(path, name="arguments"):
    with open(f"{path}{name}.txt") as input_file:
        return json.load(input_file)


def initialize_objects(args, trainer_required=False):
    # ARGUMENTS
    arguments = default_arguments()
    arguments.update(args)

    print("Initializing objects...")

    # CUDA
    device = 'cuda' if torch.cuda.is_available() and arguments["cuda"] else 'cpu'
    print(f"DEVICE: {device}")

    # ENVIRONMENT GENERATOR
    env_generator = GeneralEnvironmentGenerator(arguments["dim"])
    env_generator.set_obstacle_frequency(arguments["oFreq"])
    env_generator.set_fill_ratio(arguments["fillRatio"])
    env_generator.set_height_frequency(arguments["hFreq"])

    # load an environment
    if arguments["loadEnv"] is not None:
        env_repr = env_generator.generate_environment()

        path, name = arguments["loadEnv"]
        env_repr.load(path, name)

        env_generator.load_env_representation(env_repr)

        arguments["dim"] = env_generator.get_dimension()

    # ENVIRONMENT
    # environment characteristics
    environment = GeneralEnvironment(env_generator)
    environment.set_agent_size(arguments["agentSize"])
    environment.set_field_of_view(arguments["fov"])
    environment.activate_turning(arguments["turn"])
    environment.activate_terrain(arguments["terrain"])

    # reward signal - punishment values
    GeneralEnvironment.MOVE_PUNISH = arguments["movePunish"]
    GeneralEnvironment.TERR_PUNISH = arguments["terrainPunish"]
    GeneralEnvironment.OBSTACLE_PUNISH = arguments["obstaclePunish"]

    # reward signal - reward values
    GeneralEnvironment.DISC_REWARD = arguments["discoverReward"]
    GeneralEnvironment.CC_REWARD = arguments["coverageReward"]

    # max step multiplier
    GeneralEnvironment.MAX_STEP_MULTIPLIER = arguments["maxStepMultiplier"]

    # NETWORK GENERATOR
    state_shape = environment.get_state_shape()
    network_generator = GENERATORS[arguments["networkGen"]](
        (state_shape[1], state_shape[2]),
        state_shape[0],
        environment.get_nb_actions(),
        device
    )

    # RL AGENT
    agent_class = AGENTS[arguments["rlAgent"]]
    agent_class.EPSILON_DECAY = arguments["epsilonDecay"]
    agent_class.GAMMA = arguments["gamma"]
    agent_class.TARGET_UPDATE = arguments["targetUpdate"]
    agent_class.QUEUE_LENGTH = arguments["queueLength"]
    agent_class.LEARNING_RATE = arguments["lr"]

    agent = agent_class(
        network_generator,
        OPTIMIZERS[arguments["optim"]],
        environment.get_nb_actions()
    )

    if  arguments['loadEpisode'] is not None:
        agent.load(arguments['loadPath'], arguments['loadEpisode'])
        arguments['loadings'].append(arguments['loadEpisode'])

    if not trainer_required:
        return environment, agent

    # TRAINER
    DeepRLTrainer.NB_EPISODES = arguments["nbEpisodes"]
    DeepRLTrainer.INFO_EVERY = arguments["printEvery"]
    DeepRLTrainer.SAVE_EVERY = arguments["saveEvery"]
    DeepRLTrainer.DEVICE = device

    trainer = DeepRLTrainer(environment, agent, arguments["savePath"])

    return environment, agent, trainer


if __name__ == "__main__":
    path = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/8x_terrain/trial_3/"
    arguments = load_arguments(path, "arguments")

    env, agent = initialize_objects(arguments)

    print(env.MOVE_PUNISHMENT)
    print(env.TERRAIN_PUNISHMENT)
    print(env.OBSTACLE_PUNISHMENT)
    print(env.DISCOVER_REWARD)
    print(env.COVERAGE_REWARD)
    print(env.MAX_STEP_MULTIPLIER)

    print(agent.GAMMA)
    print(agent.EPSILON_DECAY)
    print(agent.TARGET_UPDATE)

    reset1 = env.reset()
    reset2 = env.reset()
    print(reset1 == reset2)