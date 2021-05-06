import json
import pprint
import torch
import torch.optim as optim

from environments.env_generation import EnvironmentGenerator, SingleEnvironmentGenerator
from environments.environment import Environment, EnvironmentFOV
from environments.turn_environment import SimpleTurnEnvironment

from networks.simple_q_network import SimpleDeepQNetworkGenerator
from networks.simple_q_network import SimpleDeepQNetworkGenerator2

from deep_rl.deep_q_agent import DeepQAgent
from deep_rl.double_dqn_agent import DoubleDeepQAgent
from deep_rl.trainer import DeepRLTrainer

DEFAULT_ARGUMENTS = {
    "heightRequired": False,
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
    "fov": None,
    "turnEnv": False,
    "networkGen": "simpleQ",
    "rlAgent": "deepQ",
    "inputMatch": True,
    "agentInpDepth": 3,
    "optim": "rmsProp",
    "gamma": 0.9,
    "epsilonDecay": 2000,
    "targetUpdate": 1000,
    "nbEpisodes": 2000,
    "printEvery": 50,
    "saveEvery": 250,
    "softmax": False,
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


def load_arguments(path, name ="arguments"):
    with open(f"{path}{name}.txt") as input_file:
        return json.load(input_file)


def initialize_objects(args, trainer_required=False):
    arguments = default_arguments()
    arguments.update(args)

    print("Initializing objects...")
    print(arguments)

    if arguments["loadEnv"] is not None:
        env_generator = SingleEnvironmentGenerator(arguments["loadEnv"])
        arguments["dim"] = env_generator.get_dimension()
    else:
        env_generator = EnvironmentGenerator(arguments["heightRequired"])
        env_generator.set_dimension(arguments["dim"])
        env_generator.set_height_frequency(arguments["hFreq"])
        env_generator.set_obstacle_frequency(arguments["oFreq"])
        env_generator.set_fill_ratio(arguments["fillRatio"])

    if arguments["fov"] is not None and arguments["turnEnv"]:
        raise Exception("FOV and TurnEnvironment combination not yet supported!!")

    if arguments["fov"] is None and not arguments["turnEnv"]:
        env = Environment(env_generator)
    elif arguments["fov"] is not None:
        env = EnvironmentFOV(env_generator)
        env.set_fov(arguments["fov"])
    else:
        env = SimpleTurnEnvironment(env_generator)

    env.MOVE_PUNISHMENT = arguments["movePunish"]
    env.TERRAIN_PUNISHMENT = arguments["terrainPunish"]
    env.OBSTACLE_PUNISHMENT = arguments["obstaclePunish"]
    env.DISCOVER_REWARD = arguments["discoverReward"]
    env.COVERAGE_REWARD = arguments["coverageReward"]
    env.MAX_STEP_MULTIPLIER = arguments["maxStepMultiplier"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DEVICE: {device}")

    network_generator = GENERATORS[arguments["networkGen"]](
        env.get_state_dimension(),
        env.get_input_depth() if arguments["inputMatch"] else arguments["agentInpDepth"],
        env.get_nb_actions(),
        device
    )
    agent = AGENTS[arguments["rlAgent"]](
        network_generator,
        OPTIMIZERS[arguments["optim"]],
        env.get_nb_actions()
    )
    agent.EPSILON_DECAY = arguments["epsilonDecay"]
    agent.GAMMA = arguments["gamma"]
    agent.TARGET_UPDATE = arguments["targetUpdate"]

    if not trainer_required:
        return env, agent

    DeepRLTrainer.NB_EPISODES = arguments["nbEpisodes"]
    DeepRLTrainer.INFO_EVERY = arguments["printEvery"]
    DeepRLTrainer.SAVE_EVERY = arguments["saveEvery"]
    DeepRLTrainer.SOFT_MAX = arguments["softmax"]
    DeepRLTrainer.DEVICE = device

    trainer = DeepRLTrainer(env, agent, arguments["savePath"])

    return env, agent, trainer


if __name__ == "__main__":
    path = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/8x_terrain/trial_3/"
    arguments = load_arguments(path, "arguments")

    pprint.pprint(arguments)

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