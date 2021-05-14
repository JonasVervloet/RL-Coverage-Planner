import sys, getopt
import torch.optim as optim
import json

from networks.simple_q_network import SimpleDeepQNetworkGenerator
from networks.simple_q_network import SimpleDeepQNetworkGenerator2

from deep_rl.deep_q_agent import DeepQAgent
from deep_rl.double_dqn_agent import DoubleDeepQAgent
from deep_rl.trainer import DeepRLTrainer

from load import load_arguments, default_arguments, initialize_objects

SHORT_OPTIONS = ""
LONG_OPTIONS = [
    "loadArguments=",

    "disableCuda",

    "dim=",
    "hFreq=",
    "oFreq=",
    "fillRatio=",
    "loadEnv=",

    "agentSize=",
    "fov=",
    "turn",
    "terrain",

    "movePunish=",
    "terrainPunish=",
    "obstaclePunish=",
    "discoverReward=",
    "coverageReward=",
    "maxStepMultiplier=",

    "gamma=",
    "networkGen=",
    "rlAgent=",
    "epsilonDecay=",
    "targetUpdate=",
    "queueLength=",
    "optim=",

    "nbEpisodes=",
    "printEvery=",
    "saveEvery=",
    "savePath="
]

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

def main(argv):
    try:
        options, args = getopt.getopt(argv, SHORT_OPTIONS, LONG_OPTIONS)
    except getopt.GetoptError:
        print("badly formatted command line arguments")

    arguments = default_arguments()

    for option, argument in options:
        if option == "--loadArguments":
            argument_split = argument.split(",")
            arguments.update(load_arguments(argument_split[0], argument_split[1]))

        if option == "--disableCuda":
            arguments["cuda"] = False

        if option == "--dim":
            arguments["dim"] = tuple(tuple(map(int, argument.split(","))))

        if option == "--hFreq":
            arguments["hFreq"] = tuple(map(int, argument.split(",")))

        if option == "--oFreq":
            arguments["oFreq"] = tuple(map(int, argument.split(",")))

        if option == "--fillRatio":
            arguments["fillRatio"] = float(argument)

        if option == "--loadEnv":
            arguments["loadEnv"] = tuple(argument.split(","))

        if option == "--agentSize":
            arguments["agentSize"] = int(argument)

        if option == "--fov":
            arguments["fov"] = int(argument)

        if option == "--turn":
            arguments["turn"] = True

        if option == "--terrain":
            arguments["terrain"] = True

        if option == "--movePunish":
            arguments["movePunish"] = float(argument)

        if option == "--terrainPunish":
            arguments["terrainPunish"] = float(argument)

        if option == "--obstaclePunish":
            arguments["obstaclePunish"] = float(argument)

        if option == "--discoverReward":
            arguments["discoverReward"] = float(argument)

        if option == "--coverageReward":
            arguments["coverageReward"] = float(argument)

        if option == "--maxStepMultiplier":
            arguments["maxStepMultiplier"] = int(argument)

        if option == "--gamma":
            arguments["gamma"] = float(argument)
            assert(float(argument) <= 1.0)

        if option == "--networkGen":
            if argument in GENERATORS:
                arguments["networkGen"] = argument
            else:
                raise Exception("TRAIN.py: given network generator is not defined...")

        if option == "--optim":
            if argument in OPTIMIZERS:
                arguments["optim"] = argument
            else:
                raise Exception("TRAIN.py: given optimizer is not defined...")

        if option == "--rlAgent":
            if argument in AGENTS:
                arguments["rlAgent"] = argument
            else:
                raise Exception("TRAIN.py: given agent is not defined...")

        if option == "--epsilonDecay":
            arguments["epsilonDecay"] = int(argument)

        if option == "--targetUpdate":
            arguments["targetUpdate"] = int(argument)

        if option == "--queueLength":
            arguments["queueLength"] = int(argument)

        if option == "--nbEpisodes":
            arguments["nbEpisodes"] = int(argument)

        if option == "--printEvery":
            arguments["printEvery"] = int(argument)

        if option == "--saveEvery":
            arguments["saveEvery"] = int(argument)

        if option == "--savePath":
            arguments["savePath"] = argument

    with open(f"{arguments['savePath']}arguments.txt", 'w') as output_file:
        json.dump(arguments, output_file)

    env, agent, trainer = initialize_objects(arguments, trainer_required=True)

    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])