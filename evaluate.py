import numpy as np
import sys, getopt
import torch

from load import default_arguments, load_arguments, initialize_objects

SHORT_OPTIONS = ""
LONG_OPTIONS = [
    "generalPath=",
    "folder="
]

STEP = 250
START = STEP
NB_RUNS = 300


def load_env_agent(folder):
    arguments = default_arguments()
    arguments.update(load_arguments(folder, "arguments"))

    arguments["cuda"] = False

    if arguments["loadEnv"] is not None:
        print("not yet supported...")

    env, agent = initialize_objects(arguments)

    return env, agent, arguments["nbEpisodes"]


def main(argv):
    path = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/"
    folder = "SERVER/env_8x/fov5/trial3/"

    try:
        options, args = getopt.getopt(argv, SHORT_OPTIONS, LONG_OPTIONS)
    except getopt.GetoptError:
        print("badly formatted command line arguments")

    for option, argument in options:
        if option == "--generalPath":
            path = argument

        if option == "--folder":
            folder = argument

    percentages = []

    tile_efficiencies = []
    tile_efficiencies2 = []

    terrain_efficiencies = []
    terrain_efficiencies2 = []

    rel_terrain_efficiencies = []
    rel_terrain_efficiencies2 = []

    env, agent, nb_episodes = load_env_agent(path + folder)

    for episode_nb in range(START, nb_episodes + STEP, STEP):
        print(f"episode nb: {episode_nb}")
        agent.load(path + folder, episode_nb, 'cpu')
        agent.evaluate()

        count = 0

        tile_eff_runs = []
        tile_eff_runs2 = []

        terrain_eff_runs = []
        terrain_eff_runs2 = []

        rel_terrain_eff_runs = []
        rel_terrain_eff_runs2 = []

        for i in range(NB_RUNS):
            done = False
            info = {}
            current_state = env.reset()

            while not done:
                state = current_state
                action = agent.select_action(torch.tensor(state, dtype=torch.float))
                current_state, _, done, info = env.step(action)

            tile_eff_runs2.append(info['nb_steps'] / info['total_covered_tiles'])
            terrain_eff_runs2.append(info['total_pos_terr_diff'])
            rel_terrain_eff_runs2.append(info['total_pos_terr_diff'] / info['total_covered_tiles'])

            if info['full_cc']:
                count += 1
                tile_eff_runs.append(info['nb_steps'] / info['total_covered_tiles'])
                terrain_eff_runs.append(info['total_pos_terr_diff'])
                rel_terrain_eff_runs.append(info['total_pos_terr_diff'] / info['total_covered_tiles'])

        percentages.append(count / NB_RUNS)

        tile_efficiencies.append(np.average(tile_eff_runs) if len(tile_eff_runs) > 0 else 0)
        tile_efficiencies2.append(np.average(tile_eff_runs2))

        terrain_efficiencies.append(np.average(terrain_eff_runs) if len(terrain_eff_runs) > 0 else 0)
        terrain_efficiencies2.append(np.average(terrain_eff_runs2))

        rel_terrain_efficiencies.append(np.average(rel_terrain_eff_runs) if len(rel_terrain_eff_runs) > 0 else 0)
        rel_terrain_efficiencies2.append(np.average(rel_terrain_eff_runs2))

        print(f"percentage: {percentages[-1]}")

        print(f"tile efficiency: {tile_efficiencies[-1]}")
        print(f"terrain efficiencies: {terrain_efficiencies[-1]}")
        print(f"relative terrain efficiencies: {rel_terrain_efficiencies[-1]}")

    print("saving....")

    np.save(path + folder + "eval_cover_percentages", percentages)

    np.save(path + folder + "eval_tile_efficiency", tile_efficiencies)
    np.save(path + folder + "eval_tile_efficiency2", tile_efficiencies2)

    np.save(path + folder + "eval_terrain_differences", terrain_efficiencies)
    np.save(path + folder + "eval_terrain_differences2", terrain_efficiencies2)

    np.save(path + folder + "eval_terrain_efficiency", rel_terrain_efficiencies)
    np.save(path + folder + "eval_terrain_efficiency2", rel_terrain_efficiencies2)

    print("saving done!")


if __name__ == "__main__":
    main(sys.argv[1:])