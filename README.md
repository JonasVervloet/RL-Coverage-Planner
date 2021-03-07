# Terrain-aware Complete Coverage Path Planning using Deep Reinforcement Learning
This repository applies deep reinforcement learning to the complete coverage path planning problem.
The aim for the project is to research if a RL-agent can be trained to take terrain information into account.
That is, to see whether a RL-agent will adapt its path for complete-coverage planning when the agent is presented with terrain-information.
This will enable the agent to be more energy-efficient than classical planning methods.

## Installation
After cloning the repository, create a conda environment using the `environment.yml file`.

```
conda env create -f environment.yml
conda activate RL
```

These commands should install all the needed dependencies.

## Run the code

### Environment Dashboard

```
cd tutorials/dash
python3 dash_tutorial.py
```

Then open a browser and go to [localhost:8050](http://127.0.0.1:8050/).

### Train DQN-agent

To train a Deep-Q-Network agent on a single environment:

```
cd deep_q_learning
python3 dqn_train.py
```

This will train an agent for 1000 episodes on a single environment.
The environment is a 16x16 grid with 120 tiles that need to be covered.
The agent uses a neural architecture consisting of two convolutional layers,
followed by a fully connected layer that outputs the Q-values for every action.

### Visualise training results

The following command renders the training result.
It will display a window that shows the agent interacting in the environment.
While during training, an epsilon-greedy policy is used, the agent acts in this visualization according to a completely greedy policy.

```
cd deep_q_learning
python3 dqn_visualization.py
```

After running the command, you should see something like below:
[![First Results](https://cdn.loom.com/sessions/thumbnails/7e1c09999b734aff80286c7d247d0ee6-1614948407802-with-play.gif)](https://www.loom.com/share/7e1c09999b734aff80286c7d247d0ee6)
