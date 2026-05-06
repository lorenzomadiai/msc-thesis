# Adaptive Risk-Aware Reinforcement Learning under Time Constraints

This thesis project studies a risk- and time-aware Reinforcement Learning framework designed for environments with variable time budgets. The goal is to adapt the agent's behavior to the mission context: more cautious when time allows it, more aggressive when the time constraint becomes tight.

## Overview

The problem addressed here is how to combine performance and safety dynamically. In traditional methods, the trade-off between task success and risk is often fixed; here, instead, the acceptable risk level depends on the remaining time and on the current episode state.

The project therefore introduces a hierarchical switching method that adaptively selects the most suitable policy for the current conditions.

## Project Structure

The code and results are organized to clearly separate baseline training, the proposed method, and evaluation.

### `thesis_project/src/algorithms`

This folder contains the main baselines:

- `sac_timeaware.py`: this file provides the flat policy and the aggressive baseline. Both represent the more performance-oriented version, without the switching mechanism.
- `wcsac_timeaware.py`: this file provides the conservative policy, trained with a more cautious and safety-oriented setup.

### `thesis_project/src/proposed_method`

This folder contains the implementation of the proposed method:

- `meta_env.py`: environment/metastructure logic used to manage the switching behavior.
- `build_episode_pool.py`: episode-pool construction used to generate data and evaluate policy behavior.
- `train_classifier_switch.py`: training of the switching classifier.
- `models/`: models used by the method.
- `common/` and `utils/`: shared support functions.
- `evaluation/`: internal tools for analysis and evaluation.

### `thesis_project/src/evaluation`

This folder contains everything needed to compare the policies and the proposed method on the main metrics:

- `data_collection/`: collection and preparation of evaluation data.
- `plotting/`: scripts for visualizing the results.

The analyses cover metrics such as:

- success rate,
- CVaR as a risk measure,
- comparison between conservative, aggressive, flat, and switching-based policies.

## Method

The framework is composed of three logical levels:

1. a **conservative policy**, obtained from `wcsac_timeaware.py`, which prioritizes safety;
2. an **aggressive policy** and a **flat policy**, obtained from `sac_timeaware.py`, which are more performance-oriented;
3. a **switching controller** that decides when to move from one behavior to the other based on the mission context.

The switching mechanism is formulated as an optimal stopping problem and implemented through a supervised classifier that learns when it is beneficial to switch policy.

## Experimental Setup

The experiments were carried out on the OpenAI Safety Gym benchmark, in particular on continuous navigation tasks with:

- hazardous regions that generate safety costs,
- variable time budgets,
- time-aware state representations.

The observation includes information such as:

- hazard LiDAR,
- goal LiDAR,
- robot velocity,
- remaining time,
- normalized mission budget.

## Results and Analysis

The project compares the baselines and the proposed method using metrics such as:

- task success rate,
- risk exposure,
- CVaR in high-budget regimes.

The results show that:

- flat policies struggle to represent different risk profiles with a single fixed behavior;
- an overly conservative policy hurts performance when little time is available;
- hierarchical switching enables more adaptive, context-dependent behavior.

## External Dependencies

The work relies on two key external repositories:

- **WCSAC**, for the conservative side and as a SafeRL reference;
- **Safety Gym**, for the experimental environment and navigation tasks with safety constraints.

## Technologies

- Python
- PyTorch
- OpenAI Safety Gym
- MuJoCo
- Reinforcement Learning
- Safe Reinforcement Learning
- CVaR optimization

## Thesis Title

**Learning When to Switch: Adaptive Risk-Aware Reinforcement Learning under Time Constraints**

Leiden University — MSc Computer Science (Artificial Intelligence).
