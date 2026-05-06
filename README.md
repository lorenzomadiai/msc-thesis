# Adaptive Risk-Aware Reinforcement Learning under Time Constraints

Adaptive Risk-Aware Reinforcement Learning framework for time-constrained environments. This MSc thesis studies how RL agents dynamically switch between conservative and aggressive policies based on mission time and risk conditions, combining SafeRL, CVaR-based constraints, and hierarchical policy switching in Safety Gym.

---

## Overview

Traditional Reinforcement Learning (RL) and Safe Reinforcement Learning (SafeRL) methods usually optimize a fixed trade-off between task performance and safety. However, in many real-world and mission-critical scenarios, the acceptable level of risk depends on the operational context and the remaining available time.

This project studies a time-constrained RL setting where:

- each episode is assigned a variable mission time budget,
- safer behavior should be preferred when enough time is available,
- more aggressive and risk-tolerant behavior may become necessary under tight deadlines.

To address this problem, the project proposes a hierarchical policy-switching framework that dynamically adapts the risk attitude of the agent according to the mission conditions.

---

## Method

The framework is composed of:

- a **conservative policy** trained to minimize hazard exposure and satisfy safety constraints,
- an **aggressive policy** trained to maximize task completion speed,
- a **high-level switching controller** that decides when to switch between the two behaviors.

The switching mechanism is modeled as an optimal stopping problem and implemented through a supervised switching classifier.

---

## Environment

Experiments are conducted using the OpenAI Safety Gym benchmark.

The environment consists of:

- continuous-control navigation tasks,
- hazardous regions generating safety costs,
- variable mission time budgets,
- time-aware state representations.

The observation space includes:

- hazard LiDAR observations,
- goal LiDAR observations,
- robot velocity,
- remaining mission time,
- normalized mission budget.

---

## Results

The proposed switching framework achieves:

- high task success rates,
- reduced hazard exposure compared to aggressive baselines,
- satisfaction of CVaR safety constraints in high-budget regimes.

The experiments show that:

- flat policies struggle to represent multiple risk profiles simultaneously,
- fixed conservative policies fail under strong time pressure,
- hierarchical switching enables adaptive and context-dependent behavior.

---

## Technologies

- Python
- PyTorch
- OpenAI Safety Gym
- MuJoCo
- Reinforcement Learning
- Safe Reinforcement Learning
- CVaR optimization


**Learning When to Switch: Adaptive Risk-Aware Reinforcement Learning under Time Constraints**

Leiden University — MSc Computer Science (Artificial Intelligence).
