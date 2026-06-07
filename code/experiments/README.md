# Experiment Code

This folder contains the scripts used to reproduce the quantitative and qualitative experiments from Chapter 6 of the thesis.

## Structure

- `exp1_performance_comparison/`
  - Evaluates policies on the same budget distribution used during training.
  - Used for the main comparison of success rate, mean hazard exposure, and CVaR.

- `exp2_time_vs_risk_analysis/`
  - Evaluates policies under fixed-budget sweeps.
  - Used to study how risk and success change as the available time budget varies.

- `trajectories_analysis/`
  - Collects and plots representative trajectories.
  - Used for qualitative interpretation of policy behavior.

## Fair Evaluation Rules

The experiment scripts are designed to keep comparisons fair by using:

- the same environment configuration for all policies;
- the same episode seeds across policies;
- the same budget sequence across policies;
- deterministic policy actions during evaluation;
- the same max horizon (`220`) and budget range (`120..220`).

## Thesis-Scale Runs

The main thesis evaluation uses:

- Experiment 1: `2000` episodes per seed, seeds `2208`, `2306`, `3101`;
- Experiment 2: `300` episodes per seed, seeds `1900`, `1940`, `1963`, `2010`, `2026`;
- CVaR risk level: `alpha = 0.1`;
- switch thresholds saved in results: `0.4`, `0.5`, `0.6`.

The exact commands are documented in the top-level README.

## Outputs

Experiment scripts write CSV/JSON tables to `results/tables/`. Plotting scripts write figures to `results/figures/`.

