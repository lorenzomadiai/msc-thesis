# Results

This folder contains generated tables and figures used to analyse the thesis experiments.

## Contents

- `tables/`
  - CSV and JSON outputs from evaluation and classifier-analysis scripts.

- `figures/`
  - Plots generated from the result tables.

- `other/`
  - Extra or archived outputs not part of the main thesis reproduction path.

## Main Result Groups

- `exp1_performance_comparison`
  - Main policy comparison under the training budget distribution.
  - Includes success rate, mean hazard exposure, and CVaR analysis.

- `exp2_time_vs_risk_analysis`
  - Fixed-budget sweep showing how performance and risk change with available time.

- `classifier_evaluation`
  - Classifier quality, confusion matrices, threshold sweeps, and switch-timing diagnostics.

- `trajectories`
  - Qualitative trajectory visualizations.

## Naming Convention

- `traindist_timeaware_seed*_eps2000_Bmin120_Bmax220_H220_6agents_2000ep.csv`
  - Experiment 1, thesis-scale evaluation.

- `fixedbudget_sweep_timeaware_seed*_eps300_Bmin120_Bmax220_Bstep10_H220_7agents_1ep.csv`
  - Experiment 2, fixed-budget sweep.

## Note

Results are generated artifacts. If you regenerate them, keep old thesis outputs or use a new tag so comparisons remain traceable.

