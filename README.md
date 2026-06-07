# Learning When to Switch: Adapting Risk Attitude to Reach a Goal Under Time Constraints

This repository contains the code, models, datasets, and experimental outputs for a thesis project on adaptive risk-aware reinforcement learning under time constraints.

The central research question is: how can an agent adapt its risk attitude when the available mission time changes? A risk-aware policy is safer but can be too slow under tight deadlines. An aggressive policy is faster but may incur more safety cost. The proposed method learns a high-level switching controller that starts from the risk-aware policy and decides when it is useful to switch irreversibly to the aggressive policy.

The work uses Safety Gym navigation tasks with hazards, continuous control, and explicit time budgets. The proposed method is implemented as a hierarchical controller that decides which policy deploy, the risk-aware or the aggressive one, at each decision step. Specifically, the meta-controller is a neural network that acts as a classifier, determining at each observed state whether switching to the aggressive policy is more beneficial than continuing the episode with the risk-aware policy.

## Thesis-to-Repository Map

This table is intended to map the content discussed in the thesis to the corresponding files in the GitHub repository.
| Thesis part | Main idea | Repository location |
| --- | --- | --- |
| Chapter 4, 5.4, Proposed Method | Irreversible optimal-stopping switch from conservative to aggressive policy, Episode pool, oracle labels, 36-D features, MLP switch classifier | `code/proposed_method/meta_env.py`, `code/proposed_method/common/oracle.py` `code/proposed_method/build_episode_pool.py`, `code/proposed_method/train_switching_classifier.py`  `models/` |
| Chapter 5.3, Baselines | SAC aggressive policy, SAC flat policy, WCSAC conservative policy | `code/baselines/sac_timeaware.py`, `code/baselines/wcsac_timeaware.py`, `models/` |
| Chapter 6, Results | Policy comparison, fixed-budget sweep, classifier quality, switch-timing analysis | `code/experiments/`, `code/proposed_method/evaluation/`, `results/` |
## Repository Contents

```text
thesis_project/
  code/
    baselines/
    proposed_method/
    experiments/
  data/
    pools_of_episodes/
    datasets/
  models/
    aggressive_policy/
    conservative_policy/
    flat_policy/
    switching_classifier/
  results/
    tables/
    figures/
  docs/
  externals/
  environment.yml
  environment_windows.yml
  requirements.txt
```

## Role of Each Directory

### `code/baselines`

This folder contains the reinforcement learning algorithms used to train the baseline policies; it contains also the wrapper to include time limits in the environment.

- `sac_timeaware.py` trains SAC-style time-aware policies.
  - It is used for the aggressive policy, which optimizes task completion more directly.
  - It is also used for the flat policy, which uses a fixed cost penalty instead of adaptive switching.
  - It adds `TimeBudgetWrapper`, deadline-aware termination, and time-budget observations.

- `wcsac_timeaware.py` trains the conservative risk-aware policy.
  - It is based on the WCSAC implementation present in `externals/WCSAC`.
  - It adds the same time-budget wrapper and Safety Gym setup used by the SAC baseline.
  - It optimizes the same objective function of the other algorithm but at the same time is satisfies a risk-based safety costraint related to the exposure to the hazards.

- `utils/wrappers.py` defines `TimeBudgetWrapper`.
  - The wrapper appends normalized remaining time and normalized mission budget to the observation.
  - It terminates an episode when the budget expires.
  - It optionally adds a deadline penalty.

The baseline policies are the foundation of the rest of the project. The proposed method does not learn low-level policies from scratch but it learns when to switch between already-trained low-level policies.

### `code/proposed_method`

This folder contains the proposed hierarchical switching method.

- `meta_env.py` defines the meta-level environment at which the meta-controller works:
  - Action `0` means keep using the conservative policy.
  - Action `1` means switch to the aggressive policy.
  - In the thesis setting the switch is irreversible: after switching, the aggressive policy is used until the end of the episode.

- `build_episode_pool.py` creates an offline pool of seeded episodes.
  - The pool stores episode seeds, sampled budgets, conservative-policy success/failure, and the best time to switch using the heuristic presented in the thesis report.
  - One pool of episodes is used to generate the training set and another pool of episodes is used to generate the test set.

- `train_switching_classifier.py` trains the learned meta-controller.
  - It uses the episode pool to generate a dataset of observation states (lidar info, velocity, time available) and the respective label.
  - The respective gap labels are of the form `delta(k) = Return_switch(k) - Return_wait(k)`.
  - It trains a binary classifier predicting whether switching is better than waiting based on the current observation state.
  - It saves `switching_model.pt`, `config.json`, `dataset_summary.json` and `train_history.csv`.

- `common/` contains shared logic:
  - `config.py`: Safety Gym configuration.
  - `features.py`: feature extraction for the classifier.
  - `policy_loader.py`: TensorFlow SavedModel loading for low-level policies.
  - `mujoco_state.py`: saving/restoring MuJoCo state for counterfactual oracle rollouts.
  - `oracle.py`: oracle return computation and best-switch search.

- `models/` contains the PyTorch classifier architecture.
  - `delta_net.py` defines `DeltaNet`, a small MLP used as the switching classifier.

- `utils/` contains dataset sampling, train/validation/test splitting, evaluation, and training utilities.

- `evaluation/` contains scripts for analysing classifier quality, switch timing, oracle behaviour, and per-episode switching probability.

### `code/experiments`

This folder contains scripts used to generate the thesis experiments after policies and the classifier are available.

- `exp1_performance_comparison/`
  - Evaluates all policies on the training budget distribution.
  - Produces comparison tables and plots for success rate, cost, and CVaR.

- `exp2_time_vs_risk_analysis/`
  - Evaluates policies over fixed-budget sweeps.
  - This isolates how behaviour changes as the time budget becomes tighter for the same tasks/missions.

- `trajectories_analysis/`
  - Collects and plots trajectories followed by the different policies to reach the goal.

### `data`

This folder stores intermediate reproducibility artifacts.

- `data/pools_of_episodes/for_training/`
  - Offline episode pool used to train the switching classifier.

- `data/pools_of_episodes/for_testing/`
  - Episode pool used for testing/evaluation.

- `data/datasets/training_set/`
  - Cached supervised dataset derived from the training episode pool.

- `data/datasets/test_set/`
  - Cached dataset derived from the testing episode pool.

These files are useful because the labelling process is expensive. Reusing the cached pools and datasets makes reproduction faster and more stable.

### `models`

This folder stores trained artifacts.

- `models/aggressive_policy/`
  - SAC time-aware policy used as the risk-seeking policy.

- `models/conservative_policy/`
  - WCSAC time-aware policy used as the risk-aware conservative policy.

- `models/flat_policy/`
  - Single-policy baseline with a fixed reward-cost trade-off.

- `models/switching_classifier/`
  - Learned meta-controller proposed in the thesis.
  - Main checkpoint: `switching_model.pt`.
  - Configuration: `config.json`.

The TensorFlow policy folders contain `saved_model.pb`, `variables/`, `config.json`, and training logs. The PyTorch classifier folder contains the classifier checkpoint and training history.

### `results`

This folder contains generated thesis outputs.

- `results/tables/`
  - CSV/JSON tables for the main quantitative experiments.

- `results/figures/`
  - Plots used for analysis and thesis figures.


### `docs`

This folder contains thesis documentation, including:

- `docs/thesis_madiai.pdf`

### `externals`

This folder contains local copies of third-party projects required by the thesis code.

- `externals/WCSAC`
  - Source implementation used as the basis for risk-aware SAC.
  - Original project: <https://github.com/AlgTUDelft/WCSAC>

- `externals/safety-gym`
  - Safety Gym benchmark environment.
  - Original project: <https://github.com/openai/safety-gym>


## Method Summary

The final architecture has two levels.

At the low level:

1. The conservative policy tries to reach the goal while limiting safety cost.
2. The aggressive policy tries to reach the goal quickly.
3. The flat policy is a comparison baseline that uses one fixed risk-performance trade-off in a single policy.

At the high level:

1. The classifier observes current state features.
2. It predicts `P(switch is better than waiting)`.
3. If the probability is above a threshold, the controller switches from conservative to aggressive.
4. The switch is irreversible.

## Exact Thesis Setup

The main experiments use the following setup from Chapters 3-6 of the thesis.

### Environment

| Quantity | Value |
| --- | --- |
| Environment | OpenAI Safety Gym |
| Task | Point-goal navigation |
| Robot | Point robot |
| Arena extents | `[-1.5, -1.5, 1.5, 1.5]` |
| Goal location | `(1.1, 1.1)` |
| Goal size / keepout | `0.3` / `0.305` |
| Hazard location | `(0, 0)` |
| Hazard size / keepout | `0.7` / `0.705` |
| Number of hazards | `1` |
| Lidar bins | `16` goal bins + `16` hazard bins |
| Lidar max distance | `3` |

The environment configuration is centralized in `code/proposed_method/common/config.py` and mirrored in the experiment scripts.

### Time-Budget Formulation

Each episode samples a mission budget from:

```text
B ~ Uniform({120, 125, 130, ..., 220})
```

The policies observe the original Safety Gym observation plus:

```text
time_left_norm = 2 * (B - t) / B - 1
budget_norm    = B / Bmax
```

These two values are appended by `TimeBudgetWrapper`.

### Risk and Utility Parameters

| Quantity | Thesis value |
|----------|-------------|
| Budget minimum | `120` |
| Budget maximum | `220` |
| Budget step | `5` |
| Max horizon | `220` |
| Hazard weight `lambda` | `0.02` |
| CVaR risk level `alpha` | `0.1` |
| CVaR risk bound `d` | `5` |
| Deadline penalty | `1.0` |

The high-budget regime is the upper quantile of the budget distribution. In the thesis, CVaR is evaluated in this high-budget regime because this is the regime where the safety requirement is considered operationally feasible. In the thesis setting, the high-budget regime consists of episodes whose budget belongs to the top quartile of the budget distribution (i.e., above the 75th percentile).

### Classifier Setup

| Quantity | Value |
| --- | --- |
| Input dimension | `36` |
| Feature layout | `hazard_lidar(16) + goal_lidar(16) + v_x + v_y + time_left_norm + budget_norm` |
| Model | MLP with one hidden layer |
| Hidden units | `32` |
| Trainable parameters | `1217` |
| Loss | BCE with logits |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Batch size | `16` |
| Max epochs | `500` |
| Early stopping patience | `20` |
| Feature noise std | `0.01` |
| Train/validation/test split | `0.8 / 0.1 / 0.1` |
| Decision threshold | `0.5` |
| Base seed | `2001` |

The thesis classifier test metrics are:

| Metric | Value |
| --- | --- |
| ROC-AUC | `0.947` |
| PR-AUC | `0.697` |
| Brier score | `0.072` |
| Precision | `0.751` |
| Recall | `0.583` |
| F1-score | `0.657` |

## Cloning the Repository

This project depends on two external repositories located in `externals/`:

- WCSAC
- Safety Gym

These repositories are required to reproduce the thesis experiments.

### Clone with Submodules

If the repository is configured with Git submodules, clone everything with:

```bash
git clone --recurse-submodules https://github.com/lorenzomadiai/msc-thesis.git
cd msc-thesis
```

If the repository has already been cloned, initialize the submodules with:

```bash
git submodule update --init --recursive
```

### Manual Installation of External Dependencies

If submodules are not available, clone the required repositories manually:

```bash
git clone https://github.com/AlgTUDelft/WCSAC.git externals/WCSAC
git clone https://github.com/openai/safety-gym.git externals/safety-gym
```

The environment files and installation instructions assume that both repositories are available under the `externals/` directory.

### Safety Gym Modifications
Safety Gym is used as the benchmark environment for the thesis experiments. A small compatibility patch was applied to make the environment work correctly with the local Windows and MuJoCo setup used in this project.

#### MuJoCo Dependency Patch

The original Safety Gym `setup.py` explicitly requires:

```python
mujoco_py==2.0.2.7
```
The project environment already installs the compatible version of `mujoco-py` specified in the main environment files. Therefore, Safety Gym is installed as a local editable package without forcing its original pinned MuJoCo dependency.

#### Modification in `world.py`

At approximately line 290 of `safety_gym/envs/world.py`, the original code loads the MuJoCo model directly from an XML string:

```python
self.model = load_model_from_xml(self.xml_string)
```

This was replaced by a workaround that first writes the generated XML to a temporary file and then loads the model using:

```python
import tempfile
from mujoco_py import load_model_from_path

with tempfile.NamedTemporaryFile(
    mode="w",
    suffix=".xml",
    delete=False,
    encoding="utf-8"
) as f:
    f.write(self.xml_string)
    xml_path = f.name

self.model = load_model_from_path(xml_path)
```

This patch was necessary because, on the Windows setup used for this project, `load_model_from_xml()` failed when loading the dynamically generated Safety Gym model, whereas `load_model_from_path()` worked correctly.

#### Seed Compatibility Patch

In `externals/safety-gym/safety_gym/envs/engine.py`, the automatic seed generation was patched to avoid a Windows/NumPy error:

```python
self._seed = np.random.randint(2**32) if seed is None else seed
```

was replaced with:

```python
self._seed = np.random.randint(2**31 - 1) if seed is None else seed
```

This avoids `ValueError: high is out of bounds for int32` without changing the task dynamics, observations, rewards, or costs.
## Environment Reproduction

The project uses an old RL stack:

- Python 3.7.12
- TensorFlow 1.13.1
- PyTorch 1.13.1
- Gym 0.15.7
- Safety Gym
- MuJoCo 2.0 / `mujoco-py`
- MPI / `mpi4py`

The recommended environment file for the current repository layout is `environment.yml`:

```bash
conda env create -f environment.yml
conda activate th_project
```

On Windows you can also use:

```powershell
conda env create -f environment_windows.yml
conda activate th_project
```

Both environment files install the external repositories in editable mode:

```text
-e ./externals/safety-gym
-e ./externals/WCSAC
```

### Environment Verification

After activating the environment, run these checks from the repository root:

```bash
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "import gym; print('Gym', gym.__version__)"
python -c "import safety_gym, wc_sac; print('Safety Gym and WCSAC imports OK')"
python -c "import sys; sys.path.insert(0, r'code/baselines'); from utils.wrappers import TimeBudgetWrapper; print('TimeBudgetWrapper import OK')"
```

The scripts add the required local paths themselves. The checks above simply verify that the core packages and local wrapper can be imported inside the conda environment.

### MuJoCo Setup

Safety Gym depends on `mujoco-py`, which requires MuJoCo 2.0. On Windows, place MuJoCo at:

```text
C:\Users\<USER>\.mujoco\mujoco200
```

Make sure this directory is on `PATH`:

```text
C:\Users\<USER>\.mujoco\mujoco200\bin
```

On Linux/macOS, set the equivalent MuJoCo library path required by `mujoco-py`.

You should also verify MuJoCo after activating the conda environment:

```bash
python -c "import mujoco_py; print('mujoco-py import OK')"
```

## Reproducibility Principles

Fair reproduction requires using the same:

- environment configuration in `code/proposed_method/common/config.py`;
- budget range `120` to `220`;
- budget step `5` for training/pool generation and `10` for the thesis fixed-budget sweep;
- maximum horizon `220`;
- shared episode seeds across policies;
- shared budget sequence across policies;
- deterministic policy actions at evaluation time;
- same trained low-level policies when evaluating the learned switcher;
- same classifier threshold, usually `0.5`;
- enough evaluation episodes to reduce variance.

The thesis-level protocol is:

- main policy comparison: `2000` episodes per seed, with seeds `2208`, `2306`, and `3101`;
- fixed-budget sweep: `300` episodes per seed, with seeds `1900`, `1940`, `1963`, `2010`, and `2026`;
- classifier training pool: `1000` episodes balanced as `500` conservative successes and `500` conservative failures;
- classifier dataset: approximately `3000` state-level samples from the episode pool;
- CVaR: worst `10%` of episode costs, evaluated for the high-budget regime in the main thesis risk analysis.

For quick debugging, use fewer episodes. For thesis-level reproduction, use the full episode counts above.

## Training the Baseline Policies

Run baseline training from `code/baselines`, because the baseline scripts import `utils.wrappers` relative to that folder.

```powershell
cd code\baselines
```

### Aggressive Policy

The aggressive policy is trained with the time-aware SAC script.

```powershell
python .\sac_timeaware.py `
  --env StaticEnv-v0 `
  --use_time_wrapper `
  --budget_min 120 `
  --budget_max 220 `
  --deadline_penalty 0 `
  --epochs 100 `
  --steps_per_epoch 30000 `
  --update_freq 100 `
  --batch_size 256 `
  --local_start_steps 500 `
  --local_update_after 500 `
  --hid 256 `
  --l 2 `
  --gamma 0.99 `
  --lr 0.001 `
  --lambda 0.0 `
  --seed 0 `
  --cpu 1 `
```

### Flat Policy

The flat policy is also trained with `sac_timeaware.py`, but it is interpreted as a single fixed trade-off baseline. It uses the `--lambda` flag to subtract a fixed hazard-cost penalty from the reward:

```text
reward <- reward - lambda * hazard_cost
```

Example command:

```powershell
python .\sac_timeaware.py `
  --env StaticEnv-v0 `
  --use_time_wrapper `
  --budget_min 120 `
  --budget_max 220 `
  --deadline_penalty 1 `
  --epochs 100 `
  --steps_per_epoch 30000 `
  --update_freq 100 `
  --batch_size 256 `
  --local_start_steps 500 `
  --local_update_after 500 `
  --hid 256 `
  --l 2 `
  --gamma 0.99 `
  --lr 0.001 `
  --lambda 0.02 `
  --seed 0 `
  --cpu 1
```

### Conservative Policy

The conservative policy is trained with the WCSAC-based script.

```powershell
python .\wcsac_timeaware.py `
  --env StaticEnv-v0 `
  --cost_lim 15 `
  --cl 0.1 `
  --epochs 100 `
  --steps_per_epoch 30000 `
  --update_freq 100 `
  --batch_size 256 `
  --local_start_steps 500 `
  --local_update_after 500 `
  --hid 256 `
  --l 2 `
  --gamma 0.99 `
  --lr 0.001 `
  --lr_s 50 `
  --damp_s 10 `
  --seed 0 `
  --cpu 1
```

After training, each policy directory should contain:

```text
saved_model.pb
variables/
config.json
progress.txt
```

## Building the Episode Pool

The episode pool is the offline set of seeds and budgets used to build classifier supervision. In the thesis, this pool is balanced with respect to the conservative policy outcome: `500` episodes where the conservative policy reaches the goal within budget and `500` episodes where it fails. This matters because failed conservative episodes are where switching is most likely to be useful.

```powershell
python .\code\proposed_method\build_episode_pool.py `
  --cons_dir .\models\conservative_policy `
  --agg_dir .\models\aggressive_policy `
  --budget_min 120 `
  --budget_max 220 `
  --budget_step 5 `
  --meta_interval 1 `
  --max_horizon 220 `
  --pool_size 1000 `
  --fail_frac 0.5 `
  --base_seed 1111 `
  --switch_interval 5 `
  --scan_interval 5 `
  --n_top_zones 2 `
  --output_csv .\data\pools_of_episodes\for_training\pool_1000ep_for_training.csv `
  --output_stats_json .\data\pools_of_episodes\for_training\pool_1000ep_for_training.json
```

The included repository already contains:

```text
data/pools_of_episodes/for_training/pool_1000ep_for_training.csv
data/pools_of_episodes/for_training/pool_1000ep_for_training.json
```

You can reuse them for faster reproduction.

## Training the Proposed Meta-Controller

The proposed method is obtained by training the switching classifier.

```powershell
python .\code\proposed_method\train_switching_classifier.py `
  --cons_dir .\models\conservative_policy `
  --agg_dir .\models\aggressive_policy `
  --episode_pool_csv .\data\pools_of_episodes\for_training\pool_1000ep_for_training.csv `
  --budget_min 120 `
  --budget_max 220 `
  --budget_step 5 `
  --meta_interval 1 `
  --max_horizon 220 `
  --switch_interval 5 `
  --scan_interval 5 `
  --n_top_zones 2 `
  --sampling_mode hybrid `
  --samples_per_episode 3 `
  --uniform_frac 0.5 `
  --focus_window 25 `
  --feature_history 0 `
  --hidden_size 32 `
  --n_epochs 500 `
  --batch_size 16 `
  --lr 0.001 `
  --feature_noise 0.01 `
  --early_stop_patience 20 `
  --switch_prob_threshold 0.5 `
  --eval_episodes 400 `
  --base_seed 2001 `
  --dataset_cache_path .\data\datasets\training_set\training_set_cached.npz `
  --results_dir .\models\switching_classifier
```

Expected outputs:

```text
models/switching_classifier/switching_model.pt
models/switching_classifier/config.json
models/switching_classifier/dataset_summary.json
models/switching_classifier/train_history.csv
models/switching_classifier/eval_results.json
```

The current included classifier configuration reports:

```text
hidden_size = 32
n_features = 36
feature_history = 0
threshold = 0.5
budget range = 120..220
max_horizon = 220
```

## Experiment 1: Policy Performance Comparison

This experiment evaluates all policies under the training budget distribution. Each agent sees the same episode seeds and the same sampled budgets.

```powershell
python .\code\experiments\exp1_performance_comparison\data_collection\evaluate_policies.py `
  --agent_dirs .\models\aggressive_policy .\models\conservative_policy .\models\flat_policy `
  --agent_names goal_seeking risk-aware sac_rew_shaped `
  --episodes 2000 `
  --base_seed 2208 `
  --budget_min 120 `
  --budget_max 220 `
  --max_horizon 220 `
  --results_dir .\results\tables\exp1_performance_comparison\all_policies `
  --tag 2000ep `
  --switch_classifier_ckpt .\models\switching_classifier\switching_model.pt `
  --switch_cons_dir .\models\conservative_policy `
  --switch_agg_dir .\models\aggressive_policy `
  --switch_prob_thresholds 0.4 0.5 0.6 `
  --switch_agent_name policy_switching
```

Repeat with additional seeds for fair reporting, for example:

```powershell
--base_seed 2208
--base_seed 2306
--base_seed 3101
```

Then create plots/tables:

```powershell
python .\code\experiments\exp1_performance_comparison\plotting\plot_policies_results.py `
  --csvs `
    .\results\tables\exp1_performance_comparison\all_policies\traindist_timeaware_seed2208_eps2000_Bmin120_Bmax220_H220_6agents_2000ep.csv `
    .\results\tables\exp1_performance_comparison\all_policies\traindist_timeaware_seed2306_eps2000_Bmin120_Bmax220_H220_6agents_2000ep.csv `
    .\results\tables\exp1_performance_comparison\all_policies\traindist_timeaware_seed3101_eps2000_Bmin120_Bmax220_H220_6agents_2000ep.csv `
  --alpha 0.1 `
  --out_csv .\results\tables\exp1_performance_comparison\table_CVaR_reproduction.csv `
  --out_dir .\results\figures\exp1_performance_comparison `
  --out_prefix cvar
```

If a high-budget subset is needed:

```powershell
python .\code\experiments\exp1_performance_comparison\data_collection\filter_high_budget_regime.py `
  .\results\tables\exp1_performance_comparison\all_policies\traindist_timeaware_seed2208_eps2000_Bmin120_Bmax220_H220_6agents_2000ep.csv `
  --q_low 0.75 `
  --q_high 1.0
```

Check the script defaults before running if you want a different input/output path.

## Experiment 2: Time vs Risk Budget Sweep

This experiment fixes the budget to each value in a sweep and evaluates the same seeds at every budget. It is useful for measuring how each policy behaves under increasing temporal pressure.

```powershell
python .\code\experiments\exp2_time_vs_risk_analysis\data_collection\evaluate_policies_budget_sweep.py `
  --agent_dirs .\models\aggressive_policy .\models\conservative_policy .\models\flat_policy `
  --agent_names aggressive_policy risk_aware_policy sac_rew_shaped `
  --episodes 300 `
  --base_seed 2026 `
  --budget_min 120 `
  --budget_max 220 `
  --budget_step 10 `
  --max_horizon 220 `
  --results_dir .\results\tables\exp2_time_vs_risk_analysis `
  --tag 1ep `
  --switch_classifier_ckpt .\models\switching_classifier\switching_model.pt `
  --switch_cons_dir .\models\conservative_policy `
  --switch_agg_dir .\models\aggressive_policy `
  --switch_prob_thresholds 0.4 0.5 0.6 `
  --switch_agent_name policy_switching `
  --oracle_cons_dir .\models\conservative_policy `
  --oracle_agg_dir .\models\aggressive_policy `
  --oracle_agent_name oracle_switch
```

For multi-seed reporting, repeat for seeds such as:

```powershell
--base_seed 1900
--base_seed 1940
--base_seed 1963
--base_seed 2010
--base_seed 2026
```

Then plot the sweep:

```powershell
python .\code\experiments\exp2_time_vs_risk_analysis\plotting\plot_results_over_timebudgets.py `
  --csvs `
    .\results\tables\exp2_time_vs_risk_analysis\fixedbudget_sweep_timeaware_seed1900_eps300_Bmin120_Bmax220_Bstep10_H220_7agents_1ep.csv `
    .\results\tables\exp2_time_vs_risk_analysis\fixedbudget_sweep_timeaware_seed1940_eps300_Bmin120_Bmax220_Bstep10_H220_7agents_1ep.csv `
    .\results\tables\exp2_time_vs_risk_analysis\fixedbudget_sweep_timeaware_seed1963_eps300_Bmin120_Bmax220_Bstep10_H220_7agents_1ep.csv `
    .\results\tables\exp2_time_vs_risk_analysis\fixedbudget_sweep_timeaware_seed2010_eps300_Bmin120_Bmax220_Bstep10_H220_7agents_1ep.csv `
    .\results\tables\exp2_time_vs_risk_analysis\fixedbudget_sweep_timeaware_seed2026_eps300_Bmin120_Bmax220_Bstep10_H220_7agents_1ep.csv `
  --alpha 0.1 `
  --out_dir .\results\figures\exp2_time_vs_risk_analysis
```

## Classifier Evaluation and Diagnostics

### Classifier Quality

```powershell
python .\code\proposed_method\evaluation\evaluate_switch_classifier_quality.py `
  --model_ckpt .\models\switching_classifier\switching_model.pt `
  --dataset_npz .\data\datasets\test_set\test_set_cached.npz `
  --config_json .\models\switching_classifier\config.json `
  --prob_threshold 0.5 `
  --results_dir .\results\tables\classifier_evaluation\metrics
```

### Switch Timing

```powershell
python .\code\proposed_method\evaluation\evaluate_switch_classifier_timing.py `
  --model_ckpt .\models\switching_classifier\switching_model.pt `
  --episode_pool_csv .\data\pools_of_episodes\for_testing\pool_500ep_for_testing.csv `
  --cons_dir .\models\conservative_policy `
  --agg_dir .\models\aggressive_policy `
  --budget_min 120 `
  --budget_max 220 `
  --budget_step 5 `
  --max_horizon 220 `
  --episodes 500 `
  --switch_prob_thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 `
  --base_seed 2603 `
  --results_dir .\results\tables\classifier_evaluation\switch_timing
```

### Switch-Timing Plots

```powershell
python .\code\proposed_method\evaluation\plot_switch_k_validation.py `
  --per_episode_csv .\results\tables\classifier_evaluation\switch_timing\k_compare_per_episode_eval.csv `
  --summary_csv .\results\tables\classifier_evaluation\switch_timing\k_compare_summary_eval.csv `
  --out_dir .\results\figures\classifier_evaluation\switch_timing
```

## Trajectory Analysis

Trajectory scripts collect qualitative examples for visual inspection.

Example command:

```powershell
python .\code\experiments\trajectories_analysis\collect_policies_trajectories.py `
  --agent_dirs .\models\flat_policy .\models\conservative_policy .\models\aggressive_policy `
  --agent_names flat_policy conservative_policy aggressive_policy `
  --episodes 50 `
  --budget_min 120 `
  --budget_max 220 `
  --budget_step 5 `
  --max_horizon 220 `
  --results_dir .\results\tables\trajectories `
  --switch_classifier_ckpt .\models\switching_classifier\switching_model.pt `
  --switch_cons_dir .\models\conservative_policy `
  --switch_agg_dir .\models\aggressive_policy `
  --switch_prob_threshold 0.5 `
  --switch_agent_name policy_switching
```

Then plot:

```powershell
python .\code\experiments\trajectories_analysis\plot_trajectories.py
```

Check the plotting script defaults if you want to select a specific trajectory CSV or output directory.

## Existing Saved Results

The repository already includes:

- trained low-level policies in `models/`;
- a trained switching classifier in `models/switching_classifier/`;
- episode pools in `data/pools_of_episodes/`;
- cached classifier datasets in `data/datasets/`;
- final experiment tables in `results/tables/`;
- final plots in `results/figures/`.

The saved thesis-scale result files use the following naming conventions:

- `traindist_timeaware_seed*_eps2000_Bmin120_Bmax220_H220_6agents_2000ep.csv`
  - Experiment 1, training-distribution budgets.
  - `6agents` means three low-level baselines plus three switch-controller thresholds: `0.4`, `0.5`, and `0.6`.

- `fixedbudget_sweep_timeaware_seed*_eps300_Bmin120_Bmax220_Bstep10_H220_7agents_1ep.csv`
  - Experiment 2, fixed-budget sweep.
  - `7agents` means three low-level baselines, three switch-controller thresholds, and the oracle switching policy.

Agent names used in the saved CSV files:

| Saved name | Thesis role |
| --- | --- |
| `goal_seeking` / `aggressive_policy` | aggressive SAC baseline |
| `risk-aware` / `risk_aware_policy` | conservative WCSAC baseline |
| `sac_rew_shaped` | flat SAC baseline |
| `policy_switching_pthr0.4` | proposed method, threshold `0.4` |
| `policy_switching_pthr0.5` | proposed method, threshold `0.5` |
| `policy_switching_pthr0.6` | proposed method, threshold `0.6` |
| `oracle_switch` | idealized oracle switching reference |

Therefore, there are two reproduction modes.

### Fast Reproduction

Use the existing `models/` and `data/` artifacts, then rerun only evaluation and plotting scripts.

This is best for checking that the reported numbers and figures can be regenerated.

### Full Reproduction

Start from environment setup, retrain all low-level policies, rebuild episode pools, retrain the classifier, and rerun all experiments.

This is more faithful but much more expensive because:

- SAC/WCSAC training is long;
- oracle labelling requires counterfactual MuJoCo rollouts;
- multi-seed evaluation can run thousands of episodes.

## Citation and External Sources

This project builds on:

- WCSAC: <https://github.com/AlgTUDelft/WCSAC>
- Safety Gym: <https://github.com/openai/safety-gym>

Please cite the original projects where appropriate, in addition to citing this thesis work.
