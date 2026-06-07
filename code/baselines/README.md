# Baseline Training Code

This folder contains the reinforcement-learning algorithms used to train the thesis baselines and the low-level policies used for the proposed method.

## Role in the Thesis

The baselines instantiate the static risk attitudes discussed in the thesis:

- `sac_timeaware.py` trains SAC-style time-aware policies.
  - Used for the aggressive / goal-seeking policy.
  - Used for the flat reward-shaped policy.
  - Use `--lambda 0.0` for aggressive SAC and `--lambda 0.02` for the flat baseline.
- `wcsac_timeaware.py` trains the conservative risk-aware policy.
  - Based on the WCSAC implementation in `externals/WCSAC`.
- `utils/wrappers.py` defines `TimeBudgetWrapper`.
  - Adds normalized remaining time and normalized mission budget to observations.
  - Terminates episodes when the sampled mission budget expires.

These policies are later reused by the proposed method, the switching controller does not learn low-level control; it learns when to use the conservative policy and when to switch to the aggressive policy.

## How to Run

Run these scripts from this directory so local imports such as `utils.wrappers` resolve correctly:

```powershell
cd code\baselines
```

The main README contains the full training commands for:

- aggressive SAC policy;
- flat SAC policy;
- conservative WCSAC policy.

## Important Parameters

The thesis setup uses:

- budget range: `120..220`;
- budget step: `5`;
- hidden layers: `(256, 256)`;
- batch size: `256`;
- learning rate: `1e-3`;
- SAC hazard-cost coefficient: `--lambda 0.0` for aggressive, `--lambda 0.02` for flat;
- epochs: `100`;
- steps per epoch: `30000`;
- seed: `0`.

