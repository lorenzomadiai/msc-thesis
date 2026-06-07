# Trained Models

This folder stores the trained policies and switching classifier used by the thesis experiments.

## Contents

- `aggressive_policy/`
  - Time-aware SAC policy used as the goal-seeking / risk-tolerant low-level policy.

- `conservative_policy/`
  - WCSAC policy used as the risk-aware low-level policy.

- `flat_policy/`
  - SAC policy trained with a fixed reward-cost scalarization.

- `switching_classifier/`
  - PyTorch classifier used as the proposed high-level meta-controller.


## Expected Files

Low-level TensorFlow policies usually contain:

```text
saved_model.pb
variables/
config.json
progress.txt
```

The switching classifier contains:

```text
switching_model.pt
config.json
train_history.csv
```

## Role in Reproduction

For fast reproduction, use the existing models and rerun only the evaluation and plotting scripts.

For full reproduction, retrain the low-level policies first, then rebuild the episode pool and retrain the switching classifier.


