# Proposed Method Code

This folder contains the proposed hierarchical policy-switching method from the thesis.

## Role in the Thesis

The method treats risk adaptation as an optimal-stopping problem. At each decision step, a high-level controller decides whether to:

- continue with the conservative policy; or
- switch irreversibly to the aggressive policy.

The only learned component of the proposed method is the switching classifier.

## Main Files

- `meta_env.py`
  - Defines the meta-level environment.
  - Action `0` means continue with the conservative policy.
  - Action `1` means switch to the aggressive policy.

- `build_episode_pool.py`
  - Builds a reproducible pool of seeded episodes.
  - The thesis pool contains `1000` episodes balanced as `500` conservative wins and `500` conservative failures.

- `train_switching_classifier.py`
  - Builds the supervised dataset from the episode pool.
  - Computes oracle switching labels.
  - Trains the PyTorch MLP classifier.

- `common/`
  - Shared configuration, feature extraction, policy loading, MuJoCo state handling, and oracle utilities.

- `models/`
  - Classifier architecture (`DeltaNet`).

- `utils/`
  - Dataset sampling, training, evaluation, and metric helpers.

- `evaluation/`
  - Classifier-quality, switch-timing, oracle, and diagnostic plotting scripts.

## Classifier Setup

The thesis classifier uses:

- input dimension: `36`;
- feature layout: hazard lidar `16`, goal lidar `16`, `v_x`, `v_y`, `time_left_norm`, `budget_norm`;
- hidden size: `32`;
- loss: BCE with logits;
- optimizer: Adam;
- learning rate: `1e-3`;
- batch size: `16`;
- maximum epochs: `500`;
- early stopping patience: `20`;
- threshold: `0.5`.

The trained classifier is stored in `models/switching_classifier/`.


