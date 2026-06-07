# Data Artifacts

This folder contains the datasets used to train and test the switching classifier, moreover it contains also the pools of episodes used to generate the respective datasets.

## Contents

- `pools_of_episodes/`
  - Offline pools of seeded episodes.
  - Used to construct oracle labels for the switching classifier.

- `datasets/`
  - Cached supervised datasets derived from the episode pools.
  - Avoids recomputing expensive oracle rollouts every time the classifier is trained or evaluated.

## Role in Reproduction

The full proposed-method pipeline is expensive because oracle labelling requires counterfactual MuJoCo rollouts. These cached data artifacts make the reproduction faster and more stable.

The thesis training pool contains:

- `1000` episodes;
- `500` conservative-policy successes;
- `500` conservative-policy failures.

The classifier dataset contains approximately `3000` state-level samples.

## Regenerating Data

Use:

```powershell
python .\code\proposed_method\build_episode_pool.py ...
python .\code\proposed_method\train_switching_classifier.py ...
```

The complete commands are listed in the top-level README.


