## WCSAC

Source:  
https://github.com/AlgTUDelft/WCSAC

This repository was used as the starting point for implementing the time-aware agents developed in this project. The algorithms in `code/baselines` are based on the original WCSAC implementations and were modified to incorporate a time-budget wrapper, enabling policies to condition their behaviour on the remaining time budget.

## Safety Gym

Source:  
https://github.com/openai/safety-gym

Safety Gym is used as the benchmark environment for this project.

Modifications:
- Removed the explicit MuJoCo dependency requirement to avoid version conflicts with the local MuJoCo installation.
- Added a Windows compatibility patch in `safety_gym/envs/world.py`, replacing `load_model_from_xml` with `load_model_from_path` by first writing the generated XML to a temporary file.