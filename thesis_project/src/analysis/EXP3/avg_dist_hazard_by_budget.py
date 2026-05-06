import pandas as pd
import sys
import glob
import os

# Accept file path(s) as argument, or default to current result folder
paths = sys.argv[1:] if len(sys.argv) > 1 else glob.glob(
    os.path.join(os.path.dirname(__file__), "results/**/*.csv"), recursive=True
)

dfs = []
for p in paths:
    try:
        dfs.append(pd.read_csv(p, usecols=["budget", "success", "mean_dist_hazard"]))
    except Exception as e:
        print(f"Skipping {p}: {e}")

if not dfs:
    print("No CSV files loaded.")
    sys.exit(1)

df = pd.concat(dfs, ignore_index=True)

# keep only episodes where the agent reached the goal within the time budget
df_success = df[df["success"] == 1]

result = (
    df_success.groupby("budget")["mean_dist_hazard"]
    .agg(mean="mean", std="std", count="count")
    .reset_index()
)

print(result.to_string(index=False))
