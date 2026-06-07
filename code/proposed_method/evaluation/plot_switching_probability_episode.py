#!/usr/bin/env python3
"""
plot_switch_probability_episode.py

Plot the switch-classifier probability over time for one episode from an
episode pool. The idea is to show how the classifier's predicted switching probability evolves over the course of an episode 
since time decreases and the agent could not reach the goal in time.

The script:
  1. Loads one episode from a pool CSV.
  2. Reconstructs the episode in MetaEnv using its seed and budget.
  3. Runs the switch classifier at each decision step.
  4. Stores p_switch over time.
  5. Plots:
       - classifier probability curve
       - smoothed probability curve
       - decision threshold
       - oracle k_best from the pool
       - predicted k_pred from the classifier

Expected pool CSV columns:
    seed, budget, best_k

Optional:
    episode, cons_success
"""

import os
import sys
import csv
import argparse
import warnings

import numpy as np
import torch
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
warnings.filterwarnings("ignore", category=FutureWarning)

from safety_gym.envs.engine import Engine

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Adding {_HERE} to sys.path for module imports.")
sys.path.insert(0, _HERE)

from meta_env import MetaEnv
from models import DeltaNet
from common import STATIC_CONFIG, load_policy, extract_features


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    if "net.0.weight" in state_dict:
        return state_dict

    stripped = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            stripped[k[len("module."):]] = v
        else:
            stripped[k] = v
    return stripped


def load_classifier_model(model_ckpt: str):
    if not os.path.isfile(model_ckpt):
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt}")

    ckpt = torch.load(model_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {model_ckpt}")

    state_dict = _normalize_state_dict_keys(state_dict)

    if "net.0.weight" not in state_dict:
        keys_preview = ", ".join(list(state_dict.keys())[:8])
        raise ValueError(
            "Could not infer model architecture from checkpoint. "
            f"Expected key 'net.0.weight'. First keys: {keys_preview}"
        )

    hidden_size = int(state_dict["net.0.weight"].shape[0])

    model = DeltaNet(hidden_size=hidden_size)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"Loaded classifier: {model_ckpt} (hidden_size={hidden_size})")
    return model


def load_pool_rows(csv_path: str):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Episode pool CSV not found: {csv_path}")

    rows = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV header in: {csv_path}")

        required = {"seed", "budget", "best_k"}
        fields = set(reader.fieldnames)

        if not required.issubset(fields):
            raise ValueError(
                f"Invalid CSV headers in {csv_path}. "
                f"Required columns: {sorted(required)}"
            )

        for i, row in enumerate(reader, start=1):
            best_k_raw = str(row.get("best_k", "")).strip()

            if best_k_raw in ("", "nan", "None"):
                best_k = -1
            else:
                best_k = int(float(best_k_raw))

            episode_raw = str(row.get("episode", "")).strip()
            episode = int(episode_raw) if episode_raw not in ("", "nan", "None") else i

            cons_success_raw = str(row.get("cons_success", "")).strip()
            cons_success = (
                int(float(cons_success_raw))
                if cons_success_raw not in ("", "nan", "None")
                else None
            )

            rows.append(
                {
                    "episode": int(episode),
                    "seed": int(row["seed"]),
                    "budget": int(row["budget"]),
                    "best_k": int(best_k),
                    "cons_success": cons_success,
                }
            )

    if not rows:
        raise ValueError(f"Episode pool CSV is empty: {csv_path}")

    return rows


def select_episode(pool_rows, episode_id=None, row_index=None, require_best_k=True):
    rows = pool_rows

    if require_best_k:
        rows = [r for r in rows if int(r["best_k"]) >= 0]

    if not rows:
        raise ValueError("No valid rows available after filtering.")

    if episode_id is not None:
        matches = [r for r in rows if int(r["episode"]) == int(episode_id)]
        if not matches:
            raise ValueError(f"No episode={episode_id} found in the pool.")
        return matches[0]

    if row_index is not None:
        idx = int(row_index)
        if idx < 0 or idx >= len(rows):
            raise ValueError(f"row_index must be in [0, {len(rows) - 1}].")
        return rows[idx]

    return rows[0]


def moving_average(x, window: int):
    x = np.asarray(x, dtype=np.float64)

    if window <= 1 or len(x) == 0:
        return x

    window = min(int(window), len(x))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="same")


def collect_switch_probability_trace(
    model,
    env: MetaEnv,
    seed: int,
    budget: int,
    max_horizon: int,
    threshold: float,
):
    env.seed(int(seed))
    try:
        env._env.seed(int(seed))
    except Exception:
        pass

    obs = env.reset().copy()

    env.B = int(budget)
    if env.irreversible_switch:
        obs[-2] = env._budget_norm()
        obs[-1] = 0.0
    else:
        obs[-1] = env._budget_norm()

    done = False
    ep_len = 0
    switched = False
    k_pred = -1

    steps = []
    probs = []

    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))

    while not done and ep_len < max_horizon:
        feats = extract_features(obs, env)

        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            logit = float(model(x).item())
            p_switch = 1.0 / (1.0 + np.exp(-logit))

        steps.append(int(ep_len))
        probs.append(float(p_switch))

        if not switched and p_switch > threshold:
            switched = True
            k_pred = int(ep_len)

        action = 1 if switched else 0
        obs, _r, done, _info = env.step(action)
        ep_len += 1

    return np.asarray(steps), np.asarray(probs), int(k_pred)


def plot_probability_trace(
    steps,
    probs,
    threshold,
    best_k,
    pred_k,
    budget,
    episode,
    seed,
    out_path,
    smooth_window,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probs_smooth = moving_average(probs, smooth_window)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.plot(
        steps,
        probs,
        linewidth=1.0,
        alpha=0.35,
        label="Raw classifier probability",
    )

    ax.plot(
        steps,
        probs_smooth,
        linewidth=2.0,
        label="Classifier p(switch > continue)",
    )

    ax.axhline(
        float(threshold),
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold={threshold:g}",
    )

    if best_k >= 0:
        ax.axvline(
            int(best_k),
            linestyle="--",
            linewidth=1.2,
            label=f"k_best={best_k}",
        )

    if pred_k >= 0:
        ax.axvline(
            int(pred_k),
            linestyle=":",
            linewidth=1.4,
            label=f"k_pred={pred_k}",
        )

    ax.set_title("Illustrative example of switching decision over time")
    ax.set_xlabel("Decision step k")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right")

    subtitle = f"episode={episode}, seed={seed}, budget={budget}"
    ax.text(
        0.01,
        -0.22,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Plot switch probability over time for one episode from an episode pool."
    )

    p.add_argument("--cons_dir", type=str, required=True)
    p.add_argument("--agg_dir", type=str, required=True)
    p.add_argument("--model_ckpt", type=str, required=True)
    p.add_argument("--episode_pool_csv", type=str, required=True)

    p.add_argument("--episode", type=int, default=None,
                   help="Episode id from the pool CSV. If omitted, row_index is used.")
    p.add_argument("--row_index", type=int, default=0,
                   help="0-based row index after optional best_k filtering.")

    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--require_best_k", action="store_true",
                   help="Only select episodes with best_k >= 0.")

    p.add_argument("--budget_min", type=int, default=120)
    p.add_argument("--budget_max", type=int, default=220)
    p.add_argument("--budget_step", type=int, default=5)
    p.add_argument("--meta_interval", type=int, default=1)
    p.add_argument("--max_horizon", type=int, default=0)

    p.add_argument("--smooth_window", type=int, default=7)
    p.add_argument("--results_dir", type=str,
                   default="results/threshold/switch_probability_examples")
    p.add_argument("--tag", type=str, default="")

    args = p.parse_args()

    max_horizon = args.max_horizon if args.max_horizon > 0 else args.budget_max
    os.makedirs(args.results_dir, exist_ok=True)

    pool_rows = load_pool_rows(args.episode_pool_csv)
    selected = select_episode(
        pool_rows=pool_rows,
        episode_id=args.episode,
        row_index=args.row_index,
        require_best_k=bool(args.require_best_k),
    )

    print("Selected episode:")
    print(selected)

    print("\nLoading low-level policies...")
    sess_cons, act_fn_cons = load_policy(args.cons_dir)
    sess_agg, act_fn_agg = load_policy(args.agg_dir)

    def env_fn():
        return Engine(STATIC_CONFIG)

    env = MetaEnv(
        env_fn=env_fn,
        act_fn_cons=act_fn_cons,
        act_fn_agg=act_fn_agg,
        meta_interval=args.meta_interval,
        budget_min=args.budget_min,
        budget_max=args.budget_max,
        budget_step=args.budget_step,
        irreversible_switch=True,
        seed=int(selected["seed"]),
    )

    model = load_classifier_model(args.model_ckpt)

    steps, probs, pred_k = collect_switch_probability_trace(
        model=model,
        env=env,
        seed=int(selected["seed"]),
        budget=int(selected["budget"]),
        max_horizon=int(max_horizon),
        threshold=float(args.threshold),
    )

    tag = f"_{args.tag}" if str(args.tag).strip() else ""
    out_name = (
        f"switch_probability_episode{selected['episode']}"
        f"_B{selected['budget']}"
        f"_thr{args.threshold:g}"
        f"{tag}.png"
    )
    out_path = os.path.join(args.results_dir, out_name)

    plot_probability_trace(
        steps=steps,
        probs=probs,
        threshold=float(args.threshold),
        best_k=int(selected["best_k"]),
        pred_k=int(pred_k),
        budget=int(selected["budget"]),
        episode=int(selected["episode"]),
        seed=int(selected["seed"]),
        out_path=out_path,
        smooth_window=int(args.smooth_window),
    )

    print("\nSummary:")
    print(f"  episode : {selected['episode']}")
    print(f"  seed    : {selected['seed']}")
    print(f"  budget  : {selected['budget']}")
    print(f"  best_k  : {selected['best_k']}")
    print(f"  pred_k  : {pred_k}")
    print(f"  delta_k : {pred_k - selected['best_k'] if pred_k >= 0 and selected['best_k'] >= 0 else 'NA'}")

    env.close()
    sess_cons.close()
    sess_agg.close()


if __name__ == "__main__":
    main()
