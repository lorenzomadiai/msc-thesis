#!/usr/bin/env python3
"""
train_gap_switch.py
-------------------
Supervised binary classification for switch timing using oracle gap sign.
"""

import os
import sys
import csv
import json
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from safety_gym.envs.engine import Engine

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from meta_env import MetaEnv
from models import DeltaNet
from common import STATIC_CONFIG, load_policy, N_FEATURES
from utils import (
    split_supervised_dataset,
    classification_metrics,
    load_episode_pool_csv,
    collect_dataset,
    evaluate_gap_policy,
    train_classifier,
)


def main():
    p = argparse.ArgumentParser(
        description="Train supervised switch policy by classifying oracle gap sign (delta > 0)."
    )

    # Environment
    p.add_argument("--cons_dir", type=str, required=True)
    p.add_argument("--agg_dir", type=str, required=True)
    p.add_argument("--budget_min", type=int, default=120)
    p.add_argument("--budget_max", type=int, default=220)
    p.add_argument("--budget_step", type=int, default=5)
    p.add_argument("--meta_interval", type=int, default=1)
    p.add_argument("--max_horizon", type=int, default=0,
                   help="Max env steps per episode (0 = budget_max).")

    # Dataset
    p.add_argument("--episode_pool_csv", type=str, required=True,
                   help="CSV generated offline with episode pool: columns seed,budget[,cons_success].")
    p.add_argument("--min_abs_gap", type=float, default=0.0,
                   help="Filter training samples with |delta| below threshold.")
    p.add_argument("--switch_interval", type=int, default=5,
                   help="Oracle interval for conservative value computation.")
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="Validation fraction of supervised dataset.")
    p.add_argument("--test_frac", type=float, default=0.1,
                   help="Test fraction of supervised dataset.")
    p.add_argument("--sampling_mode", type=str, default="hybrid",
                   choices=["all", "uniform", "hybrid"],
                   help="How to sample k per episode for dataset creation.")
    p.add_argument("--samples_per_episode", type=int, default=3,
                   help="Number of sampled timesteps per episode when sampling_mode != all.")
    p.add_argument("--uniform_frac", type=float, default=0.5,
                   help="Fraction of per-episode samples drawn uniformly (hybrid mode).")
    p.add_argument("--uniform_frac_failed", type=float, default=0.0,
                   help="Fraction of non-forced samples drawn uniformly on conservative-fail episodes (hybrid mode).")
    p.add_argument("--focus_window", type=int, default=25,
                   help="Half-window around k* used for focused sampling (hybrid mode).")
    p.add_argument("--force_pos_prev_k", type=int, default=0,
                   help="On conservative-fail episodes, also force k*, k*-1..k*-N as positive when switch succeeds.")
    p.add_argument("--scan_interval", type=int, default=5,
                   help="Zone width for k* search used by hybrid focused sampling.")
    p.add_argument("--n_top_zones", type=int, default=2,
                   help="Top zones densely scanned for k* search in hybrid mode.")
    p.add_argument("--print_datapoints", action="store_true",
                   help="Print every datapoint kept in the dataset for each episode.")
    p.add_argument("--dataset_episodes", type=int, default=0,
                   help="If > 0, collect dataset from a random subset of this many episodes sampled from episode_pool_csv.")
    p.add_argument("--feature_history", type=int, default=0,
                   help="Number of previous feature vectors to concatenate (0=current only).")
    p.add_argument("--dataset_cache_path", type=str, default="",
                   help="Optional path to an NPZ cache for dataset tensors (defaults to results_dir/dataset_cached.npz).")
    p.add_argument("--force_dataset_recollect", action="store_true",
                   help="Ignore any dataset cache and recollect episodes.")
    p.add_argument("--skip_dataset_cache_save", action="store_true",
                   help="Do not write the dataset cache after collection.")
    p.add_argument("--disable_dataset_cache", action="store_true",
                   help="Disable dataset cache loading and saving entirely.")

    # Training
    p.add_argument("--hidden_size", type=int, default=16)
    p.add_argument("--n_epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--feature_noise", type=float, default=0.01)
    p.add_argument("--early_stop_patience", type=int, default=20,
                   help="Stop training if val_bce does not improve for this many epochs (0 disables).")

    # Decision rule
    p.add_argument("--switch_prob_threshold", type=float, default=0.5,
                   help="Switch when predicted P(switch_better) > threshold.")

    # Reward shaping
    p.add_argument("--cost_weight", type=float, default=0.02)
    p.add_argument("--deadline_weight", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)

    # Evaluation
    p.add_argument("--eval_episodes", type=int, default=400)

    # Output
    p.add_argument("--base_seed", type=int, default=2001)
    p.add_argument("--results_dir", type=str,
                   default="results/threshold/gap_001")

    args = p.parse_args()

    max_horizon = args.max_horizon if args.max_horizon > 0 else args.budget_max
    os.makedirs(args.results_dir, exist_ok=True)

    print("\nLoading low-level policies ...")
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
        seed=args.base_seed + 99,
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    feature_history = max(0, int(args.feature_history))
    feature_dim = int(N_FEATURES * (feature_history + 1))
    print(f"\n[debug] Meta observation dim: {obs_dim}")
    print(f"[debug] Base feature dim: {N_FEATURES}")
    print(f"[debug] Feature history: {feature_history} (stack length = {feature_history + 1})")
    print(f"[debug] Target feature dim (stacked): {feature_dim}")

    cache_enabled = not bool(args.disable_dataset_cache)
    dataset_cache_path = ""
    dataset_cache_status = "disabled"
    if cache_enabled:
        cache_path_raw = args.dataset_cache_path.strip()
        if not cache_path_raw:
            cache_path_raw = os.path.join(args.results_dir, "dataset_cached.npz")
        dataset_cache_path = os.path.abspath(cache_path_raw)
        dataset_cache_status = "pending"

    dataset_spec = {
        "episode_pool_csv": os.path.abspath(args.episode_pool_csv),
        "dataset_episodes": int(args.dataset_episodes),
        "base_seed": int(args.base_seed),
        "feature_history": int(feature_history),
        "feature_dim": int(feature_dim),
        "min_abs_gap": float(args.min_abs_gap),
        "samples_per_episode": int(args.samples_per_episode),
        "sampling_mode": args.sampling_mode,
        "uniform_frac": float(args.uniform_frac),
        "uniform_frac_failed": float(args.uniform_frac_failed),
        "focus_window": int(args.focus_window),
        "force_pos_prev_k": int(args.force_pos_prev_k),
        "scan_interval": int(args.scan_interval),
        "n_top_zones": int(args.n_top_zones),
        "switch_interval": int(args.switch_interval),
        "cost_weight": float(args.cost_weight),
        "deadline_weight": float(args.deadline_weight),
        "gamma": float(args.gamma),
        "max_horizon": int(max_horizon),
        "budget_min": int(args.budget_min),
        "budget_max": int(args.budget_max),
        "budget_step": int(args.budget_step),
        "meta_interval": int(args.meta_interval),
    }
    dataset_spec_json = json.dumps(dataset_spec, sort_keys=True)

    bvals = list(range(args.budget_min, args.budget_max + 1, args.budget_step))

    all_seeds_pool, all_budgets_pool, pool_outcomes_pool, pool_best_k_pool = load_episode_pool_csv(args.episode_pool_csv)
    pool_size_total = int(len(all_seeds_pool))
    if pool_outcomes_pool is not None:
        pool_win_total = int(pool_outcomes_pool.sum())
        pool_fail_total = int(pool_size_total - pool_win_total)
        print(f"Loaded episode pool: size={pool_size_total} wins={pool_win_total} fails={pool_fail_total}")
    else:
        print(f"Loaded episode pool: size={pool_size_total} (cons_success column missing)")

    dataset_episodes = int(max(0, args.dataset_episodes))
    n_collect = pool_size_total if dataset_episodes == 0 else min(dataset_episodes, pool_size_total)
    rng_pool = np.random.RandomState(args.base_seed + 11)
    sel_idx = rng_pool.permutation(pool_size_total)[:n_collect]

    all_seeds = all_seeds_pool[sel_idx]
    all_budgets = all_budgets_pool[sel_idx]
    pool_outcomes = pool_outcomes_pool[sel_idx] if pool_outcomes_pool is not None else None
    pool_best_k = pool_best_k_pool[sel_idx] if pool_best_k_pool is not None else None

    pool_size = int(len(all_seeds))
    if pool_outcomes is not None:
        pool_win = int(pool_outcomes.sum())
        pool_fail = int(pool_size - pool_win)
        print(f"Using dataset subset: size={pool_size} wins={pool_win} fails={pool_fail}")
    else:
        pool_win = -1
        pool_fail = -1
        print(f"Using dataset subset: size={pool_size} (cons_success column missing)")

    eval_seeds = np.random.RandomState(args.base_seed + 2).randint(
        0, 2**31 - 1, size=args.eval_episodes, dtype=np.int64
    )
    eval_budgets = np.random.RandomState(args.base_seed + 3).choice(
        bvals, size=args.eval_episodes, replace=True
    )

    X = Y = L = None
    if (
        cache_enabled
        and dataset_cache_path
        and (not args.force_dataset_recollect)
        and os.path.isfile(dataset_cache_path)
    ):
        try:
            with np.load(dataset_cache_path, allow_pickle=True) as data:
                if "spec" in data.files:
                    spec_entry = data["spec"]
                    if getattr(spec_entry, "shape", ()) == ():
                        spec_json_cached = spec_entry.item()
                    else:
                        spec_json_cached = spec_entry
                    cached_spec = json.loads(str(spec_json_cached))
                    if cached_spec == dataset_spec:
                        X = data["X"]
                        Y = data["Y"]
                        L = data["L"]
                        dataset_cache_status = "loaded"
                        print(
                            f"\n[cache] Loaded cached dataset from {dataset_cache_path}"
                            f" (shape={X.shape})."
                        )
                        print(
                            f"  [cache] Dataset size from cache: {len(X)} samples, feature_dim={X.shape[1]}"
                        )
                    else:
                        dataset_cache_status = "stale"
                        print(
                            f"\n[cache] Spec mismatch for {dataset_cache_path};"
                            " regenerating dataset."
                        )
                else:
                    dataset_cache_status = "invalid"
                    print(
                        f"\n[cache] Missing spec entry in {dataset_cache_path};"
                        " regenerating dataset."
                    )
        except Exception as exc:
            dataset_cache_status = "error"
            print(f"\n[cache] Failed to load dataset cache {dataset_cache_path}: {exc}")

    if X is None:
        print("\nCollecting supervised dataset (oracle gap targets) ...")
        print("  [debug] Using common.features.extract_features for per-step inputs")
        print(f"  [debug] Expected stacked feature dimension: {feature_dim}")
        data_rng = np.random.RandomState(args.base_seed + 17)
        sampling_log_path = os.path.join(args.results_dir, "sampling_episode_log.csv")
        X, Y, L = collect_dataset(
            env=env,
            seeds=all_seeds,
            budgets=all_budgets,
            cons_outcomes=pool_outcomes,
            best_k_hints=pool_best_k,
            max_horizon=max_horizon,
            cost_weight=args.cost_weight,
            deadline_weight=args.deadline_weight,
            gamma=args.gamma,
            switch_interval=args.switch_interval,
            rng=data_rng,
            sampling_mode=args.sampling_mode,
            samples_per_episode=args.samples_per_episode,
            uniform_frac=args.uniform_frac,
            uniform_frac_failed=args.uniform_frac_failed,
            focus_window=args.focus_window,
            force_pos_prev_k=args.force_pos_prev_k,
            scan_interval=args.scan_interval,
            n_top_zones=args.n_top_zones,
            min_abs_gap=args.min_abs_gap,
            sampling_log_path=sampling_log_path,
            print_sampling=True,
            print_datapoints=args.print_datapoints,
            feature_history=feature_history,
        )
        dataset_cache_status = "collected" if cache_enabled else "disabled"

    if len(X) == 0:
        raise RuntimeError(
            "Dataset is empty. Lower --min_abs_gap or provide a larger episode pool CSV."
        )

    if X.ndim != 2:
        raise ValueError(
            f"Collected dataset has unexpected rank {X.ndim}; expected 2D (n_samples, n_features)."
        )
    if X.shape[1] != feature_dim:
        raise ValueError(
            f"Feature dimension mismatch: collected {X.shape[1]} but model expects {feature_dim}."
        )
    print(f"  [debug] Dataset tensor shape: {X.shape}")

    if (
        cache_enabled
        and dataset_cache_path
        and dataset_cache_status == "collected"
        and (not args.skip_dataset_cache_save)
    ):
        try:
            os.makedirs(os.path.dirname(dataset_cache_path), exist_ok=True)
            np.savez_compressed(
                dataset_cache_path,
                X=X,
                Y=Y,
                L=L,
                spec=np.array(dataset_spec_json, dtype=object),
            )
            dataset_cache_status = "saved"
            print(f"  [cache] Saved dataset tensors to {dataset_cache_path}")
        except Exception as exc:
            print(f"  [cache] Failed to save dataset cache {dataset_cache_path}: {exc}")

    splits = split_supervised_dataset(
        X, Y, L,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.base_seed + 29,
    )
    X_train, Y_train, L_train = splits["train"]
    X_val, Y_val, L_val = splits["val"]
    X_test, Y_test, L_test = splits["test"]

    print("\nDataset split:")
    print(f"  train: {len(X_train)}")
    print(f"  val:   {len(X_val)}")
    print(f"  test:  {len(X_test)}")

    n_pos = int((L_train == 1).sum())
    n_neg = int((L_train == 0).sum())
    print(f"  train delta>0 (switch): {n_pos} ({100.0*n_pos/max(1, len(X_train)):.1f}%)")
    print(f"  train delta<=0 (wait):  {n_neg} ({100.0*n_neg/max(1, len(X_train)):.1f}%)")

    base_layout = "hazards_lidar(16)+goal_lidar(16)+v_xy(2)+time_left_norm+budget_norm"
    feature_layout = f"{feature_history + 1}x[{base_layout}]"
    print(f"  feature_dim={feature_dim} ({feature_layout})")

    dataset_summary = {
        "n_samples": int(len(X)),
        "pool_size": int(pool_size),
        "pool_win": int(pool_win),
        "pool_fail": int(pool_fail),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "samples_per_episode_effective": float(len(X) / max(1, pool_size)),
        "n_switch_label_train": n_pos,
        "n_wait_label_train": n_neg,
        "n_features": int(feature_dim),
        "feature_history": int(feature_history),
        "feature_layout": feature_layout,
        "dataset_cache_path": dataset_cache_path,
        "dataset_cache_status": dataset_cache_status,
        "delta_mean": float(np.mean(Y)),
        "delta_std": float(np.std(Y)),
        "delta_min": float(np.min(Y)),
        "delta_max": float(np.max(Y)),
    }
    with open(os.path.join(args.results_dir, "dataset_summary.json"), "w") as f:
        json.dump(dataset_summary, f, indent=2)

    torch.manual_seed(args.base_seed)
    model = DeltaNet(input_dim=feature_dim, hidden_size=args.hidden_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"  Trainable parameters: {n_params}")

    print("\nTraining switch classifier ...")
    history = train_classifier(
        model=model,
        X_train=X_train,
        L_train=L_train,
        X_val=X_val,
        L_val=L_val,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        feature_noise=args.feature_noise,
        seed=args.base_seed + 7,
        prob_threshold=args.switch_prob_threshold,
        early_stop_patience=args.early_stop_patience,
    )

    hist_path = os.path.join(args.results_dir, "train_history.csv")
    with open(hist_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_bce", "train_acc", "val_bce", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)

    test_m = classification_metrics(model, X_test, L_test, prob_threshold=args.switch_prob_threshold)
    print("\n=== Supervised Test (held-out) ===")
    print(f"  test_bce={test_m['bce']:.5f}  test_acc={test_m['acc']:.3f}")

    print("\nEvaluating policies ...")
    eval_gap = evaluate_gap_policy(
        model,
        env,
        eval_seeds,
        eval_budgets,
        max_horizon,
        switch_prob_threshold=args.switch_prob_threshold,
        feature_history=feature_history,
    )

    print("\n=== Evaluation Summary ===")
    print(
        f"  Gap policy:    succ={eval_gap['success_rate']:.3f}  cost={eval_gap['mean_cost']:.3f}  "
        f"cvar10={eval_gap['cvar_10']:.3f}  sw%={eval_gap['frac_switched']:.2f}  "
        f"sw_step={eval_gap['mean_switch_step']:.1f}"
    )

    eval_out = {
        "supervised_test": {
            "bce": test_m["bce"],
            "acc": test_m["acc"],
            "n_test": int(len(X_test)),
        },
        "gap_policy": eval_gap,
        # "conservative": eval_cons,
        # "aggressive": eval_agg,

    }

    with open(os.path.join(args.results_dir, "eval_results.json"), "w") as f:
        json.dump(eval_out, f, indent=2)

    config = {
        "method": "gap_classification",
        "hidden_size": args.hidden_size,
        "n_features": int(feature_dim),
        "n_params": n_params,
        "pool_size": int(pool_size),
        "pool_size_total": int(pool_size_total),
        "episode_pool_csv": args.episode_pool_csv,
        "dataset_episodes": args.dataset_episodes,
        "eval_episodes": args.eval_episodes,
        "cost_weight": args.cost_weight,
        "deadline_weight": args.deadline_weight,
        "gamma": args.gamma,
        "switch_interval": args.switch_interval,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "sampling_mode": args.sampling_mode,
        "samples_per_episode": args.samples_per_episode,
        "uniform_frac": args.uniform_frac,
        "uniform_frac_failed": args.uniform_frac_failed,
        "focus_window": args.focus_window,
        "force_pos_prev_k": args.force_pos_prev_k,
        "scan_interval": args.scan_interval,
        "n_top_zones": args.n_top_zones,
        "min_abs_gap": args.min_abs_gap,
        "print_datapoints": bool(args.print_datapoints),
        "switch_prob_threshold": args.switch_prob_threshold,
        "feature_noise": args.feature_noise,
        "early_stop_patience": int(args.early_stop_patience),
        "feature_history": int(feature_history),
        "dataset_cache_path": dataset_cache_path,
        "dataset_cache_status": dataset_cache_status,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "budget_min": args.budget_min,
        "budget_max": args.budget_max,
        "budget_step": args.budget_step,
        "meta_interval": args.meta_interval,
        "max_horizon": max_horizon,
        "base_seed": args.base_seed,
        "feature_layout": feature_layout,
    }
    with open(os.path.join(args.results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    torch.save(model.state_dict(), os.path.join(args.results_dir, "gap_model.pt"))

    print("\nSaved:")
    print(f"  {os.path.join(args.results_dir, 'gap_model.pt')}")
    print(f"  {os.path.join(args.results_dir, 'config.json')}")
    print(f"  {os.path.join(args.results_dir, 'dataset_summary.json')}")
    print(f"  {os.path.join(args.results_dir, 'sampling_episode_log.csv')}")
    print(f"  {os.path.join(args.results_dir, 'train_history.csv')}")
    print(f"  {os.path.join(args.results_dir, 'eval_results.json')}")

    env.close()
    sess_cons.close()
    sess_agg.close()


if __name__ == "__main__":
    main()
