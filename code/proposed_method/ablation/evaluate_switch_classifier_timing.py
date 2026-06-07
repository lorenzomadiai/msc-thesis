#!/usr/bin/env python3
"""
evaluate_switch_classifier_timing.py
------------------------------
Sample episodes from a pre-built pool and compare predicted switch timing
(k_pred) against oracle best_k saved in the pool, for multiple switch
probability thresholds.

Expected pool CSV columns:
    seed, budget, best_k
Optional columns:
    episode, cons_success

Notes:
- best_k is treated as 0-based decision index (k=0 means switch immediately).
- k_pred is computed with the same convention.
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

_HERE = os.path.dirname("/workspace/thesis_project/src/training/supervised_learning/")
print(f"Adding {_HERE} to sys.path for module imports.")
sys.path.insert(0, _HERE)

from meta_env import MetaEnv
from models import DeltaNet
from common import STATIC_CONFIG, load_policy, extract_features


def make_switch_timing_plots(rows_all: list, out_dir: str, tag: str):
    """Create delta_k diagnostics plots grouped by threshold.

    Generates:
      - histogram of delta_k per threshold
      - stacked bar (early/on-time/late) per threshold
      - boxplot of delta_k per threshold
            - trade-off curve: no-switch on safe episodes vs switch on fail episodes
    """
    if len(rows_all) == 0:
        print("[warning] No rows to plot.")
        return []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warning] Matplotlib not available, skipping plots: {e}")
        return []

    per_thr = {}
    for r in rows_all:
        thr = float(r["threshold"])
        d_raw = r.get("delta_k", "")
        if str(d_raw).strip() == "":
            continue
        dval = float(d_raw)
        per_thr.setdefault(thr, []).append(dval)

    if len(per_thr) == 0:
        print("[warning] No valid delta_k values found, skipping plots.")
        return []

    thresholds = sorted(per_thr.keys())
    tag_sfx = f"_{tag}" if str(tag).strip() else ""
    saved = []

    # 1) Histogram: one subplot per threshold
    n_thr = len(thresholds)
    fig_h, axes = plt.subplots(n_thr, 1, figsize=(8, max(3, 2.4 * n_thr)), sharex=True)
    if n_thr == 1:
        axes = [axes]

    max_abs = max(max(abs(v) for v in vals) for vals in per_thr.values())
    max_abs = max(1, int(np.ceil(max_abs)))
    bins = np.arange(-max_abs - 0.5, max_abs + 1.5, 1.0)

    for ax, thr in zip(axes, thresholds):
        vals = np.asarray(per_thr[thr], dtype=np.float64)
        ax.hist(vals, bins=bins, alpha=0.8)
        ax.axvline(0.0, linestyle="--", linewidth=1.2, color="black")
        ax.set_ylabel("Count")
        ax.set_title(f"threshold={thr:g} (n={len(vals)})")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    axes[-1].set_xlabel("delta_k = pred_k - best_k")
    fig_h.tight_layout()
    hist_path = os.path.join(out_dir, f"delta_k_hist_by_threshold{tag_sfx}.png")
    fig_h.savefig(hist_path, dpi=220)
    plt.close(fig_h)
    saved.append(hist_path)

    # 2) Stacked bar: early/on-time/late rates per threshold
    early = []
    ontime = []
    late = []
    labels = []
    for thr in thresholds:
        vals = np.asarray(per_thr[thr], dtype=np.float64)
        n = max(1, len(vals))
        e = float(np.sum(vals < 0)) / n
        o = float(np.sum(vals == 0)) / n
        l = float(np.sum(vals > 0)) / n
        early.append(e)
        ontime.append(o)
        late.append(l)
        labels.append(f"{thr:g}")

    x = np.arange(len(thresholds))
    fig_s, ax_s = plt.subplots(figsize=(8, 4.5))
    ax_s.bar(x, early, label="Early", color="#3b82f6")
    ax_s.bar(x, ontime, bottom=early, label="On-time", color="#10b981")
    ax_s.bar(x, late, bottom=np.array(early) + np.array(ontime), label="Late", color="#ef4444")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(labels)
    ax_s.set_ylim(0.0, 1.0)
    ax_s.set_xlabel("Threshold")
    ax_s.set_ylabel("Proportion")
    ax_s.set_title("Switch timing profile by threshold")
    ax_s.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_s.legend(loc="upper right")
    fig_s.tight_layout()
    stack_path = os.path.join(out_dir, f"delta_k_early_ontime_late_stacked{tag_sfx}.png")
    fig_s.savefig(stack_path, dpi=220)
    plt.close(fig_s)
    saved.append(stack_path)

    # 3) Boxplot: delta_k distribution per threshold
    fig_b, ax_b = plt.subplots(figsize=(8, 4.5))
    data = [np.asarray(per_thr[thr], dtype=np.float64) for thr in thresholds]
    ax_b.boxplot(data, labels=[f"{thr:g}" for thr in thresholds], showfliers=True)
    ax_b.axhline(0.0, linestyle="--", linewidth=1.2, color="black")
    ax_b.set_xlabel("Threshold")
    ax_b.set_ylabel(r"$\Delta k = k_{\mathrm{pred}} - k_{\mathrm{best}}$")
    plt.title(r"Effect of threshold on switch timing ($\Delta k$)")
    ax_b.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig_b.tight_layout()
    box_path = os.path.join(out_dir, f"delta_k_boxplot_by_threshold{tag_sfx}.png")
    fig_b.savefig(box_path, dpi=220)
    plt.close(fig_b)
    saved.append(box_path)

    # 4) Safety-aware trade-off:
    #    NoSwitch@Safe = P(pred_k<0 | cons_success=1)
    #    SwitchOnFail = P(pred_k>=0 | cons_success=0)
    per_thr_safe_fail = {}
    for r in rows_all:
        thr = float(r["threshold"])

        cs_raw = r.get("cons_success", "")
        if str(cs_raw).strip() in ("", "nan", "None"):
            continue

        try:
            cs = int(float(cs_raw))
        except Exception:
            continue

        pred_k = int(r.get("pred_k", -1))
        switched = pred_k >= 0

        if thr not in per_thr_safe_fail:
            per_thr_safe_fail[thr] = {
                "safe_total": 0,
                "safe_noswitch": 0,
                "fail_total": 0,
                "fail_switch": 0,
            }

        if cs == 1:
            per_thr_safe_fail[thr]["safe_total"] += 1
            if not switched:
                per_thr_safe_fail[thr]["safe_noswitch"] += 1
        elif cs == 0:
            per_thr_safe_fail[thr]["fail_total"] += 1
            if switched:
                per_thr_safe_fail[thr]["fail_switch"] += 1

    if len(per_thr_safe_fail) > 0:
        thr_sf = sorted(per_thr_safe_fail.keys())
        no_switch_safe = []
        switch_on_fail = []
        safe_counts = []
        fail_counts = []

        for thr in thr_sf:
            d = per_thr_safe_fail[thr]
            st = int(d["safe_total"])
            ft = int(d["fail_total"])
            safe_counts.append(st)
            fail_counts.append(ft)

            no_switch_safe.append(
                (float(d["safe_noswitch"]) / st) if st > 0 else np.nan
            )
            switch_on_fail.append(
                (float(d["fail_switch"]) / ft) if ft > 0 else np.nan
            )

        fig_t, ax_t = plt.subplots(figsize=(8.5, 4.8))
        x = np.asarray(thr_sf, dtype=np.float64)
        ax_t.plot(x, no_switch_safe, marker="o", linewidth=2, label="NoSwitch@Safe")
        ax_t.plot(x, switch_on_fail, marker="o", linewidth=2, label="SwitchOnFail")
        ax_t.set_ylim(0.0, 1.0)
        ax_t.set_xlabel("Threshold")
        ax_t.set_ylabel("Rate")
        ax_t.set_title("Safety-aware switching trade-off by threshold")
        ax_t.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax_t.legend(loc="lower left")

        # Show effective sample sizes used for each curve point.
        for i, thr in enumerate(thr_sf):
            y1 = no_switch_safe[i]
            y2 = switch_on_fail[i]
            if np.isfinite(y1):
                ax_t.annotate(f"n={safe_counts[i]}", (thr, y1), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
            if np.isfinite(y2):
                ax_t.annotate(f"n={fail_counts[i]}", (thr, y2), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)

        fig_t.tight_layout()
        tradeoff_path = os.path.join(out_dir, f"safe_vs_fail_tradeoff_by_threshold{tag_sfx}.png")
        fig_t.savefig(tradeoff_path, dpi=220)
        plt.close(fig_t)
        saved.append(tradeoff_path)

    return saved


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

    out = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty/invalid CSV header in: {csv_path}")
        fields = set(reader.fieldnames)
        required = {"seed", "budget", "best_k"}
        if not required.issubset(fields):
            raise ValueError(
                f"Invalid CSV headers in {csv_path}. Required columns: {sorted(required)}"
            )

        for i, row in enumerate(reader, start=1):
            bk_raw = str(row.get("best_k", "")).strip()
            if bk_raw in ("", "nan", "None"):
                best_k = -1
            else:
                best_k = int(float(bk_raw))

            ep_raw = str(row.get("episode", "")).strip()
            episode = int(ep_raw) if ep_raw not in ("", "nan", "None") else i

            out.append(
                {
                    "episode": int(episode),
                    "seed": int(row["seed"]),
                    "budget": int(row["budget"]),
                    "best_k": int(best_k),
                    "cons_success": int(row["cons_success"])
                    if str(row.get("cons_success", "")).strip() not in ("", "nan", "None")
                    else None,
                }
            )

    if len(out) == 0:
        raise ValueError(f"Episode pool CSV is empty: {csv_path}")
    return out


def sample_rows(rows: list, n_samples: int, base_seed: int, require_best_k: bool):
    if require_best_k:
        rows = [r for r in rows if int(r["best_k"]) >= 0]
        if len(rows) == 0:
            raise ValueError(
                "No rows with valid best_k>=0 found. "
                "Run build_episode_pool with best_k columns or disable --require_best_k."
            )

    n = int(max(1, n_samples))
    n = min(n, len(rows))
    rng = np.random.RandomState(base_seed)
    idx = rng.permutation(len(rows))[:n]
    return [rows[int(i)] for i in idx]


def rollout_predicted_switch_k(
    model,
    env: MetaEnv,
    seed: int,
    budget: int,
    max_horizon: int,
    switch_prob_threshold: float,
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

    p_thr = float(np.clip(switch_prob_threshold, 1e-6, 1.0 - 1e-6))

    while not done and ep_len < max_horizon:
        if not switched:
            feats = extract_features(obs, env)
            with torch.no_grad():
                x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
                logit = float(model(x).item())
                p_switch = 1.0 / (1.0 + np.exp(-logit))
            if p_switch > p_thr:
                switched = True
                # 0-based decision index to match pool best_k.
                k_pred = int(ep_len)

        action = 1 if switched else 0
        obs, _r, done, _info = env.step(action)
        ep_len += 1

    return int(k_pred)


def summarize_threshold(rows_eval: list, max_horizon: int):
    n = len(rows_eval)
    if n == 0:
        return {
            "n": 0,
            "n_with_best_k": 0,
            "switch_rate": np.nan,
            "exact_match_rate": np.nan,
            "mean_signed_error_switched_only": np.nan,
            "mean_abs_error_switched_only": np.nan,
            "mean_abs_error_penalized": np.nan,
            "median_abs_error_penalized": np.nan,
        }

    has_best = np.array([int(r["best_k"]) >= 0 for r in rows_eval], dtype=bool)
    pred = np.array([int(r["pred_k"]) for r in rows_eval], dtype=np.int64)
    best = np.array([int(r["best_k"]) for r in rows_eval], dtype=np.int64)
    switched = pred >= 0

    valid = has_best & switched
    penalized_pred = np.where(pred >= 0, pred, max_horizon)

    exact = (pred == best) & has_best
    signed_err = pred - best
    abs_err = np.abs(signed_err)
    abs_err_pen = np.abs(penalized_pred - best)

    return {
        "n": int(n),
        "n_with_best_k": int(np.sum(has_best)),
        "switch_rate": float(np.mean(switched)),
        "exact_match_rate": float(np.mean(exact[has_best])) if np.any(has_best) else np.nan,
        "mean_signed_error_switched_only": float(np.mean(signed_err[valid])) if np.any(valid) else np.nan,
        "mean_abs_error_switched_only": float(np.mean(abs_err[valid])) if np.any(valid) else np.nan,
        "mean_abs_error_penalized": float(np.mean(abs_err_pen[has_best])) if np.any(has_best) else np.nan,
        "median_abs_error_penalized": float(np.median(abs_err_pen[has_best])) if np.any(has_best) else np.nan,
    }


def main():
    p = argparse.ArgumentParser(
        description=(
            "Sample episodes from pool and compare classifier-predicted switch k "
            "against oracle best_k for multiple probability thresholds."
        )
    )

    # Env/policies
    p.add_argument("--cons_dir", type=str, required=True)
    p.add_argument("--agg_dir", type=str, required=True)
    p.add_argument("--budget_min", type=int, default=120)
    p.add_argument("--budget_max", type=int, default=220)
    p.add_argument("--budget_step", type=int, default=5)
    p.add_argument("--meta_interval", type=int, default=1)
    p.add_argument("--max_horizon", type=int, default=0,
                   help="Max env steps per episode (0 = budget_max).")

    # Model/pool
    p.add_argument("--model_ckpt", type=str, required=True,
                   help="Path to trained classifier checkpoint (e.g. gap_model.pt).")
    p.add_argument("--episode_pool_csv", type=str, required=True,
                   help="Episode pool CSV with seed,budget,best_k columns.")

    # Sampling/evaluation
    p.add_argument("--episodes", type=int, default=200,
                   help="Number of sampled episodes from pool.")
    p.add_argument("--switch_prob_thresholds", type=float, nargs="+", required=True,
                   help="List of thresholds to evaluate.")
    p.add_argument("--require_best_k", action="store_true",
                   help="If set, sample only rows with best_k>=0 (recommended).")
    p.add_argument("--base_seed", type=int, default=2603)

    # Output
    p.add_argument("--results_dir", type=str,
                   default="results/threshold/pool_k_compare")
    p.add_argument("--tag", type=str, default="")

    args = p.parse_args()

    thresholds = [float(t) for t in args.switch_prob_thresholds]
    for t in thresholds:
        if not (0.0 <= t <= 1.0):
            p.error(f"Each threshold must be in [0,1]. Invalid: {t}")

    max_horizon = args.max_horizon if args.max_horizon > 0 else args.budget_max
    os.makedirs(args.results_dir, exist_ok=True)

    rows_pool = load_pool_rows(args.episode_pool_csv)
    rows_sampled = sample_rows(
        rows=rows_pool,
        n_samples=args.episodes,
        base_seed=args.base_seed + 1,
        require_best_k=bool(args.require_best_k),
    )

    print(
        f"Loaded pool rows={len(rows_pool)} | sampled={len(rows_sampled)} "
        f"| require_best_k={bool(args.require_best_k)}"
    )

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

    model = load_classifier_model(args.model_ckpt)

    all_rows = []
    summary = []

    for thr in thresholds:
        rows_eval_thr = []
        print(f"\nEvaluating threshold={thr:.4f} ...")
        for r in rows_sampled:
            pred_k = rollout_predicted_switch_k(
                model=model,
                env=env,
                seed=int(r["seed"]),
                budget=int(r["budget"]),
                max_horizon=max_horizon,
                switch_prob_threshold=float(thr),
            )

            best_k = int(r["best_k"])
            abs_err = abs(pred_k - best_k) if (pred_k >= 0 and best_k >= 0) else np.nan
            abs_err_pen = abs((pred_k if pred_k >= 0 else max_horizon) - best_k) if best_k >= 0 else np.nan

            row_out = {
                "threshold": float(thr),
                "episode": int(r["episode"]),
                "seed": int(r["seed"]),
                "budget": int(r["budget"]),
                "cons_success": "" if r["cons_success"] is None else int(r["cons_success"]),
                "best_k": int(best_k),
                "pred_k": int(pred_k),
                "switched": int(pred_k >= 0),
                "delta_k": int(pred_k - best_k) if (pred_k >= 0 and best_k >= 0) else "",
                "abs_error": float(abs_err) if np.isfinite(abs_err) else "",
                "abs_error_penalized": float(abs_err_pen) if np.isfinite(abs_err_pen) else "",
            }
            rows_eval_thr.append(row_out)
            all_rows.append(row_out)

        s = summarize_threshold(rows_eval_thr, max_horizon=max_horizon)
        s["threshold"] = float(thr)
        summary.append(s)

        print(
            "  "
            f"switch_rate={s['switch_rate']:.3f}  "
            f"exact={s['exact_match_rate']:.3f}  "
            f"mae(sw_only)={s['mean_abs_error_switched_only']:.3f}  "
            f"mae(penalized)={s['mean_abs_error_penalized']:.3f}"
        )

    tag = f"_{args.tag}" if str(args.tag).strip() else ""
    per_ep_csv = os.path.join(args.results_dir, f"k_compare_per_episode{tag}.csv")
    summary_csv = os.path.join(args.results_dir, f"k_compare_summary{tag}.csv")
    summary_json = os.path.join(args.results_dir, f"k_compare_summary{tag}.json")

    with open(per_ep_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "episode",
                "seed",
                "budget",
                "cons_success",
                "best_k",
                "pred_k",
                "switched",
                "delta_k",
                "abs_error",
                "abs_error_penalized",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "n",
                "n_with_best_k",
                "switch_rate",
                "exact_match_rate",
                "mean_signed_error_switched_only",
                "mean_abs_error_switched_only",
                "mean_abs_error_penalized",
                "median_abs_error_penalized",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    payload = {
        "config": {
            "cons_dir": args.cons_dir,
            "agg_dir": args.agg_dir,
            "model_ckpt": args.model_ckpt,
            "episode_pool_csv": args.episode_pool_csv,
            "episodes": int(args.episodes),
            "sampled_episodes": int(len(rows_sampled)),
            "require_best_k": bool(args.require_best_k),
            "switch_prob_thresholds": thresholds,
            "budget_min": args.budget_min,
            "budget_max": args.budget_max,
            "budget_step": args.budget_step,
            "meta_interval": args.meta_interval,
            "max_horizon": int(max_horizon),
            "base_seed": args.base_seed,
            "tag": args.tag,
        },
        "summary": summary,
    }
    with open(summary_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved:")
    print(f"  {per_ep_csv}")
    print(f"  {summary_csv}")
    print(f"  {summary_json}")

    plot_paths = make_switch_timing_plots(
        rows_all=all_rows,
        out_dir=args.results_dir,
        tag=args.tag,
    )
    if len(plot_paths) > 0:
        print("  plots:")
        for pth in plot_paths:
            print(f"    {pth}")

    env.close()
    sess_cons.close()
    sess_agg.close()


if __name__ == "__main__":
    main()
