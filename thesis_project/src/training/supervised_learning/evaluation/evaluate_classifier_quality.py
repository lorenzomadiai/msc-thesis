#!/usr/bin/env python3
"""
evaluate_classifier_quality.py
------------------------------
Evaluate classification/probability quality of a trained switch classifier.

This script is intentionally separate from rollout-based timing evaluation:
- rollout metrics answer: "did we switch at the right timestep?"
- this script answers: "are predicted probabilities/class labels correct?"

Expected dataset NPZ entries:
- X: features, shape (N, D)
- L: binary labels {0,1}, shape (N,)

Optional config JSON can be passed to reuse split settings from training.
"""

import os
import csv
import json
import argparse
import sys
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Adding {_HERE} to sys.path for module imports.")
sys.path.insert(0, _HERE)
from models import DeltaNet


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "net.0.weight" in state_dict:
        return state_dict
    stripped = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            stripped[k[len("module."):]] = v
        else:
            stripped[k] = v
    return stripped


def load_classifier_model(model_ckpt: str) -> DeltaNet:
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
    input_dim = int(state_dict["net.0.weight"].shape[1])

    model = DeltaNet(input_dim=input_dim, hidden_size=hidden_size)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(
        "Loaded classifier: "
        f"{model_ckpt} (input_dim={input_dim}, hidden_size={hidden_size})"
    )
    return model


def load_dataset_npz(dataset_npz: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(dataset_npz):
        raise FileNotFoundError(f"Dataset NPZ not found: {dataset_npz}")

    with np.load(dataset_npz, allow_pickle=True) as data:
        if "X" not in data.files or "L" not in data.files:
            raise ValueError(
                f"Dataset NPZ must contain X and L arrays. Found keys: {list(data.files)}"
            )
        X = np.asarray(data["X"], dtype=np.float32)
        L = np.asarray(data["L"], dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,D). Got shape: {X.shape}")
    if L.ndim != 1:
        raise ValueError(f"L must be 1D (N,). Got shape: {L.shape}")
    if len(X) != len(L):
        raise ValueError(f"X and L must have same length. Got {len(X)} vs {len(L)}")
    if len(X) == 0:
        raise ValueError("Dataset is empty.")

    return X, L


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return np.nan

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]

    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = float(np.sum(ranks[y_true == 1]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return np.nan

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / n_pos

    y_prev = np.concatenate(([0], y_sorted[:-1]))
    is_pos_hit = (y_sorted == 1) & (y_prev == 0)
    if not np.any(is_pos_hit):
        return np.nan
    return float(np.mean(precision[is_pos_hit]))


def calibration_table(y_true: np.ndarray, p: np.ndarray, n_bins: int):
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n_bins = int(max(2, n_bins))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_rows = []
    ece = 0.0
    n = len(p)

    for b in range(n_bins):
        lo = float(edges[b])
        hi = float(edges[b + 1])

        if b == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(p[mask]))
            frac_pos = float(np.mean(y_true[mask]))
            gap = abs(avg_conf - frac_pos)
            ece += (count / max(1, n)) * gap
        else:
            avg_conf = np.nan
            frac_pos = np.nan
            gap = np.nan

        bin_rows.append(
            {
                "bin_idx": b,
                "bin_lo": lo,
                "bin_hi": hi,
                "count": count,
                "avg_confidence": avg_conf,
                "frac_positive": frac_pos,
                "abs_gap": gap,
            }
        )

    return float(ece), bin_rows


def _compute_metrics_from_probs(
    y_true: np.ndarray,
    probs: np.ndarray,
    bce: float,
    prob_threshold: float,
    n_bins: int,
) -> Dict[str, float]:
    truth = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    pred = (probs > prob_threshold).astype(np.int64)
    pos_mask = truth == 1
    neg_mask = truth == 0

    tp = int(np.sum((pred == 1) & (truth == 1)))
    tn = int(np.sum((pred == 0) & (truth == 0)))
    fp = int(np.sum((pred == 1) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    npv = tn / max(1, tn + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    accuracy = float((tp + tn) / max(1, len(truth)))
    balanced_accuracy = 0.5 * (recall + specificity)

    brier = float(np.mean((probs - truth.astype(np.float64)) ** 2))
    brier_cond_pos = float(np.mean((probs[pos_mask] - 1.0) ** 2)) if np.any(pos_mask) else np.nan
    brier_cond_neg = float(np.mean((probs[neg_mask] - 0.0) ** 2)) if np.any(neg_mask) else np.nan
    auc = _roc_auc_binary(truth, probs)
    ap = _average_precision_binary(truth, probs)
    ece, _ = calibration_table(truth, probs, n_bins=n_bins)

    return {
        "n": int(len(truth)),
        "positive_rate_true": float(np.mean(truth)),
        "positive_rate_pred": float(np.mean(pred)),
        "positive_rate_gap_abs": float(abs(np.mean(pred) - np.mean(truth))),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "bce": float(bce),
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "npv": float(npv),
        "f1": float(f1),
        "brier_score": brier,
        "brier_score_cond_y1": brier_cond_pos,
        "brier_score_cond_y0": brier_cond_neg,
        "roc_auc": float(auc) if np.isfinite(auc) else np.nan,
        "pr_auc": float(ap) if np.isfinite(ap) else np.nan,
        "ece": float(ece),
    }


def evaluate_split(
    model: DeltaNet,
    X: np.ndarray,
    L: np.ndarray,
    prob_threshold: float,
    n_bins: int,
) -> Dict[str, float]:
    if len(X) == 0:
        return {
            "n": 0,
            "positive_rate_true": np.nan,
            "positive_rate_pred": np.nan,
            "positive_rate_gap_abs": np.nan,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "bce": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "specificity": np.nan,
            "npv": np.nan,
            "f1": np.nan,
            "brier_score": np.nan,
            "brier_score_cond_y1": np.nan,
            "brier_score_cond_y0": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "ece": np.nan,
        }

    y = np.asarray(L, dtype=np.float32)
    x_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    p_thr = float(np.clip(prob_threshold, 1e-6, 1.0 - 1e-6))

    model.eval()
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        bce = float(nn.functional.binary_cross_entropy_with_logits(logits, y_t).item())
    return _compute_metrics_from_probs(
        y_true=y,
        probs=probs,
        bce=bce,
        prob_threshold=p_thr,
        n_bins=n_bins,
    )


def threshold_sweep(
    model: DeltaNet,
    X: np.ndarray,
    L: np.ndarray,
    n_bins: int,
    sweep_points: int,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    if len(X) == 0:
        return [], {}

    y = np.asarray(L, dtype=np.float32)
    x_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        bce = float(nn.functional.binary_cross_entropy_with_logits(logits, y_t).item())

    rows = []
    grid = np.linspace(0.01, 0.99, int(max(3, sweep_points)))
    for thr in grid:
        m = _compute_metrics_from_probs(
            y_true=y,
            probs=probs,
            bce=bce,
            prob_threshold=float(thr),
            n_bins=n_bins,
        )
        m["threshold"] = float(thr)
        rows.append(m)

    best_f1 = max(rows, key=lambda r: (r["f1"], -abs(r["precision"] - r["recall"])))
    best_bal_acc = max(rows, key=lambda r: r["balanced_accuracy"])

    best = {
        "best_f1_threshold": float(best_f1["threshold"]),
        "best_f1": float(best_f1["f1"]),
        "best_balanced_accuracy_threshold": float(best_bal_acc["threshold"]),
        "best_balanced_accuracy": float(best_bal_acc["balanced_accuracy"]),
    }
    return rows, best


def confusion_matrices_fixed_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
) -> List[Dict[str, float]]:
    truth = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)

    rows = []
    for thr in thresholds:
        t = float(thr)
        pred = (probs > t).astype(np.int64)
        tp = int(np.sum((pred == 1) & (truth == 1)))
        tn = int(np.sum((pred == 0) & (truth == 0)))
        fp = int(np.sum((pred == 1) & (truth == 0)))
        fn = int(np.sum((pred == 0) & (truth == 1)))
        rows.append(
            {
                "threshold": t,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )
    return rows


def _load_training_config(config_json: str) -> Dict:
    if not config_json:
        return {}
    if not os.path.isfile(config_json):
        raise FileNotFoundError(f"Config JSON not found: {config_json}")
    with open(config_json, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate classifier probability quality (BCE, accuracy, Brier, ROC-AUC, ECE) "
            "from dataset tensors, independent from rollout timing metrics."
        )
    )

    parser.add_argument("--model_ckpt", type=str, required=True,
                        help="Path to trained classifier checkpoint (e.g. gap_model.pt).")
    parser.add_argument("--dataset_npz", type=str, required=True,
                        help="Path to dataset cache NPZ containing X and L arrays.")

    parser.add_argument("--config_json", type=str, default="",
                        help="Optional training config.json to reuse switch_prob_threshold.")

    parser.add_argument("--prob_threshold", type=float, default=-1.0,
                        help="Classification threshold on predicted probability. If <0, use config or 0.5.")
    parser.add_argument("--n_bins", type=int, default=10,
                        help="Number of bins used for ECE/reliability table.")
    parser.add_argument("--sweep_points", type=int, default=41,
                        help="Number of threshold points in [0.01,0.99] for F1/BalAcc sweep.")

    parser.add_argument("--results_dir", type=str,
                        default="results/threshold/classifier_quality")
    parser.add_argument("--tag", type=str, default="")

    args = parser.parse_args()

    cfg = _load_training_config(args.config_json)

    prob_threshold = float(
        args.prob_threshold if args.prob_threshold >= 0.0 else cfg.get("switch_prob_threshold", 0.5)
    )
    prob_threshold = float(np.clip(prob_threshold, 1e-6, 1.0 - 1e-6))

    os.makedirs(args.results_dir, exist_ok=True)

    model = load_classifier_model(args.model_ckpt)
    X, L = load_dataset_npz(args.dataset_npz)
    feature_dim_data = int(X.shape[1])

    with torch.no_grad():
        first_weight = model.net[0].weight
        feature_dim_model = int(first_weight.shape[1])

    if feature_dim_data != feature_dim_model:
        raise ValueError(
            f"Feature dimension mismatch: dataset has {feature_dim_data}, model expects {feature_dim_model}."
        )

    metrics_test = evaluate_split(model, X, L, prob_threshold, n_bins=args.n_bins)
    X_primary, L_primary = X, L
    with torch.no_grad():
        p_primary = torch.sigmoid(model(torch.tensor(X_primary, dtype=torch.float32))).cpu().numpy()
    ece_primary, bins_primary = calibration_table(L_primary, p_primary, n_bins=args.n_bins)

    tag = f"_{args.tag}" if str(args.tag).strip() else ""
    metrics_json = os.path.join(args.results_dir, f"classifier_quality_summary{tag}.json")
    metrics_csv = os.path.join(args.results_dir, f"classifier_quality_summary{tag}.csv")
    reliability_csv = os.path.join(args.results_dir, f"reliability_test{tag}.csv")
    sweep_csv = os.path.join(args.results_dir, f"threshold_sweep_test{tag}.csv")
    confmat_csv = os.path.join(args.results_dir, f"confusion_matrices_threshold_0.1_to_0.9{tag}.csv")

    sweep_rows, sweep_best = threshold_sweep(
        model=model,
        X=X_primary,
        L=L_primary,
        n_bins=args.n_bins,
        sweep_points=args.sweep_points,
    )

    fixed_thresholds = np.arange(0.1, 1.0, 0.1)
    confmat_rows = confusion_matrices_fixed_thresholds(
        y_true=L_primary,
        probs=p_primary,
        thresholds=fixed_thresholds,
    )

    threshold_free_summary = {
        "roc_auc": float(metrics_test["roc_auc"]),
        "pr_auc": float(metrics_test["pr_auc"]),
        "bce": float(metrics_test["bce"]),
        "brier_score": float(metrics_test["brier_score"]),
    }

    payload = {
        "config": {
            "model_ckpt": args.model_ckpt,
            "dataset_npz": args.dataset_npz,
            "config_json": args.config_json,
            "prob_threshold": prob_threshold,
            "n_bins": int(args.n_bins),
            "eval_split": "test",
            "tag": args.tag,
        },
        "splits": {
            "test": metrics_test,
        },
        "threshold_free_summary": threshold_free_summary,
        "primary_split_ece": float(ece_primary),
        "primary_threshold_sweep": sweep_best,
        "confusion_matrices_threshold_0.1_to_0.9": confmat_rows,
    }

    with open(metrics_json, "w") as f:
        json.dump(payload, f, indent=2)

    with open(metrics_csv, "w", newline="") as f:
        fieldnames = [
            "split",
            "n",
            "positive_rate_true",
            "positive_rate_pred",
            "positive_rate_gap_abs",
            "tp",
            "tn",
            "fp",
            "fn",
            "bce",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "npv",
            "f1",
            "brier_score",
            "brier_score_cond_y1",
            "brier_score_cond_y0",
            "roc_auc",
            "pr_auc",
            "ece",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        out = {"split": "test"}
        out.update(metrics_test)
        writer.writerow(out)

    with open(reliability_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin_idx",
                "bin_lo",
                "bin_hi",
                "count",
                "avg_confidence",
                "frac_positive",
                "abs_gap",
            ],
        )
        writer.writeheader()
        writer.writerows(bins_primary)

    with open(sweep_csv, "w", newline="") as f:
        fieldnames = [
            "threshold",
            "n",
            "positive_rate_true",
            "positive_rate_pred",
            "positive_rate_gap_abs",
            "tp",
            "tn",
            "fp",
            "fn",
            "bce",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "npv",
            "f1",
            "brier_score",
            "brier_score_cond_y1",
            "brier_score_cond_y0",
            "roc_auc",
            "pr_auc",
            "ece",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_rows)

    with open(confmat_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "tp", "tn", "fp", "fn"])
        writer.writeheader()
        writer.writerows(confmat_rows)

    primary_metrics = payload["splits"]["test"]
    print("\n=== Threshold-Free Quality Summary ===")
    print(
        f"ROC-AUC={threshold_free_summary['roc_auc']:.3f}  "
        f"PR-AUC={threshold_free_summary['pr_auc']:.3f}  "
        f"BCE={threshold_free_summary['bce']:.5f}  "
        f"Brier={threshold_free_summary['brier_score']:.5f}"
    )
    print("\n=== Additional Diagnostics ===")
    print(
        f"split=test n={primary_metrics['n']} "
        f"acc={primary_metrics['accuracy']:.3f} "
        f"bal_acc={primary_metrics['balanced_accuracy']:.3f} "
        f"prec={primary_metrics['precision']:.3f} "
        f"rec={primary_metrics['recall']:.3f} "
        f"spec={primary_metrics['specificity']:.3f} "
        f"brier_y1={primary_metrics['brier_score_cond_y1']:.5f} "
        f"brier_y0={primary_metrics['brier_score_cond_y0']:.5f} "
        f"f1={primary_metrics['f1']:.3f} "
        f"pos_true={primary_metrics['positive_rate_true']:.3f} "
        f"pos_pred={primary_metrics['positive_rate_pred']:.3f} "
        f"ece={primary_metrics['ece']:.4f}"
    )
    if sweep_best:
        print(
            f"best_f1_thr={sweep_best['best_f1_threshold']:.3f} "
            f"best_f1={sweep_best['best_f1']:.3f} "
            f"best_bal_acc_thr={sweep_best['best_balanced_accuracy_threshold']:.3f} "
            f"best_bal_acc={sweep_best['best_balanced_accuracy']:.3f}"
        )
    print(
        f"confusion: tp={primary_metrics['tp']} tn={primary_metrics['tn']} "
        f"fp={primary_metrics['fp']} fn={primary_metrics['fn']}"
    )
    print("confusion matrices (threshold 0.1 to 0.9):")
    for row in confmat_rows:
        print(
            f"  thr={row['threshold']:.1f} "
            f"tp={row['tp']} tn={row['tn']} fp={row['fp']} fn={row['fn']}"
        )
    print("\nSaved:")
    print(f"  {metrics_json}")
    print(f"  {metrics_csv}")
    print(f"  {reliability_csv}")
    print(f"  {sweep_csv}")
    print(f"  {confmat_csv}")


if __name__ == "__main__":
    main()
