"""Dataset-level utilities for supervised switch learning.

Includes pool CSV loading, train/val/test splitting, and classification metrics.
"""

import csv
import os
import numpy as np
import torch
import torch.nn as nn


def split_supervised_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    L: np.ndarray,
    val_frac: float,
    test_frac: float,
    seed: int = 0,
):
    """Split dataset into train/val/test with shuffled indices."""
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")

    val_frac = float(np.clip(val_frac, 0.0, 0.8))
    test_frac = float(np.clip(test_frac, 0.0, 0.8))
    if val_frac + test_frac >= 0.95:
        test_frac = max(0.0, 0.95 - val_frac)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)

    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    if n_test + n_val >= n:
        overflow = n_test + n_val - (n - 1)
        n_test = max(0, n_test - overflow)
    if n_test + n_val >= n:
        n_val = max(0, n_val - 1)

    i_test = idx[:n_test]
    i_val = idx[n_test:n_test + n_val]
    i_train = idx[n_test + n_val:]

    return {
        "train": (X[i_train], Y[i_train], L[i_train]),
        "val": (X[i_val], Y[i_val], L[i_val]),
        "test": (X[i_test], Y[i_test], L[i_test]),
    }


def classification_metrics(model, X: np.ndarray, L: np.ndarray, prob_threshold: float = 0.5):
    """Compute BCE/accuracy metrics on a split."""
    if len(X) == 0:
        return {"bce": np.nan, "acc": np.nan, "pos_rate_pred": np.nan}

    X_t = torch.tensor(X, dtype=torch.float32)
    L_t = torch.tensor(L, dtype=torch.float32)
    p_thr = float(np.clip(prob_threshold, 1e-6, 1.0 - 1e-6))
    logit_thr = float(np.log(p_thr / (1.0 - p_thr)))

    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        bce = float(nn.functional.binary_cross_entropy_with_logits(logits, L_t).item())
        pred_lbl = (logits > logit_thr).float()
        acc = float(torch.mean((pred_lbl == L_t).float()).item())
        pos_rate_pred = float(torch.mean(pred_lbl).item())
    return {"bce": bce, "acc": acc, "pos_rate_pred": pos_rate_pred}


def load_episode_pool_csv(csv_path: str):
    """Load pre-screened episode pool CSV.

    Required columns: seed,budget
    Optional columns: cons_success,best_k
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Episode pool CSV not found: {csv_path}")

    seeds = []
    budgets = []
    outcomes = []
    best_ks = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"seed", "budget"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Invalid episode pool CSV headers in {csv_path}. "
                "Required columns: seed,budget (optional: cons_success)."
            )
        has_outcome = "cons_success" in set(reader.fieldnames)
        has_best_k = "best_k" in set(reader.fieldnames)

        for row in reader:
            seeds.append(int(row["seed"]))
            budgets.append(int(row["budget"]))
            if has_outcome:
                outcomes.append(int(row["cons_success"]))
            if has_best_k:
                bk_raw = str(row.get("best_k", "")).strip()
                best_ks.append(int(bk_raw) if bk_raw not in ("", "nan", "None") else -1)

    if len(seeds) == 0:
        raise ValueError(f"Episode pool CSV is empty: {csv_path}")

    seeds_arr = np.array(seeds, dtype=np.int64)
    budgets_arr = np.array(budgets, dtype=np.int64)
    outcomes_arr = np.array(outcomes, dtype=np.int64) if len(outcomes) == len(seeds) else None
    best_k_arr = np.array(best_ks, dtype=np.int64) if len(best_ks) == len(seeds) else None
    return seeds_arr, budgets_arr, outcomes_arr, best_k_arr
