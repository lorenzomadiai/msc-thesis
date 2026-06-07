"""Training loop utilities for the supervised DeltaNet classifier."""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from .data_utils import classification_metrics


def train_classifier(
    model,
    X_train: np.ndarray,
    L_train: np.ndarray,
    X_val: np.ndarray,
    L_val: np.ndarray,
    n_epochs: int,
    batch_size: int,
    lr: float,
    feature_noise: float = 0.0,
    seed: int = 0,
    prob_threshold: float = 0.5,
    early_stop_patience: int = 0,
):
    """Train switch classifier with BCE-with-logits loss."""
    rng = np.random.RandomState(seed)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    L_t = torch.tensor(L_train, dtype=torch.float32)

    history = []
    n = len(X_train)
    has_val = len(X_val) > 0
    best_val = np.inf
    best_state = None
    patience = max(0, int(early_stop_patience))
    epochs_since_best = 0
    stopped_early = False

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = rng.permutation(n)

        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            xb = X_t[idx]
            lb = L_t[idx]

            if feature_noise > 0.0:
                xb = xb + torch.randn_like(xb) * feature_noise

            logits = model(xb)
            loss = criterion(logits, lb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        train_m = classification_metrics(model, X_train, L_train, prob_threshold=prob_threshold)
        val_m = classification_metrics(model, X_val, L_val, prob_threshold=prob_threshold) if has_val else {
            "bce": np.nan,
            "acc": np.nan,
            "pos_rate_pred": np.nan,
        }

        improved = has_val and np.isfinite(val_m["bce"]) and val_m["bce"] < best_val
        if improved:
            best_val = val_m["bce"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        elif has_val and patience > 0:
            epochs_since_best += 1
        else:
            epochs_since_best = 0

        row = {
            "epoch": epoch,
            "train_bce": train_m["bce"],
            "train_acc": train_m["acc"],
            "val_bce": val_m["bce"],
            "val_acc": val_m["acc"],
        }
        history.append(row)

        if epoch <= 5 or epoch % 10 == 0 or epoch == n_epochs:
            if has_val:
                print(
                    f"  [epoch {epoch:4d}/{n_epochs}] "
                    f"train_bce={row['train_bce']:.5f} "
                    f"val_bce={row['val_bce']:.5f} "
                    f"train_acc={row['train_acc']:.3f} "
                    f"val_acc={row['val_acc']:.3f}"
                )
            else:
                print(
                    f"  [epoch {epoch:4d}/{n_epochs}] "
                    f"bce={row['train_bce']:.5f} "
                    f"acc={row['train_acc']:.3f}"
                )

        if has_val and patience > 0 and epochs_since_best >= patience:
            print(
                f"  [early-stop] no val_bce improvement for {patience} epochs "
                f"(best={best_val:.5f}); stopping at epoch {epoch}."
            )
            stopped_early = True
            break

    if has_val and best_state is not None:
        model.load_state_dict(best_state)

    return history
