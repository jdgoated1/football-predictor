"""Evaluation metrics for 3-class football outcome predictions.

H/D/A is an ORDINAL outcome (Home < Draw < Away from one team's perspective).
RPS is the canonical metric for football because it penalises predicting a
distant wrong class more than a near one.
"""
from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier: mean squared error between probs and one-hot truth."""
    onehot = np.eye(probs.shape[1])[y_true]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def ranked_probability_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """RPS for ordered outcomes. Lower is better.
    For 3 classes (H, D, A):
        RPS = 0.5 * [(P(H) - I(H))^2 + (P(H)+P(D) - I(H)+I(D))^2]
    """
    onehot = np.eye(probs.shape[1])[y_true]
    cum_p = np.cumsum(probs, axis=1)
    cum_t = np.cumsum(onehot, axis=1)
    # Sum squared differences across cumulative bins (excluding the last, which is always 1)
    return float(((cum_p[:, :-1] - cum_t[:, :-1]) ** 2).sum(axis=1).mean() / (probs.shape[1] - 1))


def calibration_buckets(y_true: np.ndarray, probs: np.ndarray, klass: int = 0,
                        n_buckets: int = 10) -> list[tuple[float, float, int]]:
    """For predicted-probability vs observed-frequency comparison.
    Returns list of (mean_pred, mean_obs, n) per bucket for the given class."""
    p = probs[:, klass]
    y = (y_true == klass).astype(int)
    edges = np.linspace(0, 1, n_buckets + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.sum() == 0:
            continue
        out.append((float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())))
    return out


def odds_to_probs(odds_h: np.ndarray, odds_d: np.ndarray,
                  odds_a: np.ndarray) -> np.ndarray:
    """Convert decimal bookmaker odds to normalised implied probabilities.
    Strips the overround (bookmaker margin)."""
    inv = np.stack([1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a], axis=1)
    s = inv.sum(axis=1, keepdims=True)
    return inv / s


def report(y_true: np.ndarray, probs: np.ndarray, label: str = "model") -> dict:
    from sklearn.metrics import accuracy_score, log_loss
    pred = probs.argmax(axis=1)
    return {
        "label": label,
        "acc": float(accuracy_score(y_true, pred)),
        "logloss": float(log_loss(y_true, probs, labels=[0, 1, 2])),
        "brier": brier_score(y_true, probs),
        "rps": ranked_probability_score(y_true, probs),
    }


def print_report(rep: dict) -> None:
    print(f"  {rep['label']:14}  acc={rep['acc']:.3f}  logloss={rep['logloss']:.3f}  "
          f"brier={rep['brier']:.3f}  rps={rep['rps']:.4f}")
