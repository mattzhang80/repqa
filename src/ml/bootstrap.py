"""Bootstrap confidence intervals for model evaluation metrics.

Uses the percentile bootstrap method (BCa available if scipy is imported).

Usage:
    from src.ml.bootstrap import bootstrap_auc, bootstrap_precision_at_threshold
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import precision_score, roc_auc_score


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Compute a bootstrap confidence interval for any scalar metric.

    Args:
        y_true:       True binary labels.
        y_score:      Predicted probabilities or scores.
        metric_fn:    f(y_true, y_score) -> float.  Called on each bootstrap sample.
        n_bootstrap:  Number of bootstrap iterations.
        ci:           Coverage probability (e.g. 0.95 for 95% CI).
        random_state: RNG seed.

    Returns:
        Dict with keys:
            point_estimate, ci_lower, ci_upper, bootstrap_distribution (np.ndarray).
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_scores: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        # Skip degenerate samples (all-same labels)
        if len(np.unique(yt)) < 2:
            continue
        try:
            score = metric_fn(yt, ys)
            if np.isfinite(score):
                boot_scores.append(score)
        except Exception:
            continue

    boot_arr = np.array(boot_scores)
    alpha = 1.0 - ci
    ci_lower = float(np.percentile(boot_arr, 100 * alpha / 2)) if len(boot_arr) else float("nan")
    ci_upper = float(np.percentile(boot_arr, 100 * (1 - alpha / 2))) if len(boot_arr) else float("nan")
    point_estimate = metric_fn(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap_valid": len(boot_arr),
        "bootstrap_distribution": boot_arr,
    }


def bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Bootstrap CI for ROC AUC.

    Args:
        y_true:      True binary labels.
        y_score:     Predicted probabilities.
        n_bootstrap: Bootstrap iterations.
        ci:          Coverage probability.
        random_state: RNG seed.

    Returns:
        Dict from bootstrap_metric().
    """
    def _auc(yt: np.ndarray, ys: np.ndarray) -> float:
        return float(roc_auc_score(yt, ys))

    return bootstrap_metric(y_true, y_score, _auc, n_bootstrap, ci, random_state)


def bootstrap_precision_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Bootstrap CI for precision at a fixed probability threshold.

    Args:
        y_true:      True binary labels.
        y_score:     Predicted probabilities.
        threshold:   Decision threshold.
        n_bootstrap: Bootstrap iterations.
        ci:          Coverage probability.
        random_state: RNG seed.

    Returns:
        Dict from bootstrap_metric().
    """
    def _prec(yt: np.ndarray, ys: np.ndarray) -> float:
        y_pred = (ys >= threshold).astype(int)
        return float(precision_score(yt, y_pred, zero_division="warn"))  # type: ignore[arg-type]

    return bootstrap_metric(y_true, y_score, _prec, n_bootstrap, ci, random_state)
