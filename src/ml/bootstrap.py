"""Bootstrap confidence intervals for model evaluation metrics.

Two bootstrap designs:

  * Rep-level (default, ``groups=None``): resample rows with replacement.
    Standard textbook bootstrap; appropriate when observations are i.i.d.

  * Cluster / group bootstrap (``groups`` given): resample *groups* with
    replacement and emit every row from each sampled group.  For RepQA,
    reps within a session share a user, a camera setup, a form intent,
    and a day-of-recording — they are NOT independent.  Rep-level
    bootstrap on clustered data inflates the effective sample size and
    produces over-confident (too narrow) intervals.  Cluster bootstrap
    respects the dependency structure; its CIs are wider and honest.

Two interval methods:

  * Percentile: symmetric α/2 quantiles of the bootstrap distribution.
    Simple, robust, but biased when the distribution is skewed or the
    statistic has finite-sample bias.

  * BCa (bias-corrected accelerated, default): adjusts the quantiles for
    (1) median bias of the bootstrap distribution and (2) variance
    non-constancy via a jackknife-derived acceleration term.  More
    accurate coverage under modest-n and skewed statistics.  Falls back
    to percentile automatically when BCa is ill-defined (e.g. acceleration
    divides by zero, or the jackknife yields all-NaN metrics).

Both AUC and precision/recall at a threshold have their own convenience
wrappers — see ``bootstrap_auc``, ``bootstrap_precision_at_threshold``,
``bootstrap_recall_at_threshold``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats
from sklearn.metrics import precision_score, recall_score, roc_auc_score

MetricFn = Callable[[np.ndarray, np.ndarray], float]


# ── Core bootstrap ────────────────────────────────────────────────────────────

def _safe_metric(
    metric_fn: MetricFn, yt: np.ndarray, ys: np.ndarray
) -> float | None:
    """Evaluate a metric; return None if the sample is degenerate.

    A sample is degenerate when the metric is undefined on it — most
    commonly "only one class in y_true" (AUC) or "no positive predictions
    at this threshold" (precision).  Silently returning NaN would pollute
    the bootstrap distribution with synthetic extreme values; we filter
    them out and report how many samples survived via ``n_valid``.
    """
    if len(yt) == 0:
        return None
    try:
        score = metric_fn(yt, ys)
    except Exception:
        return None
    if score is None or not np.isfinite(score):
        return None
    return float(score)


def _bca_interval(
    point: float,
    boot: np.ndarray,
    jackknife: np.ndarray,
    ci: float,
) -> tuple[float, float] | None:
    """BCa lower/upper limits.  Returns None if BCa is ill-defined."""
    if len(boot) < 2:
        return None
    jk = jackknife[np.isfinite(jackknife)]
    if len(jk) < 2:
        return None

    # z0: bias-correction in standard-normal space
    prop_less = float(np.mean(boot < point))
    # ppf(0) or ppf(1) → ±inf — both make BCa pathological; fall back
    if prop_less <= 0.0 or prop_less >= 1.0:
        return None
    z0 = float(stats.norm.ppf(prop_less))

    # a: acceleration from jackknife (negative of skewness divided by 6)
    jk_mean = float(np.mean(jk))
    u = jk_mean - jk
    num = float(np.sum(u ** 3))
    den = 6.0 * (float(np.sum(u ** 2)) ** 1.5)
    if den == 0.0:
        return None
    a = num / den

    alpha = 1.0 - ci
    z_lo = float(stats.norm.ppf(alpha / 2.0))
    z_hi = float(stats.norm.ppf(1.0 - alpha / 2.0))

    def _adjusted(zq: float) -> float:
        denom = 1.0 - a * (z0 + zq)
        if denom == 0.0:
            return float("nan")
        return float(stats.norm.cdf(z0 + (z0 + zq) / denom))

    alpha_lo = _adjusted(z_lo)
    alpha_hi = _adjusted(z_hi)
    if not (np.isfinite(alpha_lo) and np.isfinite(alpha_hi)):
        return None
    lo = float(np.quantile(boot, alpha_lo))
    hi = float(np.quantile(boot, alpha_hi))
    return lo, hi


def _jackknife(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: MetricFn,
    groups: np.ndarray | None,
) -> np.ndarray:
    """Leave-one-out (or leave-one-group-out) jackknife metrics.

    Units left out match the bootstrap design: for cluster bootstrap,
    we leave out one GROUP per iteration, not one observation.  Any
    left-out slice that yields a degenerate metric contributes NaN; BCa
    drops these before computing acceleration.
    """
    scores: list[float] = []
    if groups is None:
        n = len(y_true)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            s = _safe_metric(metric_fn, y_true[mask], y_score[mask])
            scores.append(float("nan") if s is None else s)
    else:
        unique_groups = np.unique(groups)
        for g in unique_groups:
            mask = groups != g
            s = _safe_metric(metric_fn, y_true[mask], y_score[mask])
            scores.append(float("nan") if s is None else s)
    return np.asarray(scores, dtype=float)


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: MetricFn,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
    groups: np.ndarray | None = None,
    method: str = "bca",
) -> dict:
    """Bootstrap confidence interval for any scalar metric.

    Args:
        y_true:       True binary labels (1-D array, length n).
        y_score:      Predicted probabilities or scores (1-D array, length n).
        metric_fn:    ``f(y_true, y_score) -> float``.  Called on every
                      bootstrap sample and on ``(y_true, y_score)`` for the
                      point estimate.  Return NaN or raise to mark a sample
                      as degenerate (e.g. only one class present).
        n_bootstrap:  Number of bootstrap iterations.
        ci:           Coverage probability (e.g. 0.95).
        random_state: RNG seed.
        groups:       Optional group identifier per observation.  When
                      given, bootstrap resamples *groups* with replacement
                      and emits every observation in each sampled group
                      (cluster bootstrap).  The jackknife used for BCa
                      acceleration also switches to leave-one-group-out.
        method:       ``"bca"`` (default) or ``"percentile"``.  BCa falls
                      back to percentile if acceleration is ill-defined
                      (e.g. constant jackknife, or all bootstrap samples
                      ≥ / ≤ point estimate).

    Returns:
        Dict with keys:
            point_estimate        — metric on the full sample.
            ci_lower, ci_upper    — interval endpoints (NaN if undefined).
            ci                    — the requested coverage probability.
            method_used           — "bca" or "percentile" (records any
                                    fallback).
            n_bootstrap           — requested iterations.
            n_valid               — iterations that yielded a finite metric.
            bootstrap_distribution — 1-D np.ndarray of valid metric values.
            cluster               — True if group bootstrap was used.
            n_groups              — number of distinct groups (0 if none).
    """
    if method not in {"bca", "percentile"}:
        raise ValueError(f"Unknown method '{method}'; use 'bca' or 'percentile'.")

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(random_state)

    # ── Point estimate on the full sample ─────────────────────────────────────
    pe = _safe_metric(metric_fn, y_true, y_score)
    point_estimate = float("nan") if pe is None else pe

    # ── Bootstrap sampling ────────────────────────────────────────────────────
    boot_scores: list[float] = []
    if groups is None:
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            s = _safe_metric(metric_fn, y_true[idx], y_score[idx])
            if s is not None:
                boot_scores.append(s)
        n_groups = 0
        cluster = False
    else:
        groups = np.asarray(groups)
        if len(groups) != len(y_true):
            raise ValueError("len(groups) must equal len(y_true)")
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        # Pre-build index per group once — O(n) one-time work
        group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
        for _ in range(n_bootstrap):
            chosen = rng.choice(unique_groups, size=n_groups, replace=True)
            idx = np.concatenate([group_to_idx[g] for g in chosen])
            s = _safe_metric(metric_fn, y_true[idx], y_score[idx])
            if s is not None:
                boot_scores.append(s)
        cluster = True

    boot = np.asarray(boot_scores, dtype=float)
    n_valid = len(boot)

    # ── Interval estimation ──────────────────────────────────────────────────
    method_used = method
    if n_valid < 2 or not np.isfinite(point_estimate):
        ci_lower = float("nan")
        ci_upper = float("nan")
    else:
        bca_ok: tuple[float, float] | None = None
        if method == "bca":
            jk = _jackknife(y_true, y_score, metric_fn, groups)
            bca_ok = _bca_interval(point_estimate, boot, jk, ci)
            if bca_ok is None:
                method_used = "percentile"

        if bca_ok is not None:
            ci_lower, ci_upper = bca_ok
        else:
            alpha = 1.0 - ci
            ci_lower = float(np.quantile(boot, alpha / 2.0))
            ci_upper = float(np.quantile(boot, 1.0 - alpha / 2.0))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci": ci,
        "method_used": method_used,
        "n_bootstrap": n_bootstrap,
        "n_valid": n_valid,
        "bootstrap_distribution": boot,
        "cluster": cluster,
        "n_groups": n_groups,
    }


# ── Convenience wrappers ──────────────────────────────────────────────────────

def bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    **kwargs,
) -> dict:
    """Bootstrap CI for ROC AUC.  See :func:`bootstrap_metric` for kwargs."""
    def _auc(yt: np.ndarray, ys: np.ndarray) -> float:
        if len(np.unique(yt)) < 2:
            raise ValueError("AUC undefined: one class only")
        return float(roc_auc_score(yt, ys))

    return bootstrap_metric(y_true, y_score, _auc, **kwargs)


def bootstrap_precision_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    **kwargs,
) -> dict:
    """Bootstrap CI for precision at a fixed probability threshold."""
    def _prec(yt: np.ndarray, ys: np.ndarray) -> float:
        y_pred = (ys >= threshold).astype(int)
        if int(np.sum(y_pred)) == 0:
            raise ValueError("Precision undefined: no positive predictions")
        return float(precision_score(yt, y_pred, zero_division=0))  # type: ignore[arg-type]

    return bootstrap_metric(y_true, y_score, _prec, **kwargs)


def bootstrap_recall_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    **kwargs,
) -> dict:
    """Bootstrap CI for recall at a fixed probability threshold."""
    def _rec(yt: np.ndarray, ys: np.ndarray) -> float:
        if int(np.sum(yt)) == 0:
            raise ValueError("Recall undefined: no true positives in sample")
        y_pred = (ys >= threshold).astype(int)
        return float(recall_score(yt, y_pred, zero_division=0))  # type: ignore[arg-type]

    return bootstrap_metric(y_true, y_score, _rec, **kwargs)
