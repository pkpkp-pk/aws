"""Metrics for ranker evaluation. Pure functions over labeled pools.

Labels: per-candidate pass/fail from the cached pool (ground truth).
Rankers emit an ordering (best first) and, where available, a 0..1
score per candidate for AUROC/calibration.
"""
from __future__ import annotations

import random


def runs_to_first_pass(order: list[int], labels: list[bool]) -> int | None:
    """Position (1-based) of the first passing candidate in `order`.
    None if no candidate passes."""
    for rank, idx in enumerate(order, start=1):
        if labels[idx]:
            return rank
    return None


def top_k_hit(order: list[int], labels: list[bool], k: int) -> bool:
    return any(labels[idx] for idx in order[:k])


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """Rank-based AUROC. None when labels are all one class."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def brier(scores: list[float], labels: list[bool]) -> float | None:
    """Brier score of 0..1 scores vs binary labels. None if empty."""
    if not scores:
        return None
    return sum((s - float(y)) ** 2 for s, y in zip(scores, labels)) / len(scores)


def bootstrap_ci(values: list[float], n: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    rng = random.Random(seed)
    if not values:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(alpha / 2 * n)]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return lo, hi
