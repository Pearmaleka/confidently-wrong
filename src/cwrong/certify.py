"""Finite-sample certification of a selective-risk bound (Learn then Test, fixed sequence).

Candidate thresholds are tested from the most to the least selective. For threshold ``j``
the null hypothesis is ``H_j: risk(j) > alpha``; its p-value is the exact binomial lower tail
``P(Bin(N_j, alpha) <= E_j)`` where ``N_j`` calibration tasks are answered and ``E_j`` of them
wrong. The procedure keeps the last threshold of the initial run of rejections at level
``delta`` and abstains on everything if the first test fails. Fixed-sequence testing controls
the family-wise error, so ``P(risk(chosen) > alpha) <= delta``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats

__all__ = ["binomial_pvalue", "calibration_size", "fixed_sequence"]


def binomial_pvalue(errors: int, answered: int, alpha: float) -> float:
    if answered == 0:
        return 1.0
    return float(stats.binom.cdf(errors, answered, alpha))


def fixed_sequence(
    answered: Sequence[np.ndarray], wrong: np.ndarray, alpha: float, delta: float
) -> int:
    """Index of the certified threshold, or -1 to abstain on everything.

    ``answered[j]`` is a boolean mask of the calibration tasks answered at threshold ``j``
    (thresholds ordered from most to least selective); ``wrong`` marks wrong answers."""
    chosen = -1
    for j, mask in enumerate(answered):
        p = binomial_pvalue(int(np.sum(wrong & mask)), int(np.sum(mask)), alpha)
        if p > delta:
            break
        chosen = j
    return chosen


def kl_bernoulli(p: float, q: float) -> float:
    """``kl(p || q)`` between Bernoulli distributions."""
    out = 0.0
    if p > 0:
        out += p * math.log(p / q)
    if p < 1:
        out += (1 - p) * math.log((1 - p) / (1 - q))
    return out


def calibration_size(risk: float, alpha: float, coverage: float, delta: float) -> float:
    """Chernoff heuristic: labelled tasks needed to certify a threshold whose true risk is

    ``risk < alpha`` at answer rate ``coverage``:
    ``log(1/delta) / (coverage kl(risk || alpha))``."""
    if not 0 <= risk < alpha < 1:
        raise ValueError("need 0 <= risk < alpha < 1")
    return math.log(1 / delta) / (coverage * kl_bernoulli(risk, alpha))
