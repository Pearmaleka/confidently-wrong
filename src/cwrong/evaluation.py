"""Evaluating an agent with several runs per task: design effect and allocation.

With ``I`` tasks and ``k`` runs each, the run-level error rate estimate has variance

    Var = theta (1 - theta) / (I k) * (1 + (k - 1) icc),

where ``icc = Var(e(X)) / (theta (1 - theta))`` is the intra-task correlation of errors and
``e(x)`` the error probability on task ``x``. The same quantity measures how much of the error
variance is systematic, which is what sampling-based confidence cannot see.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

__all__ = [
    "cluster_ci",
    "design_effect",
    "icc_estimate",
    "naive_coverage",
    "optimal_runs",
]


def design_effect(icc: float, k: int) -> float:
    return 1 + (k - 1) * icc


def naive_coverage(icc: float, k: int, level: float = 0.95) -> float:
    """Asymptotic coverage of the binomial CI that treats all ``I k`` runs as independent."""
    z = stats.norm.ppf(0.5 + level / 2)
    return float(2 * stats.norm.cdf(z / math.sqrt(design_effect(icc, k))) - 1)


def optimal_runs(icc: float, cost_ratio: float) -> float:
    """Runs per task minimising the variance for a fixed budget, when a new task costs

    ``cost_ratio`` times a run: ``sqrt(cost_ratio (1 - icc) / icc)``."""
    if not 0 < icc <= 1 or cost_ratio <= 0:
        raise ValueError("need 0 < icc <= 1 and cost_ratio > 0")
    return math.sqrt(cost_ratio * (1 - icc) / icc)


def icc_estimate(errors: np.ndarray) -> float:
    """Moment estimate of the error ICC from an ``I x k`` 0/1 array (``k >= 2``).

    Uses ``E[Z_t Z_s] = E[e(X)^2]`` for two different runs on the same task."""
    z = np.asarray(errors, dtype=float)
    k = z.shape[1]
    if k < 2:
        raise ValueError("need at least two runs per task")
    s = z.sum(axis=1)
    theta = z.mean()
    second = np.mean(s * (s - 1) / (k * (k - 1)))
    return float((second - theta**2) / (theta * (1 - theta)))


def cluster_ci(errors: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """CI for the error rate from task means (runs within a task are not independent)."""
    m = np.asarray(errors, dtype=float).mean(axis=1)
    se = m.std(ddof=1) / math.sqrt(len(m))
    z = stats.norm.ppf(0.5 + level / 2)
    return float(m.mean() - z * se), float(m.mean() + z * se)
