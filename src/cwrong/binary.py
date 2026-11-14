"""The binary model: closed-form selective risk of agreement-based abstention.

A task is a decision with two possible answers (e.g. "is the effect significant?"). On a task
``x`` each run of the agent is correct independently with probability ``A(x)``, its *run
accuracy*. Over a population of tasks ``A`` has some distribution, represented here by atoms.
With ``k`` runs (``k`` odd) the agent reports the majority answer, and the *agreement* is the
share of runs that voted for it, ``V / k`` with ``V = max(N, k - N)`` and ``N`` the number of
correct runs. Agreement is a label-invariant score: it never looks at which answer won.

Answering when ``V >= m`` gives

    coverage(m) = E[ P(N >= m | A) + P(N <= k - m | A) ]
    risk(m)     = E[ P(N <= k - m | A) ] / coverage(m)

since for ``m > k/2`` an answered task has a wrong majority exactly when ``k - N >= m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "Population",
    "asymptotic_curve",
    "policy_risk",
    "risk_coverage",
    "selective_floor",
    "unanimity_prob",
    "vote_table",
]


@dataclass(frozen=True)
class Population:
    """Distribution of per-task run accuracy ``A``, as atoms ``a`` with weights ``w``."""

    a: np.ndarray
    w: np.ndarray

    def __post_init__(self) -> None:
        if self.a.shape != self.w.shape or self.a.ndim != 1:
            raise ValueError("a and w must be 1-d arrays of the same length")
        if (self.a < 0).any() or (self.a > 1).any() or (self.w < 0).any():
            raise ValueError("a must lie in [0, 1] and w must be non-negative")
        if not np.isclose(self.w.sum(), 1.0):
            raise ValueError("weights must sum to 1")

    @classmethod
    def mixture(
        cls,
        right: float,
        wrong: float,
        beta: tuple[float, float] = (6.0, 2.0),
        grid: int = 4000,
    ) -> Population:
        """Deterministic tasks (``A = 1`` with mass ``right``, ``A = 0`` with mass ``wrong``)

        plus a Beta(``beta``) continuum of noisy tasks, discretised on ``grid`` midpoints."""
        if right < 0 or wrong < 0 or right + wrong > 1:
            raise ValueError("right and wrong must be non-negative and sum to at most 1")
        mid = (np.arange(grid) + 0.5) / grid
        dens = stats.beta.pdf(mid, *beta)
        cont = (1 - right - wrong) * dens / dens.sum()
        a = np.concatenate([[0.0, 1.0], mid])
        w = np.concatenate([[wrong, right], cont])
        return cls(a, w)

    @property
    def pi_d(self) -> float:
        """Mass of deterministic tasks, ``P(A in {0, 1})``."""
        return float(self.w[(self.a == 0) | (self.a == 1)].sum())

    @property
    def rho(self) -> float:
        """Share of deterministic tasks on which the agent is always wrong."""
        return float(self.w[self.a == 0].sum() / self.pi_d) if self.pi_d > 0 else float("nan")

    @property
    def error_rate(self) -> float:
        """Per-run error rate ``theta = E[1 - A]``."""
        return float(np.sum(self.w * (1 - self.a)))

    @property
    def icc(self) -> float:
        """Intra-task correlation of run errors, ``Var(1 - A) / (theta (1 - theta))``."""
        theta = self.error_rate
        var = float(np.sum(self.w * (1 - self.a) ** 2)) - theta**2
        return var / (theta * (1 - theta))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.a, size=n, p=self.w)


def _check_k(k: int) -> None:
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be a positive odd number (no ties in the majority)")


def risk_coverage(pop: Population, k: int) -> list[tuple[int, float, float]]:
    """``(m, coverage, risk)`` for every threshold ``m = k, k - 1, ..., (k + 1) / 2``."""
    _check_k(k)
    out = []
    for m in range(k, (k + 1) // 2 - 1, -1):
        right = stats.binom.sf(m - 1, k, pop.a)  # P(N >= m)
        wrong = stats.binom.cdf(k - m, k, pop.a)  # P(N <= k - m)
        cov = float(np.sum(pop.w * (right + wrong)))
        err = float(np.sum(pop.w * wrong))
        out.append((m, cov, err / cov if cov > 0 else float("nan")))
    return out


def unanimity_prob(pop: Population, k: int) -> float:
    """``P(U)``: all ``k`` runs agree, ``E[A^k + (1 - A)^k]``."""
    return float(np.sum(pop.w * (pop.a**k + (1 - pop.a) ** k)))


def selective_floor(pop: Population, k: int, coverage: np.ndarray | float) -> np.ndarray:
    """Theorem 1(c): any threshold policy on a label-invariant score that ranks unanimity

    first has selective risk at least ``pi_D rho / max(c, P(U))``."""
    c = np.asarray(coverage, dtype=float)
    return pop.pi_d * pop.rho / np.maximum(c, unanimity_prob(pop, k))


def asymptotic_curve(pop: Population) -> list[tuple[float, float, float]]:
    """``(t, coverage, risk)`` of answering when ``|2A - 1| >= t``: the limit ``k -> inf``.

    In the limit the agreement reveals how strongly the agent prefers one answer, ``|2A - 1|``,
    and nothing about whether that answer is right, ``sign(2A - 1)``."""
    strength = np.abs(2 * pop.a - 1)
    out = []
    for t in np.unique(strength[pop.w > 0])[::-1]:
        keep = strength >= t
        cov = float(pop.w[keep].sum())
        err = float(pop.w[keep & (pop.a < 0.5)].sum())
        out.append((float(t), cov, err / cov))
    return out


def vote_table(pop: Population, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For ``v = (k + 1) / 2, ..., k``: ``P(V = v, majority right)`` and ``P(V = v, wrong)``.

    In the binary model every label-invariant score is a function of ``V`` (and of independent
    randomness), so these two vectors determine the selective risk of every such policy."""
    _check_k(k)
    v = np.arange((k + 1) // 2, k + 1)
    right = np.array([np.sum(pop.w * stats.binom.pmf(x, k, pop.a)) for x in v])
    wrong = np.array([np.sum(pop.w * stats.binom.pmf(k - x, k, pop.a)) for x in v])
    return v, right, wrong


def policy_risk(pop: Population, k: int, answer_prob: np.ndarray) -> tuple[float, float]:
    """Coverage and selective risk of answering with probability ``answer_prob[i]`` when

    ``V = (k + 1) / 2 + i``."""
    _, right, wrong = vote_table(pop, k)
    phi = np.asarray(answer_prob, dtype=float)
    cov = float(np.sum(phi * (right + wrong)))
    return cov, float(np.sum(phi * wrong) / cov) if cov > 0 else float("nan")
