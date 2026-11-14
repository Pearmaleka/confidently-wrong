"""Monte Carlo counterparts of the closed forms, used to check them."""

from __future__ import annotations

import numpy as np

from cwrong.binary import Population

__all__ = ["simulate_votes", "two_source_risk"]


def simulate_votes(
    pop: Population, n_tasks: int, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw tasks and ``k`` runs each. Returns run accuracies, agreement votes ``V`` and

    whether the majority answer is correct."""
    a = pop.sample(n_tasks, rng)
    n_correct = rng.binomial(k, a)
    votes = np.maximum(n_correct, k - n_correct)
    return a, votes, n_correct > k / 2


def two_source_risk(rho: float, beta1: float, beta0: float) -> float:
    """Proposition 3: on deterministic tasks, answering only when a second source agrees.

    ``beta1``: probability that the second source gives the same answer when the agent is
    wrong; ``beta0``: when it is right."""
    num = rho * beta1
    den = num + (1 - rho) * beta0
    return num / den if den > 0 else float("nan")
