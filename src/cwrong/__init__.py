"""Limits of sampling-based confidence for LLM agents: closed forms and simulations."""

from cwrong.binary import (
    Population,
    asymptotic_curve,
    policy_risk,
    risk_coverage,
    selective_floor,
    unanimity_prob,
    vote_table,
)
from cwrong.certify import binomial_pvalue, calibration_size, fixed_sequence
from cwrong.evaluation import (
    cluster_ci,
    design_effect,
    icc_estimate,
    naive_coverage,
    optimal_runs,
)
from cwrong.simulate import simulate_votes, two_source_risk

__all__ = [
    "Population",
    "asymptotic_curve",
    "binomial_pvalue",
    "calibration_size",
    "cluster_ci",
    "design_effect",
    "fixed_sequence",
    "icc_estimate",
    "naive_coverage",
    "optimal_runs",
    "policy_risk",
    "risk_coverage",
    "selective_floor",
    "simulate_votes",
    "two_source_risk",
    "unanimity_prob",
    "vote_table",
]

__version__ = "0.1.0"
