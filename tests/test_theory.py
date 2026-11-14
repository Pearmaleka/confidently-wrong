"""Every closed form in the note, checked against simulation or against brute force."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from cwrong import (
    Population,
    asymptotic_curve,
    calibration_size,
    cluster_ci,
    design_effect,
    fixed_sequence,
    icc_estimate,
    naive_coverage,
    optimal_runs,
    policy_risk,
    risk_coverage,
    selective_floor,
    simulate_votes,
    two_source_risk,
    unanimity_prob,
    vote_table,
)

MIXED = Population.mixture(right=0.30, wrong=0.15, grid=800)
NOISY = Population.mixture(right=0.0, wrong=0.0, grid=800)


def test_population_summaries():
    assert MIXED.pi_d == pytest.approx(0.45)
    assert MIXED.rho == pytest.approx(1 / 3)
    rng = np.random.default_rng(0)
    a = MIXED.sample(400_000, rng)
    assert np.mean(1 - a) == pytest.approx(MIXED.error_rate, abs=3e-3)
    with pytest.raises(ValueError):
        Population.mixture(0.7, 0.5)
    with pytest.raises(ValueError):
        risk_coverage(MIXED, 4)


@pytest.mark.parametrize("k", [3, 7])
def test_risk_coverage_matches_simulation(k):
    rng = np.random.default_rng(k)
    _, votes, right = simulate_votes(MIXED, 300_000, k, rng)
    for m, cov, risk in risk_coverage(MIXED, k):
        answered = votes >= m
        assert answered.mean() == pytest.approx(cov, abs=4e-3)
        assert (~right[answered]).mean() == pytest.approx(risk, abs=6e-3)


@settings(max_examples=300, deadline=None)
@given(
    right=st.floats(0, 0.6),
    wrong=st.floats(0, 0.4),
    k=st.sampled_from([1, 3, 5, 9, 15]),
    phi=st.lists(st.floats(0, 1), min_size=8, max_size=8),
    b=st.floats(0.5, 8),
)
def test_theorem_1_holds_for_every_label_invariant_policy(right, wrong, k, phi, b):
    """In the binary model a label-invariant policy is any function of the votes V."""
    if right + wrong > 0.95 or right + wrong < 0.02:
        return
    pop = Population.mixture(right, wrong, beta=(b, 2.0), grid=200)
    n = (k + 1) // 2
    p = np.array(phi[:n])
    cov, risk = policy_risk(pop, k, p)
    if cov < 1e-9:
        return
    pd, rho = pop.pi_d, pop.rho
    # (a) any policy
    assert risk >= rho * max(cov - (1 - pd), 0) / cov - 1e-9
    # (b) monotone: unanimity answered at least as readily as anything else
    mono = p.copy()
    mono[-1] = mono.max()
    _, risk_m = policy_risk(pop, k, mono)
    assert risk_m >= pd * rho - 1e-9
    # (c) threshold policies on the votes, randomised at the boundary
    for m in range(n):
        for frac in (0.3, 1.0):
            thr = np.zeros(n)
            thr[m + 1 :] = 1.0
            thr[m] = frac
            cov_t, risk_t = policy_risk(pop, k, thr)
            assert risk_t >= float(selective_floor(pop, k, cov_t)) - 1e-9


def test_unanimity_tends_to_rho_and_curves_converge():
    assert unanimity_prob(MIXED, 401) == pytest.approx(MIXED.pi_d, abs=0.01)
    top = risk_coverage(MIXED, 401)[0]
    assert top[2] == pytest.approx(MIXED.rho, abs=0.02)
    # Large k agreement thresholding approaches the |2A - 1| oracle curve.
    asym = asymptotic_curve(MIXED)
    k = 801
    for m, cov, risk in risk_coverage(MIXED, k)[::40]:
        t = 2 * m / k - 1
        ref = min(asym, key=lambda row: abs(row[0] - t))
        assert risk == pytest.approx(ref[2], abs=0.03)
        assert cov == pytest.approx(ref[1], abs=0.03)


def test_noisy_population_benefits_mixed_population_does_not():
    noisy = risk_coverage(NOISY, 9)
    assert noisy[0][2] < 0.5 * noisy[-1][2]  # abstaining helps a lot
    mixed = risk_coverage(MIXED, 9)
    assert mixed[0][2] > mixed[-1][2]  # the most confident answers are the worst


def test_two_source_formula():
    rng = np.random.default_rng(3)
    rho, b1, b0 = 0.4, 0.3, 0.8
    n = 400_000
    wrong = rng.random(n) < rho
    agree = np.where(wrong, rng.random(n) < b1, rng.random(n) < b0)
    assert wrong[agree].mean() == pytest.approx(two_source_risk(rho, b1, b0), abs=4e-3)
    assert two_source_risk(rho, 0.6, 0.6) == pytest.approx(rho)  # shared errors: no gain


def test_fixed_sequence_is_valid_and_abstains_when_it_must():
    rng = np.random.default_rng(4)
    k, alpha, delta = 5, 0.12, 0.1
    # Population where some thresholds have risk below alpha: validity check.
    pop = Population.mixture(right=0.35, wrong=0.02, grid=400)
    exact = {m: r for m, _, r in risk_coverage(pop, k)}
    thresholds = [m for m, _, _ in risk_coverage(pop, k)]
    bad, certified = 0, 0
    reps = 2000
    for _ in range(reps):
        _, votes, right = simulate_votes(pop, 300, k, rng)
        j = fixed_sequence([votes >= m for m in thresholds], ~right, alpha, delta)
        if j >= 0:
            certified += 1
            bad += exact[thresholds[j]] > alpha
    assert bad / reps <= delta + 0.02
    assert certified / reps > 0.5
    # pi_D rho > alpha: every monotone threshold is a true null, so abstain w.p. >= 1 - delta.
    assert MIXED.pi_d * MIXED.rho > alpha
    abstain = 0
    for _ in range(500):
        _, votes, right = simulate_votes(MIXED, 300, k, rng)
        abstain += fixed_sequence([votes >= m for m in thresholds], ~right, alpha, delta) < 0
    assert abstain / 500 >= 1 - delta - 0.03


def test_calibration_size_heuristic():
    n = calibration_size(0.05, 0.1, 0.5, 0.1)
    assert 250 < n < 300
    with pytest.raises(ValueError):
        calibration_size(0.2, 0.1, 0.5, 0.1)


def test_design_effect_and_naive_coverage_match_simulation():
    rng = np.random.default_rng(5)
    pop, n_tasks, k, reps = MIXED, 60, 5, 3000
    theta = pop.error_rate
    est, naive_hits, cluster_hits = [], 0, 0
    for _ in range(reps):
        a = pop.sample(n_tasks, rng)
        z = (rng.random((n_tasks, k)) > a[:, None]).astype(float)
        m = z.mean()
        est.append(m)
        se = np.sqrt(m * (1 - m) / (n_tasks * k))
        naive_hits += abs(m - theta) <= 1.96 * se
        lo, hi = cluster_ci(z)
        cluster_hits += lo <= theta <= hi
    var_formula = theta * (1 - theta) / (n_tasks * k) * design_effect(pop.icc, k)
    assert np.var(est) == pytest.approx(var_formula, rel=0.08)
    assert naive_hits / reps == pytest.approx(naive_coverage(pop.icc, k), abs=0.03)
    assert cluster_hits / reps == pytest.approx(0.95, abs=0.02)


def test_icc_estimate_and_allocation():
    rng = np.random.default_rng(6)
    a = MIXED.sample(20_000, rng)
    z = (rng.random((20_000, 4)) > a[:, None]).astype(float)
    assert icc_estimate(z) == pytest.approx(MIXED.icc, abs=0.02)
    assert optimal_runs(1.0, 10) == 0.0
    assert optimal_runs(0.2, 4) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        icc_estimate(z[:, :1])


def test_vote_table_sums_to_one():
    for k in (1, 3, 11):
        _, r, w = vote_table(MIXED, k)
        assert (r + w).sum() == pytest.approx(1.0)
        assert w.sum() == pytest.approx(1 - stats.binom.sf(k // 2, k, MIXED.a) @ MIXED.w)
