"""Figures and numbers for the note. Run from the repository root:

    python experiments/make_figures.py

Writes paper/figures/*.pdf and experiments/results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cwrong import (
    Population,
    asymptotic_curve,
    calibration_size,
    fixed_sequence,
    naive_coverage,
    risk_coverage,
    selective_floor,
    simulate_votes,
    two_source_risk,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
BLUE, ORANGE, GREEN, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#0b0b0b"

POPULATIONS = {
    "noisy": Population.mixture(right=0.0, wrong=0.0),
    "mixed": Population.mixture(right=0.30, wrong=0.15),
    "mild": Population.mixture(right=0.35, wrong=0.02),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
    }
)


def curves_figure(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    rng = np.random.default_rng(0)
    for ax, name, title in zip(
        axes,
        ("noisy", "mixed"),
        ("(a) no deterministic tasks", "(b) 45% deterministic, a third of them wrong"),
        strict=True,
    ):
        pop = POPULATIONS[name]
        for k, color in ((5, BLUE), (21, GREEN)):
            rc = risk_coverage(pop, k)
            ax.plot(
                [c for _, c, _ in rc],
                [r for _, _, r in rc],
                "-o",
                ms=2.5,
                lw=1.4,
                color=color,
                label=f"agreement, $k={k}$",
            )
        asym = asymptotic_curve(pop)
        ax.plot(
            [c for _, c, _ in asym],
            [r for _, _, r in asym],
            color=INK,
            lw=1.0,
            ls="--",
            label=r"$k\to\infty$",
        )
        # Monte Carlo check for k = 5
        _, votes, right = simulate_votes(pop, 200_000, 5, rng)
        for m in range(5, 2, -1):
            ans = votes >= m
            ax.plot(
                ans.mean(),
                (~right[ans]).mean(),
                "x",
                color=ORANGE,
                ms=5,
                mew=1.1,
                label="simulation, $k=5$" if m == 5 else None,
            )
        if pop.pi_d > 0:
            c = np.linspace(0.02, 1, 200)
            ax.plot(
                c,
                selective_floor(pop, 21, c),
                color=GREY,
                lw=1.0,
                ls=":",
                label="Theorem 1(c) floor, $k=21$",
            )
            ax.axhline(pop.pi_d * pop.rho, color=GREY, lw=0.8, ls="-.")
            ax.text(0.03, pop.pi_d * pop.rho + 0.008, r"$\pi_D\rho$", color=GREY, fontsize=8)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("coverage (share of tasks answered)")
        ax.set_xlim(0, 1.02)
        results[f"curve_{name}"] = {
            str(k): [[m, c, r] for m, c, r in risk_coverage(pop, k)] for k in (5, 21)
        }
    axes[0].set_ylabel("selective risk (error rate when answering)")
    axes[0].set_ylim(0, 0.42)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", fontsize=6.8, frameon=False)
    fig.savefig(FIG / "risk_coverage.pdf")
    fig.savefig(FIG / "risk_coverage.png", dpi=200)
    plt.close(fig)


def certification_figure(results: dict) -> None:
    k, alpha, delta, reps = 5, 0.10, 0.10, 1000
    rng = np.random.default_rng(1)
    sizes = [50, 100, 200, 400, 800, 1600]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    out = {}
    for name, color in (("mild", BLUE), ("mixed", ORANGE)):
        pop = POPULATIONS[name]
        rc = risk_coverage(pop, k)
        thresholds = [m for m, _, _ in rc]
        cov_of = {m: c for m, c, _ in rc}
        risk_of = {m: r for m, _, r in rc}
        feasible = [c for _, c, r in rc if r <= alpha]
        means, violations = [], []
        for n in sizes:
            certified, bad = [], 0
            for _ in range(reps):
                _, votes, right = simulate_votes(pop, n, k, rng)
                j = fixed_sequence([votes >= m for m in thresholds], ~right, alpha, delta)
                certified.append(cov_of[thresholds[j]] if j >= 0 else 0.0)
                bad += j >= 0 and risk_of[thresholds[j]] > alpha
            means.append(float(np.mean(certified)))
            violations.append(bad / reps)
        ax.plot(
            sizes,
            means,
            "-o",
            ms=3,
            color=color,
            lw=1.4,
            label=f"{name}: $\\pi_D\\rho={pop.pi_d * pop.rho:.3f}$",
        )
        if feasible:
            ax.axhline(max(feasible), color=color, lw=0.8, ls=":")
        out[name] = {
            "n": sizes,
            "mean_certified_coverage": means,
            "violation_rate": violations,
            "best_feasible_coverage": max(feasible) if feasible else 0.0,
            "pi_d_rho": pop.pi_d * pop.rho,
        }
    ax.set_xscale("log")
    ax.set_xlabel("labelled calibration tasks $n$")
    ax.set_ylabel("mean certified coverage")
    ax.set_ylim(-0.03, 1.0)
    ax.legend(fontsize=7, frameon=False, loc="center right")
    fig.savefig(FIG / "certification.pdf")
    fig.savefig(FIG / "certification.png", dpi=200)
    plt.close(fig)
    results["certification"] = {"k": k, "alpha": alpha, "delta": delta, "reps": reps, **out}


def numbers(results: dict) -> None:
    for name, pop in POPULATIONS.items():
        full = {k: risk_coverage(pop, k)[-1][2] for k in (1, 5, 21)}
        top = {k: risk_coverage(pop, k)[0][2] for k in (5, 21)}
        results[f"population_{name}"] = {
            "pi_d": pop.pi_d,
            "rho": pop.rho if pop.pi_d > 0 else None,
            "run_error_rate": pop.error_rate,
            "icc": pop.icc,
            "full_coverage_risk": full,
            "unanimous_risk": top,
        }
    results["naive_ci_coverage"] = {
        f"icc={icc},k={k}": naive_coverage(icc, k)
        for icc in (0.2, 0.5, 0.8, 1.0)
        for k in (3, 5, 10)
    }
    results["two_source"] = {
        f"rho=0.5,b={b}": two_source_risk(0.5, 1 - b, b) for b in (0.6, 0.8, 0.95)
    }
    results["calibration_size"] = {
        "risk=0.05,alpha=0.10,c=0.5,delta=0.1": calibration_size(0.05, 0.10, 0.5, 0.1),
        "risk=0.02,alpha=0.05,c=0.5,delta=0.1": calibration_size(0.02, 0.05, 0.5, 0.1),
    }


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    numbers(results)
    curves_figure(results)
    certification_figure(results)
    path = ROOT / "experiments" / "results.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if not k.startswith("curve")}, indent=1))


if __name__ == "__main__":
    main()
