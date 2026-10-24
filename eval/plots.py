"""Figures for the article: Pareto (accuracy vs latency), calibration, bars.

Reads results/metrics.json (from eval.run_rankers). Writes PNGs to results/.

    python -m eval.plots
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = json.loads((RESULTS / "metrics.json").read_text())
    names = list(metrics)

    # 1. Pareto: mean runs-to-first-pass vs latency
    fig, ax = plt.subplots(figsize=(6, 4))
    for n in names:
        m = metrics[n]
        if m["mean_runs_to_first_pass"] is None:
            continue
        ax.scatter(m["mean_latency_s"], m["mean_runs_to_first_pass"], label=n)
        ax.annotate(n, (m["mean_latency_s"], m["mean_runs_to_first_pass"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean decision latency (s)")
    ax.set_ylabel("mean test runs to first pass")
    ax.set_title("Search cost vs decision cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "pareto.png", dpi=150)

    # 2. Bars: runs to first pass with CI
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [metrics[n]["mean_runs_to_first_pass"] or 0 for n in names]
    errs = []
    for n in names:
        lo, hi = metrics[n]["rfp_ci95"]
        mean = metrics[n]["mean_runs_to_first_pass"] or 0
        errs.append([max(0, mean - lo), max(0, hi - mean)])
    ax.bar(names, vals, yerr=list(zip(*errs)), capsize=4)
    ax.set_ylabel("mean test runs to first pass")
    ax.set_title("Ranker comparison (lower is better)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(RESULTS / "runs_to_first_pass.png", dpi=150)

    print(f"wrote {RESULTS/'pareto.png'} and {RESULTS/'runs_to_first_pass.png'}")


if __name__ == "__main__":
    main()
