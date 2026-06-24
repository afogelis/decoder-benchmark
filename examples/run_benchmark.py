"""Run a decoder benchmark and save the leaderboard, Pareto and accuracy plots.

    python examples/run_benchmark.py

Outputs (gitignored) are written to ``outputs/``.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from decbench import build_leaderboard, format_leaderboard, run_benchmark
from decbench.types import BenchmarkConfig
from decbench.viz import plot_accuracy_vs_p, plot_pareto


def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    config = BenchmarkConfig(
        decoders=["mwpm", "union_find", "bp"],
        distances=[3, 5],
        error_rates=[0.005, 0.008, 0.01, 0.012],
        shots=5_000,
        seed=2026,
    )
    result = run_benchmark(config)

    print(format_leaderboard(build_leaderboard(result)))
    with open("outputs/benchmark.json", "w", encoding="utf-8") as handle:
        json.dump(json.loads(result.model_dump_json()), handle, indent=2)

    ax = plot_pareto(result)
    ax.figure.tight_layout()
    ax.figure.savefig("outputs/pareto.png", dpi=150)
    plt.close(ax.figure)

    ax = plot_accuracy_vs_p(result, distance=5)
    ax.figure.tight_layout()
    ax.figure.savefig("outputs/accuracy_vs_p_d5.png", dpi=150)
    plt.close(ax.figure)

    print("saved outputs/benchmark.json, outputs/pareto.png, outputs/accuracy_vs_p_d5.png")


if __name__ == "__main__":
    main()
