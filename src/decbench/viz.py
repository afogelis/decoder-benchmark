"""Accuracy/runtime trade-off plots for the decoder benchmark."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .leaderboard import build_leaderboard
from .types import BenchmarkResult


def plot_pareto(result: BenchmarkResult, *, ax: Axes | None = None) -> Axes:
    """Scatter mean runtime vs mean logical error rate, one point per decoder.

    The lower-left frontier is the Pareto-optimal set: decoders that are both
    fast and accurate. This is the single most useful summary for picking a
    decoder under a latency budget.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    for row in build_leaderboard(result):
        ax.scatter(row.mean_microseconds_per_shot, row.mean_logical_error_rate, s=80)
        ax.annotate(
            row.decoder,
            (row.mean_microseconds_per_shot, row.mean_logical_error_rate),
            textcoords="offset points",
            xytext=(6, 4),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean runtime (microseconds / shot)")
    ax.set_ylabel("Mean logical error rate")
    ax.set_title("Decoder accuracy vs runtime")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_accuracy_vs_p(result: BenchmarkResult, *, distance: int, ax: Axes | None = None) -> Axes:
    """Plot logical error rate vs physical error rate per decoder at one distance."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    decoders = sorted({record.decoder for record in result.records})
    for decoder in decoders:
        rows = sorted(
            (r for r in result.records if r.decoder == decoder and r.distance == distance),
            key=lambda r: r.p,
        )
        if not rows:
            continue
        ax.plot(
            [r.p for r in rows], [r.logical_error_rate for r in rows], marker="o", label=decoder
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Logical error rate")
    ax.set_title(f"Decoder accuracy at distance d = {distance}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax
