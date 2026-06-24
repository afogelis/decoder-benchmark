"""Accuracy/runtime trade-off plots for the decoder benchmark."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .leaderboard import build_leaderboard
from .types import BenchmarkResult


def plot_pareto(result: BenchmarkResult, *, ax: Axes | None = None) -> Axes:
    """Scatter mean runtime vs mean logical error rate, grouped by backend.

    Points are coloured by implementation backend. The horizontal (runtime)
    axis is only comparable *within* a backend: a compiled C++ decoder will
    out-run a pure-Python one regardless of algorithm, so the cross-language
    runtime gap reflects the language, not the decoding quality. The vertical
    (accuracy) axis is comparable across all points.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5.5))

    rows = build_leaderboard(result)
    backends = sorted({row.backend for row in rows})
    markers = ["o", "s", "^", "D", "v", "P"]
    colors = [f"C{i}" for i in range(len(backends))]
    backend_marker = {backend: markers[i % len(markers)] for i, backend in enumerate(backends)}
    backend_color = {backend: colors[i] for i, backend in enumerate(backends)}

    seen: set[str] = set()
    for row in rows:
        label = row.backend if row.backend not in seen else None
        seen.add(row.backend)
        ax.scatter(
            row.mean_microseconds_per_shot,
            row.mean_logical_error_rate,
            s=90,
            marker=backend_marker[row.backend],
            color=backend_color[row.backend],
            label=label,
        )
        ax.annotate(
            row.decoder,
            (row.mean_microseconds_per_shot, row.mean_logical_error_rate),
            textcoords="offset points",
            xytext=(6, 4),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean runtime (microseconds / shot) -- compare within a backend only")
    ax.set_ylabel("Mean logical error rate (comparable across all)")
    ax.set_title("Decoder accuracy vs runtime, grouped by implementation backend")
    ax.legend(title="backend", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_accuracy_tier(result: BenchmarkResult, *, ax: Axes | None = None) -> Axes:
    """Bar chart of mean logical error rate per decoder (accuracy tier).

    Every decoder sees the same syndromes, so this ranking measures decoding
    quality alone, independent of implementation language.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    rows = sorted(build_leaderboard(result), key=lambda r: r.mean_logical_error_rate)
    names = [row.decoder for row in rows]
    values = [row.mean_logical_error_rate for row in rows]
    ax.bar(names, values)
    ax.set_yscale("log")
    ax.set_ylabel("Mean logical error rate (lower is better)")
    ax.set_xlabel("Decoder")
    ax.set_title("Accuracy tier: logical error rate, comparable across all decoders")
    ax.grid(True, axis="y", which="both", alpha=0.3)
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
