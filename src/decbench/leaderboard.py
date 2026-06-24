"""Aggregate benchmark records into ranked leaderboards.

Accuracy and runtime are reported as two *separate* tiers. Accuracy (logical
error rate) is a property of the algorithm and is directly comparable across
every decoder, because they all decode the same syndrome batch. Runtime is a
property of the *implementation*: comparing a pure-Python decoder against a
compiled C++ library measures the language, not the algorithm, so runtime is
only compared within a backend tier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import BenchmarkResult

#: Implementation backend for each decoder. Runtime is only meaningfully
#: comparable *within* a backend; see the module docstring.
DECODER_BACKEND: dict[str, str] = {
    "mwpm": "compiled (PyMatching, C++)",
    "bposd": "compiled (ldpc, C++)",
    "union_find": "pure Python (educational)",
    "bp": "pure Python (educational)",
}


def backend_of(decoder: str) -> str:
    """Return the implementation backend tier for ``decoder``."""
    return DECODER_BACKEND.get(decoder, "unknown")


@dataclass(frozen=True)
class LeaderboardRow:
    """Per-decoder summary aggregated across all swept points."""

    decoder: str
    mean_logical_error_rate: float
    mean_microseconds_per_shot: float
    mean_peak_kib: float
    num_points: int
    backend: str = "unknown"


def build_leaderboard(result: BenchmarkResult) -> list[LeaderboardRow]:
    """Average accuracy/runtime/memory per decoder and sort by accuracy.

    Lower mean logical error rate ranks first; ties break on speed. This mirrors
    how a model leaderboard surfaces the best accuracy/cost trade-off rather
    than a single metric.
    """
    by_decoder: dict[str, list] = {}
    for record in result.records:
        by_decoder.setdefault(record.decoder, []).append(record)

    rows = [
        LeaderboardRow(
            decoder=name,
            mean_logical_error_rate=float(np.mean([r.logical_error_rate for r in records])),
            mean_microseconds_per_shot=float(np.mean([r.microseconds_per_shot for r in records])),
            mean_peak_kib=float(np.mean([r.peak_kib for r in records])),
            num_points=len(records),
            backend=backend_of(name),
        )
        for name, records in by_decoder.items()
    ]
    rows.sort(key=lambda row: (row.mean_logical_error_rate, row.mean_microseconds_per_shot))
    return rows


def format_leaderboard(rows: list[LeaderboardRow]) -> str:
    """Render the accuracy leaderboard (kept for backwards compatibility)."""
    return format_accuracy_tier(rows)


def format_accuracy_tier(rows: list[LeaderboardRow]) -> str:
    """Accuracy tier: rank every decoder by mean logical error rate.

    This is the scientifically meaningful comparison -- all decoders see the
    same syndromes, so the logical error rate measures decoding quality alone.
    """
    ordered = sorted(rows, key=lambda row: row.mean_logical_error_rate)
    header = f"{'decoder':<14}{'mean LER':>14}{'points':>9}  {'backend':<28}"
    lines = [
        "ACCURACY TIER (lower is better; comparable across all decoders)",
        header,
        "-" * len(header),
    ]
    for row in ordered:
        lines.append(
            f"{row.decoder:<14}{row.mean_logical_error_rate:>14.4e}{row.num_points:>9}  {row.backend:<28}"
        )
    return "\n".join(lines)


def format_runtime_tier(rows: list[LeaderboardRow]) -> str:
    """Runtime tier: report speed/memory grouped by implementation backend.

    Runtimes are NOT comparable across backends (a compiled C++ decoder will
    always beat a pure-Python one regardless of algorithm), so rows are grouped
    by backend and only ranked within each group.
    """
    by_backend: dict[str, list[LeaderboardRow]] = {}
    for row in rows:
        by_backend.setdefault(row.backend, []).append(row)

    header = f"{'decoder':<14}{'us/shot':>14}{'peak KiB':>14}"
    lines = ["RUNTIME TIER (compare only within a backend; cross-language is not meaningful)"]
    for backend in sorted(by_backend):
        lines.append(f"\n[{backend}]")
        lines.append(header)
        lines.append("-" * len(header))
        for row in sorted(by_backend[backend], key=lambda r: r.mean_microseconds_per_shot):
            lines.append(
                f"{row.decoder:<14}{row.mean_microseconds_per_shot:>14.2f}{row.mean_peak_kib:>14.1f}"
            )
    return "\n".join(lines)
