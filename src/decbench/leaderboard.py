"""Aggregate benchmark records into a ranked leaderboard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import BenchmarkResult


@dataclass(frozen=True)
class LeaderboardRow:
    """Per-decoder summary aggregated across all swept points."""

    decoder: str
    mean_logical_error_rate: float
    mean_microseconds_per_shot: float
    mean_peak_kib: float
    num_points: int


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
        )
        for name, records in by_decoder.items()
    ]
    rows.sort(key=lambda row: (row.mean_logical_error_rate, row.mean_microseconds_per_shot))
    return rows


def format_leaderboard(rows: list[LeaderboardRow]) -> str:
    """Render a fixed-width text table of the leaderboard."""
    header = f"{'decoder':<14}{'mean LER':>14}{'us/shot':>14}{'peak KiB':>14}{'points':>9}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.decoder:<14}{row.mean_logical_error_rate:>14.4e}"
            f"{row.mean_microseconds_per_shot:>14.2f}{row.mean_peak_kib:>14.1f}{row.num_points:>9}"
        )
    return "\n".join(lines)
