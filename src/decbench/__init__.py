"""Benchmarking framework for surface-code decoders."""

from . import decoders  # noqa: F401  (registers built-in decoders on import)
from .base import Decoder, DecoderProfile, profile_decode
from .dem_matrices import DemMatrices, dem_to_matrices
from .leaderboard import LeaderboardRow, build_leaderboard, format_leaderboard
from .registry import available_decoders, get_decoder, register_decoder
from .runner import run_benchmark
from .types import BenchmarkConfig, BenchmarkResult, RunRecord
from .viz import plot_accuracy_vs_p, plot_pareto

__version__ = "0.1.0"

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "Decoder",
    "DecoderProfile",
    "DemMatrices",
    "LeaderboardRow",
    "RunRecord",
    "available_decoders",
    "build_leaderboard",
    "dem_to_matrices",
    "format_leaderboard",
    "get_decoder",
    "plot_accuracy_vs_p",
    "plot_pareto",
    "profile_decode",
    "register_decoder",
    "run_benchmark",
]
