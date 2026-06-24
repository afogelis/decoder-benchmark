"""Benchmarking framework for surface-code decoders."""

from . import decoders  # noqa: F401  (registers built-in decoders on import)
from .base import Decoder, DecoderProfile, profile_decode
from .dem_matrices import DemMatrices, dem_to_matrices
from .leaderboard import (
    DECODER_BACKEND,
    LeaderboardRow,
    backend_of,
    build_leaderboard,
    format_accuracy_tier,
    format_leaderboard,
    format_runtime_tier,
)
from .registry import available_decoders, get_decoder, register_decoder
from .runner import run_benchmark
from .types import BenchmarkConfig, BenchmarkResult, RunRecord
from .viz import plot_accuracy_tier, plot_accuracy_vs_p, plot_pareto

__version__ = "0.1.0"

__all__ = [
    "DECODER_BACKEND",
    "BenchmarkConfig",
    "BenchmarkResult",
    "Decoder",
    "DecoderProfile",
    "DemMatrices",
    "LeaderboardRow",
    "RunRecord",
    "available_decoders",
    "backend_of",
    "build_leaderboard",
    "dem_to_matrices",
    "format_accuracy_tier",
    "format_leaderboard",
    "format_runtime_tier",
    "get_decoder",
    "plot_accuracy_tier",
    "plot_accuracy_vs_p",
    "plot_pareto",
    "profile_decode",
    "register_decoder",
    "run_benchmark",
]
