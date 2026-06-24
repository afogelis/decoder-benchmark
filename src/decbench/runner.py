"""Benchmark runner: sweep decoders across distances and error rates.

For each (distance, physical error rate) point we build one noisy surface-code
circuit with the companion ``surface-code-simulator`` package, sample a fixed
batch of syndromes once, and then score every decoder on that *same* batch.
Sharing the syndrome batch makes accuracy differences between decoders directly
comparable and keeps sampling cost out of the per-decoder runtime numbers.
"""

from __future__ import annotations

import numpy as np
from surfacecode.circuits import build_surface_code_circuit
from surfacecode.metrics import wilson_interval
from surfacecode.sampling import sample_syndromes
from surfacecode.types import ExperimentConfig

from .base import profile_decode
from .registry import get_decoder
from .types import BenchmarkConfig, BenchmarkResult, RunRecord


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Execute the sweep described by ``config`` and return all run records."""
    records: list[RunRecord] = []
    for distance in config.distances:
        rounds = config.rounds if config.rounds is not None else distance
        for p in config.error_rates:
            experiment = ExperimentConfig(
                distance=distance,
                rounds=rounds,
                p=p,
                shots=config.shots,
                basis=config.basis,
                seed=config.seed,
            )
            circuit = build_surface_code_circuit(experiment)
            sample = sample_syndromes(circuit, shots=config.shots, seed=config.seed)

            for decoder_name in config.decoders:
                record = _benchmark_one(decoder_name, circuit, sample, distance, p, rounds)
                records.append(record)
    return BenchmarkResult(records=records)


def _benchmark_one(decoder_name, circuit, sample, distance, p, rounds) -> RunRecord:
    decoder = get_decoder(decoder_name)
    decoder.fit(circuit)
    predictions, profile = profile_decode(decoder, sample.detection_events)

    mismatches = np.any(predictions != sample.observable_flips, axis=1)
    num_failures = int(np.count_nonzero(mismatches))
    estimate = wilson_interval(num_failures, sample.num_shots)

    return RunRecord(
        decoder=decoder_name,
        distance=distance,
        p=p,
        rounds=rounds,
        shots=sample.num_shots,
        num_failures=num_failures,
        logical_error_rate=estimate.logical_error_rate,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        wall_seconds=profile.wall_seconds,
        microseconds_per_shot=profile.microseconds_per_shot,
        peak_kib=profile.peak_kib,
    )
