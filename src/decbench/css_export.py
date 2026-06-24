"""Load qLDPC matrix exports from ``qldpc-builder`` and benchmark decoders on them.

The export bundle written by ``qldpc export --stim`` contains parity-check
matrices, metadata, and a simplified Stim syndrome-extraction circuit. This
module loads that bundle and runs the same decoder interface used for surface
codes, linking repo 8 (construction) to repo 2 (benchmarking).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

from .base import profile_decode
from surfacecode.metrics import wilson_interval
from .registry import get_decoder


@dataclass(frozen=True)
class CssExport:
    """Parity-check export bundle from ``qldpc-builder``."""

    name: str
    metadata: dict
    check_x: np.ndarray
    check_z: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray
    stim_path: Path | None


@dataclass(frozen=True)
class CssBenchmarkRecord:
    decoder: str
    code: str
    shots: int
    num_failures: int
    logical_error_rate: float
    ci_low: float
    ci_high: float
    wall_seconds: float
    microseconds_per_shot: float


def load_css_export(path: Path | str) -> CssExport:
    """Load an export directory produced by ``qldpc export``."""
    root = Path(path)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing metadata.json in {root}")

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    stim_path = root / "syndrome.stim"
    return CssExport(
        name=str(metadata.get("name", root.name)),
        metadata=metadata,
        check_x=np.load(root / "Hx.npy"),
        check_z=np.load(root / "Hz.npy"),
        logical_x=np.load(root / "logical_x.npy"),
        logical_z=np.load(root / "logical_z.npy"),
        stim_path=stim_path if stim_path.is_file() else None,
    )


def benchmark_css_export(
    path: Path | str,
    *,
    decoders: list[str] | None = None,
    shots: int = 2000,
    seed: int = 2026,
) -> list[CssBenchmarkRecord]:
    """Benchmark decoders on a Stim export bundle."""
    export = load_css_export(path)
    if export.stim_path is None:
        raise FileNotFoundError(
            f"{path} has no syndrome.stim; re-export with `qldpc export <code> --stim`"
        )

    circuit = stim.Circuit.from_file(str(export.stim_path))
    sampler = circuit.compile_detector_sampler(seed=seed)
    events, observables = sampler.sample(shots, separate_observables=True)
    events = events.astype(np.uint8)
    observables = observables.astype(bool)

    chosen = decoders or ["bp"]
    records: list[CssBenchmarkRecord] = []
    for decoder_name in chosen:
        decoder = get_decoder(decoder_name)
        decoder.fit(circuit)
        predictions, profile = profile_decode(decoder, events)
        mismatches = np.any(predictions != observables, axis=1)
        num_failures = int(np.count_nonzero(mismatches))
        estimate = wilson_interval(num_failures, shots)
        records.append(
            CssBenchmarkRecord(
                decoder=decoder_name,
                code=export.name,
                shots=shots,
                num_failures=num_failures,
                logical_error_rate=estimate.logical_error_rate,
                ci_low=estimate.ci_low,
                ci_high=estimate.ci_high,
                wall_seconds=profile.wall_seconds,
                microseconds_per_shot=profile.microseconds_per_shot,
            )
        )
    return records
