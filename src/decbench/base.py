"""Decoder interface and profiling helpers.

Every decoder implements the same small contract so the runner can treat them
uniformly -- much like a surveillance pipeline scoring many models against the
same event stream. A decoder is *fit* once to a circuit's detector error model,
then *decodes* batches of syndromes into predicted logical observable flips.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import stim


@runtime_checkable
class Decoder(Protocol):
    """A surface-code decoder.

    Implementations must set ``name`` and support fitting to a circuit followed
    by batched decoding.
    """

    name: str

    def fit(self, circuit: stim.Circuit) -> None:
        """Prepare the decoder for a specific circuit (build graphs/matrices)."""
        ...

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """Decode ``(shots, num_detectors)`` syndromes into ``(shots, num_observables)`` flips."""
        ...


@dataclass(frozen=True)
class DecoderProfile:
    """Runtime and memory profile of a single ``decode_batch`` call."""

    wall_seconds: float
    peak_kib: float
    shots: int

    @property
    def microseconds_per_shot(self) -> float:
        return 1e6 * self.wall_seconds / self.shots if self.shots else float("nan")


def profile_decode(
    decoder: Decoder, detection_events: np.ndarray
) -> tuple[np.ndarray, DecoderProfile]:
    """Time and memory-profile a decode, returning ``(predictions, profile)``.

    Peak memory is measured with :mod:`tracemalloc` so it reflects Python-level
    allocations attributable to the decode call rather than whole-process RSS.
    """
    shots = int(detection_events.shape[0])
    tracemalloc.start()
    start = time.perf_counter()
    predictions = decoder.decode_batch(detection_events)
    wall_seconds = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    profile = DecoderProfile(wall_seconds=wall_seconds, peak_kib=peak_bytes / 1024.0, shots=shots)
    return np.asarray(predictions, dtype=bool), profile
