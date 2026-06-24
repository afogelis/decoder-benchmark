"""Minimum-weight perfect matching decoder (PyMatching backend).

MWPM is the community-standard surface-code decoder and serves as the accuracy
reference all other decoders are measured against. PyMatching implements a
heavily optimized blossom algorithm in C++, so it is also a strong runtime
baseline.
"""

from __future__ import annotations

import numpy as np
import pymatching
import stim


class MwpmDecoder:
    """Wraps :class:`pymatching.Matching` behind the :class:`Decoder` protocol."""

    name = "mwpm"

    def __init__(self) -> None:
        self._matching: pymatching.Matching | None = None

    def fit(self, circuit: stim.Circuit) -> None:
        dem = circuit.detector_error_model(decompose_errors=True)
        self._matching = pymatching.Matching.from_detector_error_model(dem)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        if self._matching is None:
            raise RuntimeError("decoder must be fit() before decode_batch()")
        return np.asarray(self._matching.decode_batch(detection_events), dtype=bool)
