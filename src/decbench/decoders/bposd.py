"""Belief-propagation + ordered-statistics decoding (BP-OSD) via ``ldpc``.

Plain belief propagation is dominated on the surface code (see ``bp.py``)
because the Tanner graph is highly degenerate and full of short cycles. The
standard fix in the literature is to follow BP with an ordered-statistics
decoding (OSD) post-processing step that resolves the residual syndrome with a
small Gaussian-elimination search. This is the decoder that is actually
competitive with matching, and it is the workhorse decoder for quantum LDPC
codes.

Rather than reimplement OSD by hand, this wrapper uses Joschka Roffe's
optimized ``ldpc`` package as a *reference* BP-OSD decoder. It is an optional
dependency (``pip install 'decoder-benchmark[optimized]'``); the decoder is
only registered when ``ldpc`` is importable, so the core benchmark still works
on interpreters where ``ldpc`` has no wheel (e.g. very new CPython versions).
"""

from __future__ import annotations

import numpy as np
import stim

from ..dem_matrices import dem_to_matrices


def ldpc_is_available() -> bool:
    """Return True if the optional ``ldpc`` dependency can be imported."""
    try:
        import ldpc  # noqa: F401
    except Exception:
        return False
    return True


def _make_bposd(check_matrix: np.ndarray, priors: np.ndarray):
    """Construct an ``ldpc`` BP-OSD decoder, supporting both v1 and v2 APIs."""
    channel = [float(p) for p in priors]

    # ldpc v2: ldpc.BpOsdDecoder(pcm, error_channel=..., osd_method=...).
    try:
        from ldpc import BpOsdDecoder

        return BpOsdDecoder(
            check_matrix.astype(np.uint8),
            error_channel=channel,
            max_iter=0,  # 0 -> library default (block length)
            bp_method="product_sum",
            osd_method="osd_cs",
            osd_order=4,
        )
    except Exception:
        pass

    # ldpc v1: ldpc.bposd_decoder(H, channel_probs=..., osd_method=...).
    from ldpc import bposd_decoder

    return bposd_decoder(
        check_matrix.astype(np.uint8),
        channel_probs=channel,
        max_iter=0,
        bp_method="ms",
        osd_method="osd_cs",
        osd_order=4,
    )


class BpOsdDecoder:
    """BP-OSD reference decoder backed by the ``ldpc`` package."""

    name = "bposd"

    def __init__(self) -> None:
        if not ldpc_is_available():
            raise RuntimeError(
                "the 'bposd' decoder requires the optional 'ldpc' package; "
                "install with: pip install 'decoder-benchmark[optimized]'"
            )
        self._decoder = None
        self._observable_matrix: np.ndarray | None = None

    def fit(self, circuit: stim.Circuit) -> None:
        dem = circuit.detector_error_model(decompose_errors=False)
        matrices = dem_to_matrices(dem)
        self._observable_matrix = matrices.observable_matrix
        self._decoder = _make_bposd(matrices.check_matrix, matrices.priors)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        if self._decoder is None or self._observable_matrix is None:
            raise RuntimeError("decoder must be fit() before decode_batch()")
        events = np.asarray(detection_events, dtype=np.uint8)
        num_observables = self._observable_matrix.shape[0]
        predictions = np.zeros((events.shape[0], num_observables), dtype=bool)
        for shot in range(events.shape[0]):
            error = np.asarray(self._decoder.decode(events[shot]), dtype=np.uint8)
            flips = (self._observable_matrix @ error) & 1
            predictions[shot] = flips.astype(bool)
        return predictions
