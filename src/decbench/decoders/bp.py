"""Belief-propagation decoder (from-scratch log-domain sum-product).

Belief propagation treats the detector error model as a Tanner graph and passes
log-likelihood-ratio messages between error-mechanism variables and detector
checks. It is fast and parallelisable, but on surface codes its accuracy is
limited by the many short cycles and degenerate errors in the graph: BP alone
frequently fails to converge and is, in practice, dominated by matching-based
decoders. We implement plain BP (no ordered-statistics post-processing) so the
benchmark can surface exactly this well-documented limitation rather than hide
it behind a library.
"""

from __future__ import annotations

import numpy as np
import stim

from ..dem_matrices import dem_to_matrices

_TANH_CLIP = 1.0 - 1e-12


class BeliefPropagationDecoder:
    """Sum-product BP over the (undecomposed) detector error model."""

    name = "bp"

    def __init__(self, *, max_iterations: int = 40) -> None:
        self.max_iterations = max_iterations
        self._check_of_edge: np.ndarray | None = None
        self._var_of_edge: np.ndarray | None = None
        self._var_edges: list[np.ndarray] = []
        self._prior_llr: np.ndarray | None = None
        self._sign: np.ndarray | None = None  # per-check, depends on syndrome at decode time
        self._check_matrix: np.ndarray | None = None
        self._observable_matrix: np.ndarray | None = None
        self._num_checks = 0
        self._num_vars = 0

    def fit(self, circuit: stim.Circuit) -> None:
        dem = circuit.detector_error_model(decompose_errors=False)
        matrices = dem_to_matrices(dem)
        self._check_matrix = matrices.check_matrix
        self._observable_matrix = matrices.observable_matrix
        self._num_checks = matrices.num_detectors
        self._num_vars = matrices.num_mechanisms

        priors = np.clip(matrices.priors, 1e-15, 0.5 - 1e-15)
        self._prior_llr = np.log((1.0 - priors) / priors)

        checks, variables = np.nonzero(matrices.check_matrix)
        self._check_of_edge = checks.astype(np.int64)
        self._var_of_edge = variables.astype(np.int64)
        self._var_edges = [
            np.nonzero(self._var_of_edge == var)[0] for var in range(self._num_vars)
        ]

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        if self._check_matrix is None or self._observable_matrix is None:
            raise RuntimeError("decoder must be fit() before decode_batch()")
        events = np.asarray(detection_events, dtype=np.uint8)
        num_observables = self._observable_matrix.shape[0]
        predictions = np.zeros((events.shape[0], num_observables), dtype=bool)
        for shot in range(events.shape[0]):
            error = self._run_bp(events[shot])
            flips = (self._observable_matrix @ error) & 1
            predictions[shot] = flips.astype(bool)
        return predictions

    def _run_bp(self, syndrome: np.ndarray) -> np.ndarray:
        assert self._prior_llr is not None and self._check_of_edge is not None
        assert self._var_of_edge is not None and self._check_matrix is not None

        check_of_edge = self._check_of_edge
        var_of_edge = self._var_of_edge
        num_edges = check_of_edge.shape[0]
        sign = np.where(syndrome[check_of_edge] == 1, -1.0, 1.0)

        msg_v2c = self._prior_llr[var_of_edge].copy()
        msg_c2v = np.zeros(num_edges, dtype=float)
        best_error = np.zeros(self._num_vars, dtype=np.uint8)

        for _ in range(self.max_iterations):
            # Check update: tanh product rule with leave-one-out exclusion.
            tanh_half = np.clip(np.tanh(0.5 * msg_v2c), -_TANH_CLIP, _TANH_CLIP)
            tanh_half = np.where(tanh_half == 0.0, 1e-12, tanh_half)
            product_per_check = np.ones(self._num_checks, dtype=float)
            np.multiply.at(product_per_check, check_of_edge, tanh_half)
            excluded = product_per_check[check_of_edge] / tanh_half
            excluded = np.clip(excluded, -_TANH_CLIP, _TANH_CLIP)
            msg_c2v = sign * 2.0 * np.arctanh(excluded)

            # Variable update: total belief minus the incoming message.
            total = self._prior_llr.copy()
            np.add.at(total, var_of_edge, msg_c2v)
            msg_v2c = total[var_of_edge] - msg_c2v

            error = (total < 0.0).astype(np.uint8)
            implied = (self._check_matrix @ error) & 1
            best_error = error
            if np.array_equal(implied, syndrome):
                break

        return best_error
