"""Convert a Stim detector error model into parity-check matrices.

A graphlike or hypergraph detector error model (DEM) is, mathematically, a set
of independent error mechanisms. Each mechanism flips a subset of detectors and
a subset of logical observables with some probability. Stacking these as
columns gives:

* ``H`` (detectors x mechanisms) -- the check matrix; ``H @ e (mod 2)`` is the
  syndrome produced by error vector ``e``.
* ``L`` (observables x mechanisms) -- the logical action; ``L @ e (mod 2)`` is
  the true observable flip we must predict.
* ``priors`` (mechanisms,) -- the independent probability of each mechanism.

These three objects are all that a belief-propagation or maximum-likelihood
decoder needs, independent of the underlying circuit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


@dataclass(frozen=True)
class DemMatrices:
    """Parity-check representation of a detector error model."""

    check_matrix: np.ndarray  # shape (num_detectors, num_mechanisms), uint8
    observable_matrix: np.ndarray  # shape (num_observables, num_mechanisms), uint8
    priors: np.ndarray  # shape (num_mechanisms,), float

    @property
    def num_detectors(self) -> int:
        return int(self.check_matrix.shape[0])

    @property
    def num_mechanisms(self) -> int:
        return int(self.check_matrix.shape[1])

    @property
    def num_observables(self) -> int:
        return int(self.observable_matrix.shape[0])


def dem_to_matrices(dem: stim.DetectorErrorModel) -> DemMatrices:
    """Flatten ``dem`` into :class:`DemMatrices`.

    The DEM should be built with ``decompose_errors=False`` so each ``error``
    instruction is a single mechanism (separators are tolerated and ignored,
    which merges decomposed components back into one column).
    """
    flat = dem.flattened()
    det_columns: list[list[int]] = []
    obs_columns: list[list[int]] = []
    priors: list[float] = []

    for instruction in flat:
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        detectors: list[int] = []
        observables: list[int] = []
        for target in instruction.targets_copy():
            if target.is_separator():
                continue
            if target.is_relative_detector_id():
                detectors.append(target.val)
            elif target.is_logical_observable_id():
                observables.append(target.val)
        det_columns.append(detectors)
        obs_columns.append(observables)
        priors.append(probability)

    num_mechanisms = len(priors)
    check = np.zeros((flat.num_detectors, num_mechanisms), dtype=np.uint8)
    observable = np.zeros((flat.num_observables, num_mechanisms), dtype=np.uint8)
    for column, (detectors, observables) in enumerate(zip(det_columns, obs_columns)):
        for detector in detectors:
            check[detector, column] = 1
        for observable_id in observables:
            observable[observable_id, column] = 1

    return DemMatrices(
        check_matrix=check,
        observable_matrix=observable,
        priors=np.asarray(priors, dtype=float),
    )
