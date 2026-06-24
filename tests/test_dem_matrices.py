import numpy as np
from surfacecode.circuits import build_surface_code_circuit
from surfacecode.types import ExperimentConfig

from decbench.dem_matrices import dem_to_matrices


def test_matrices_have_consistent_shapes():
    config = ExperimentConfig(distance=3, rounds=3, p=0.01, shots=1)
    circuit = build_surface_code_circuit(config)
    dem = circuit.detector_error_model(decompose_errors=False)
    matrices = dem_to_matrices(dem)

    assert matrices.check_matrix.shape == (matrices.num_detectors, matrices.num_mechanisms)
    assert matrices.observable_matrix.shape == (matrices.num_observables, matrices.num_mechanisms)
    assert matrices.priors.shape == (matrices.num_mechanisms,)
    assert matrices.num_detectors == circuit.num_detectors
    assert matrices.num_observables == circuit.num_observables


def test_priors_are_valid_probabilities():
    config = ExperimentConfig(distance=3, rounds=3, p=0.02, shots=1)
    circuit = build_surface_code_circuit(config)
    matrices = dem_to_matrices(circuit.detector_error_model(decompose_errors=False))
    assert np.all(matrices.priors > 0.0)
    assert np.all(matrices.priors < 1.0)
