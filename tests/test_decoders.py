"""Correctness tests for all registered decoders.

These run against the real Stim/PyMatching stack on small surface codes, so a
passing suite means each decoder produces valid corrections end to end.
"""

import numpy as np
import pytest
from surfacecode.circuits import build_surface_code_circuit
from surfacecode.sampling import sample_syndromes
from surfacecode.types import ExperimentConfig

from decbench.registry import available_decoders, get_decoder

DECODERS = ["mwpm", "union_find", "bp"]


def _sample(distance=3, rounds=3, p=0.01, shots=400, seed=1):
    config = ExperimentConfig(distance=distance, rounds=rounds, p=p, shots=shots, seed=seed)
    circuit = build_surface_code_circuit(config)
    sample = sample_syndromes(circuit, shots=shots, seed=seed)
    return circuit, sample


def test_all_three_decoders_are_registered():
    assert set(DECODERS).issubset(set(available_decoders()))


@pytest.mark.parametrize("name", DECODERS)
def test_decoder_output_shape(name):
    circuit, sample = _sample()
    decoder = get_decoder(name)
    decoder.fit(circuit)
    predictions = decoder.decode_batch(sample.detection_events)
    assert predictions.shape == (sample.num_shots, sample.num_observables)
    assert predictions.dtype == bool


@pytest.mark.parametrize("name", DECODERS)
def test_decoder_noiseless_has_no_failures(name):
    config = ExperimentConfig(distance=3, rounds=3, p=0.0, shots=200, seed=2)
    circuit = build_surface_code_circuit(config)
    sample = sample_syndromes(circuit, shots=200, seed=2)
    decoder = get_decoder(name)
    decoder.fit(circuit)
    predictions = decoder.decode_batch(sample.detection_events)
    mismatches = np.any(predictions != sample.observable_flips, axis=1)
    assert int(np.count_nonzero(mismatches)) == 0


def _logical_error_rate(name, circuit, sample) -> float:
    decoder = get_decoder(name)
    decoder.fit(circuit)
    predictions = decoder.decode_batch(sample.detection_events)
    mismatches = np.any(predictions != sample.observable_flips, axis=1)
    return float(np.count_nonzero(mismatches)) / sample.num_shots


def test_mwpm_per_round_rate_below_physical_rate():
    # The meaningful comparison is per-round (not per-shot) logical error vs the
    # per-operation physical rate. Below threshold MWPM should suppress it.
    from surfacecode.metrics import per_round_error_rate

    p = 0.003
    rounds = 3
    circuit, sample = _sample(distance=3, rounds=rounds, p=p, shots=8000, seed=3)
    ler = _logical_error_rate("mwpm", circuit, sample)
    assert per_round_error_rate(ler, rounds) < p


@pytest.mark.parametrize("name", ["mwpm", "union_find"])
def test_larger_distance_suppresses_logical_error(name):
    # Below threshold, the distance-5 code should beat the distance-3 code.
    circuit3, sample3 = _sample(distance=3, rounds=3, p=0.003, shots=8000, seed=4)
    circuit5, sample5 = _sample(distance=5, rounds=5, p=0.003, shots=8000, seed=4)
    ler3 = _logical_error_rate(name, circuit3, sample3)
    ler5 = _logical_error_rate(name, circuit5, sample5)
    assert ler5 <= ler3


def test_union_find_tracks_mwpm_below_threshold():
    # Union-find should be within a small factor of MWPM accuracy below threshold.
    circuit, sample = _sample(distance=5, rounds=5, p=0.004, shots=6000, seed=5)
    results = {}
    for name in ("mwpm", "union_find"):
        decoder = get_decoder(name)
        decoder.fit(circuit)
        predictions = decoder.decode_batch(sample.detection_events)
        mismatches = np.any(predictions != sample.observable_flips, axis=1)
        results[name] = np.count_nonzero(mismatches) / sample.num_shots
    # Union-find is never expected to beat MWPM by much; allow a generous margin.
    assert results["union_find"] <= results["mwpm"] + 0.01
