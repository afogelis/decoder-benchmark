from decbench.leaderboard import build_leaderboard
from decbench.runner import run_benchmark
from decbench.types import BenchmarkConfig


def test_benchmark_produces_one_record_per_combination():
    config = BenchmarkConfig(
        decoders=["mwpm", "union_find"],
        distances=[3],
        error_rates=[0.005, 0.01],
        shots=300,
        seed=1,
    )
    result = run_benchmark(config)
    assert len(result.records) == 2 * 1 * 2
    for record in result.records:
        assert 0.0 <= record.logical_error_rate <= 1.0
        assert record.wall_seconds >= 0.0
        assert record.peak_kib >= 0.0


def test_leaderboard_ranks_by_accuracy():
    config = BenchmarkConfig(
        decoders=["mwpm", "union_find", "bp"],
        distances=[3],
        error_rates=[0.005],
        shots=400,
        seed=2,
    )
    rows = build_leaderboard(run_benchmark(config))
    assert [row.decoder for row in rows]  # non-empty
    lers = [row.mean_logical_error_rate for row in rows]
    assert lers == sorted(lers)
