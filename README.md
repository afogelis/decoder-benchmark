# Decoder Benchmark

A framework for benchmarking surface-code decoders on a level playing field. Given the same
batch of error syndromes, it scores each decoder on **accuracy** (logical error rate), **runtime**
(microseconds per shot) and **memory** (peak allocation), then ranks them on a leaderboard and a
Pareto frontier.

Think of it as trade-surveillance for quantum errors: many candidate models compete to infer the
most likely underlying event (here, the logical error) from a stream of signals (here, syndromes),
and we measure which model wins under a latency and accuracy budget.

This is repo 2 of a seven-part [QEC research portfolio](https://github.com/afogelis/qec-portfolio) and builds on
[`surface-code-simulator`](https://github.com/afogelis/surface-code-simulator).

## Results at a glance

![Accuracy/runtime Pareto frontier for MWPM, union-find and belief propagation.](docs/pareto.png)

*Accuracy versus runtime. MWPM and union-find define the accuracy frontier; belief propagation is dominated on both axes.*

![Logical error rate versus physical error rate at code distance 5 for each decoder.](docs/accuracy_vs_p_d5.png)

*Logical error rate versus physical error rate at distance 5.*

## Decoders

| Decoder | Implementation | Notes |
|---------|----------------|-------|
| `mwpm` | [PyMatching](https://pymatching.readthedocs.io/) (C++) | Minimum-weight perfect matching; community-standard accuracy reference. |
| `union_find` | From scratch (this repo) | Delfosse-Nickerson cluster growth + spanning-forest peeling on the matching graph. Near-linear time. |
| `bp` | From scratch (this repo) | Log-domain sum-product belief propagation on the detector error model. |

The union-find and belief-propagation decoders are implemented directly (no compiled decoder
libraries) so the benchmark can expose *why* each algorithm wins or loses, not just call a black box.

## What this demonstrates

- **Algorithms:** a correct, self-contained union-find decoder (disjoint-set growth + peeling) and a log-domain BP decoder.
- **Benchmarking discipline:** shared syndrome batches, accuracy/runtime/memory profiling, a ranked leaderboard, and a Pareto trade-off plot.
- **A real research finding:** plain BP is *dominated* on surface codes by matching-based decoders because of graph degeneracy and short cycles, reproducing the consensus in the decoder literature.

## Install

```bash
pip install -e ".[dev]"   # also installs surface-code-simulator from GitHub
```

For local development against a checked-out simulator, install both editable into one environment:

```bash
pip install -e ../surface-code-simulator
pip install -e . --no-deps
```

## Quick start

```bash
pytest
python examples/run_benchmark.py     # writes outputs/{benchmark.json,pareto.png,accuracy_vs_p_d5.png}
```

```bash
decbench list
decbench run --decoders mwpm,union_find,bp --distances 3,5 --p 0.005,0.01 --shots 5000 --output outputs/run.json
```

## Example leaderboard

A representative run (distances 3 and 5; p in {0.005, 0.01}) ranks decoders by mean logical error rate:

```
decoder             mean LER       us/shot      peak KiB   points
-----------------------------------------------------------------
mwpm              4.29e-02          2.13         238.0        4
union_find        5.77e-02       1310.62          39.4        4
bp                9.98e-02       3210.32         403.3        4
```

MWPM and union-find define the accuracy frontier (matching the literature); the pure-Python
union-find and BP are slower than PyMatching's optimised C++, and BP is dominated on both axes.

## Library usage

```python
from decbench import BenchmarkConfig, run_benchmark, build_leaderboard, format_leaderboard

result = run_benchmark(BenchmarkConfig(
    decoders=["mwpm", "union_find", "bp"],
    distances=[3, 5], error_rates=[0.005, 0.01], shots=5_000, seed=2026,
))
print(format_leaderboard(build_leaderboard(result)))
```

### Adding your own decoder

Implement the `Decoder` protocol (`fit(circuit)` + `decode_batch(detection_events)`) and register it:

```python
from decbench import register_decoder
register_decoder("my_decoder", MyDecoder)
```

The companion `ml-qec-decoder` repo registers machine-learning decoders this way.

## Layout

- `src/decbench/decoders/` — `mwpm`, `union_find`, `bp`
- `src/decbench/dem_matrices.py` — detector error model to parity-check matrices
- `src/decbench/{runner,leaderboard,viz,base,registry}.py` — framework
- `tests/` — correctness tests on the real Stim/PyMatching stack
- `examples/` — runnable benchmark

## License

MIT — see [LICENSE](LICENSE).
