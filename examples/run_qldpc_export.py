"""Benchmark decoders on a qldpc-builder export bundle.

First export a code from qldpc-builder::

    pip install -e ../qldpc-builder[stim]
    qldpc export bb72 --output artifacts/bb72 --stim

Then run this script from decoder-benchmark::

    python examples/run_qldpc_export.py ../qldpc-builder/artifacts/bb72
"""

from __future__ import annotations

import sys

from decbench.css_export import benchmark_css_export, load_css_export
from decbench.decoders import ldpc_is_available


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python examples/run_qldpc_export.py <export-directory>")
        raise SystemExit(2)

    export_path = sys.argv[1]
    export = load_css_export(export_path)
    print(f"loaded {export.name}: n={export.metadata['n']}, k={export.metadata['k']}")
    if export.metadata.get("d_exact") is not None:
        print(f"  d_exact={export.metadata['d_exact']}")
    elif export.metadata.get("d_literature") is not None:
        print(f"  d_literature={export.metadata['d_literature']}")

    decoders = ["bp"]
    if ldpc_is_available():
        decoders.append("bposd")

    records = benchmark_css_export(export_path, decoders=decoders, shots=1500, seed=2026)
    for record in records:
        print(
            f"{record.decoder:8s}  LER={record.logical_error_rate:.4e} "
            f"[{record.ci_low:.4e}, {record.ci_high:.4e}]  "
            f"{record.microseconds_per_shot:.1f} us/shot"
        )


if __name__ == "__main__":
    main()
