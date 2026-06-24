"""Command-line interface for the decoder benchmark.

Examples
--------
    decbench list
    decbench run --decoders mwpm,union_find,bp --distances 3,5 --p 0.005,0.01 --shots 5000
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .leaderboard import build_leaderboard, format_leaderboard
from .registry import available_decoders
from .runner import run_benchmark
from .types import BenchmarkConfig


def _ints(raw: str) -> list[int]:
    return [int(token) for token in raw.split(",") if token.strip()]


def _floats(raw: str) -> list[float]:
    return [float(token) for token in raw.split(",") if token.strip()]


def _names(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _cmd_list(_: argparse.Namespace) -> int:
    print("\n".join(available_decoders()))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = BenchmarkConfig(
        decoders=_names(args.decoders),
        distances=_ints(args.distances),
        error_rates=_floats(args.p),
        rounds=args.rounds,
        shots=args.shots,
        basis=args.basis,
        seed=args.seed,
    )
    result = run_benchmark(config)
    print(format_leaderboard(build_leaderboard(result)))
    print()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(json.loads(result.model_dump_json()), handle, indent=2)
        print(f"wrote {len(result.records)} records to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="decbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="List registered decoders.")
    listing.set_defaults(func=_cmd_list)

    run = sub.add_parser("run", help="Run a benchmark sweep.")
    run.add_argument("--decoders", type=str, default="mwpm,union_find,bp")
    run.add_argument("--distances", type=str, required=True, help="Comma-separated, e.g. 3,5")
    run.add_argument("--p", type=str, required=True, help="Comma-separated physical rates.")
    run.add_argument("--rounds", type=int, default=None)
    run.add_argument("--shots", type=int, default=20_000)
    run.add_argument("--basis", choices=["X", "Z"], default="Z")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
