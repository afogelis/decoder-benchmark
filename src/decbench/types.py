"""Typed configuration and result records for the benchmark."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Basis = Literal["X", "Z"]


class BenchmarkConfig(BaseModel):
    """Inputs describing a decoder benchmark sweep."""

    model_config = {"frozen": True}

    decoders: list[str] = Field(..., min_length=1)
    distances: list[int] = Field(..., min_length=1)
    error_rates: list[float] = Field(..., min_length=1)
    rounds: int | None = Field(default=None, description="Rounds per shot; defaults to distance.")
    shots: int = Field(default=20_000, ge=1)
    basis: Basis = Field(default="Z")
    seed: int | None = Field(default=None)


class RunRecord(BaseModel):
    """One (decoder, distance, p) measurement."""

    model_config = {"frozen": True}

    decoder: str
    distance: int
    p: float
    rounds: int
    shots: int
    num_failures: int
    logical_error_rate: float
    ci_low: float
    ci_high: float
    wall_seconds: float
    microseconds_per_shot: float
    peak_kib: float


class BenchmarkResult(BaseModel):
    """A full benchmark sweep: many :class:`RunRecord` rows."""

    records: list[RunRecord]

    def for_decoder(self, name: str) -> list[RunRecord]:
        return [record for record in self.records if record.decoder == name]
