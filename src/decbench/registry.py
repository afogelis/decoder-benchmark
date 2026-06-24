"""A tiny registry mapping decoder names to zero-argument factories."""

from __future__ import annotations

from collections.abc import Callable

from .base import Decoder

_FACTORIES: dict[str, Callable[[], Decoder]] = {}


def register_decoder(name: str, factory: Callable[[], Decoder]) -> None:
    """Register ``factory`` under ``name`` (last registration wins)."""
    _FACTORIES[name] = factory


def get_decoder(name: str) -> Decoder:
    """Instantiate a registered decoder by name."""
    if name not in _FACTORIES:
        raise KeyError(f"unknown decoder '{name}'; available: {sorted(_FACTORIES)}")
    return _FACTORIES[name]()


def available_decoders() -> list[str]:
    """Return the sorted names of all registered decoders."""
    return sorted(_FACTORIES)
