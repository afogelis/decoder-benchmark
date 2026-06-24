"""Decoder implementations and their registration."""

from ..registry import register_decoder
from .bp import BeliefPropagationDecoder
from .mwpm import MwpmDecoder
from .union_find import UnionFindDecoder

register_decoder("mwpm", MwpmDecoder)
register_decoder("union_find", UnionFindDecoder)
register_decoder("bp", BeliefPropagationDecoder)

__all__ = [
    "BeliefPropagationDecoder",
    "MwpmDecoder",
    "UnionFindDecoder",
]
