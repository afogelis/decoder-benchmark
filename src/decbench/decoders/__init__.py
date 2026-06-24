"""Decoder implementations and their registration."""

from ..registry import register_decoder
from .bp import BeliefPropagationDecoder
from .bposd import BpOsdDecoder, ldpc_is_available
from .mwpm import MwpmDecoder
from .union_find import UnionFindDecoder

register_decoder("mwpm", MwpmDecoder)
register_decoder("union_find", UnionFindDecoder)
register_decoder("bp", BeliefPropagationDecoder)

# 'bposd' is an optional reference decoder: only register it when its optional
# 'ldpc' dependency is importable, so the core benchmark works everywhere.
if ldpc_is_available():
    register_decoder("bposd", BpOsdDecoder)

__all__ = [
    "BeliefPropagationDecoder",
    "BpOsdDecoder",
    "MwpmDecoder",
    "UnionFindDecoder",
    "ldpc_is_available",
]
