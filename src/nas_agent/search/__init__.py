"""
search package
--------------
Search operators and baseline proposers (Arms A, B, and G).
"""

from nas_agent.search.bandit import ContextualBanditProposer
from nas_agent.search.evolution import RegularizedEvolutionProposer
from nas_agent.search.operators import (
    MACRO_OPERATORS,
    MICRO_OPERATORS,
    OPERATOR_NAMES,
    apply_named_operator,
    mutate_macro_block,
    mutate_macro_depth,
    mutate_macro_expand,
    mutate_macro_width,
)
from nas_agent.search.random_search import RandomSearchProposer

__all__ = [
    "ContextualBanditProposer",
    "RegularizedEvolutionProposer",
    "RandomSearchProposer",
    "OPERATOR_NAMES",
    "MACRO_OPERATORS",
    "MICRO_OPERATORS",
    "apply_named_operator",
    "mutate_macro_block",
    "mutate_macro_depth",
    "mutate_macro_expand",
    "mutate_macro_width",
]
