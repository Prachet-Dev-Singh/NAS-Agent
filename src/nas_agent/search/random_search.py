"""
random_search.py
----------------
Arm A: Uniform Random Search Proposer baseline.

Behavior:
- Emits random macro proposals sampled uniformly from the valid search space.
- If existing feasible parents exist in the store, occasionally explores micro surgery operators.
"""

from __future__ import annotations

import random
from typing import Optional

from nas_agent.calibration.budget import sample_random_macro_proposal
from nas_agent.config import BudgetConfig
from nas_agent.memory.queries import get_near_feasible_parents, get_pareto_front
from nas_agent.memory.store import CandidateStore
from nas_agent.search.operators import MICRO_OPERATORS, apply_named_operator
from nas_agent.state import MacroProposal, Proposal


class RandomSearchProposer:
    """Proposes candidates uniformly at random (Arm A)."""

    def __init__(self, seed: int = 42, micro_prob: float = 0.25) -> None:
        self.rng = random.Random(seed)
        self.micro_prob = micro_prob

    def propose(
        self,
        store: CandidateStore,
        run_id: str,
        caps: BudgetConfig,
        min_val_acc: float,
    ) -> Proposal:
        """Sample next proposal uniformly at random."""
        # 1. Decide between macro from scratch or micro operator
        if self.rng.random() < self.micro_prob:
            parents = get_near_feasible_parents(store, run_id, caps, min_val_acc)
            if parents:
                parent = self.rng.choice(parents)
                if parent.spec_json:
                    import json
                    parent_spec = MacroProposal.model_validate(json.loads(parent.spec_json))
                    op_name = self.rng.choice(MICRO_OPERATORS)
                    return apply_named_operator(op_name, parent_spec, parent.candidate_id, self.rng)

        # Default: sample random valid macro proposal from search space
        return sample_random_macro_proposal(self.rng)
