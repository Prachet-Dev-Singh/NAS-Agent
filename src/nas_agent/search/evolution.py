"""
evolution.py
------------
Arm B: Regularized (Aging) Evolution Proposer baseline (Real et al., 2019).

Behavior:
- Maintains a fixed-size population of evaluated models (population_size = 10).
- Performs tournament selection of size T = 3:
    Samples 3 random models from population, selects the fittest
    (prioritizing feasible accuracy, then lowest constraint violation).
- Mutates the winner using one of the 8 canonical search operators.
- Aging / Regularization: when population exceeds capacity, the oldest model dies.
"""

from __future__ import annotations

import json
import random
from typing import List, Optional

from nas_agent.calibration.budget import sample_random_macro_proposal
from nas_agent.config import BudgetConfig
from nas_agent.memory.queries import compute_violation_ratio
from nas_agent.memory.store import CandidateStore
from nas_agent.search.operators import OPERATOR_NAMES, apply_named_operator
from nas_agent.state import CandidateRecord, MacroProposal, Proposal


class RegularizedEvolutionProposer:
    """Aging evolution search loop (Arm B)."""

    def __init__(
        self,
        population_size: int = 10,
        tournament_size: int = 3,
        seed: int = 42,
    ) -> None:
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.rng = random.Random(seed)

    def _fitness(self, c: CandidateRecord, caps: BudgetConfig) -> float:
        """Scalar fitness: feasible accuracy in [0, 1]; infeasible penalized by -violation."""
        v = compute_violation_ratio(c.params_m, c.macs_g, c.peak_vram_mb, caps)
        if v == 0.0 and c.val_acc is not None:
            return float(c.val_acc)
        # Infeasible fitness is negative violation
        return -v

    def propose(
        self,
        store: CandidateStore,
        run_id: str,
        caps: BudgetConfig,
        min_val_acc: float,
    ) -> Proposal:
        """Select tournament winner from aging population and mutate."""
        all_candidates = store.get_run_candidates(run_id)
        # Keep only evaluated candidates with valid spec_json
        evaluated = [c for c in all_candidates if c.spec_json and c.status in ("ok", "feasible", "infeasible")]

        # Initialization phase: if population is small, emit random macro proposals
        if len(evaluated) < self.population_size:
            return sample_random_macro_proposal(self.rng)

        # Aging population: retain the most recent population_size candidates
        population = evaluated[-self.population_size:]

        # Tournament selection
        sample_size = min(len(population), self.tournament_size)
        tournament = self.rng.sample(population, sample_size)
        winner = max(tournament, key=lambda c: self._fitness(c, caps))

        parent_spec = None
        if winner.spec_json:
            try:
                parent_spec = MacroProposal.model_validate(json.loads(winner.spec_json))
            except Exception:
                parent_spec = None

        if parent_spec is None:
            return sample_random_macro_proposal(self.rng)

        # Select random mutation operator
        operator_name = self.rng.choice(OPERATOR_NAMES)
        return apply_named_operator(
            operator_name=operator_name,
            parent_spec=parent_spec,
            parent_id=winner.candidate_id,
            rng=self.rng,
        )
