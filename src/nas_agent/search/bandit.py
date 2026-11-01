"""
bandit.py
---------
Arm G: Contextual Multi-Armed Bandit via Thompson Sampling (No LLM) (Blueprint Section 8.4).

Specification:
1. Arms (8 Operators):
   - Macro: mutate_block, mutate_width, mutate_depth, mutate_expand
   - Micro: svd_lowrank, prune_internal, widen_internal, insert_block
2. Context Space (2 states):
   - context = "feasible" or "infeasible" (collapsed to prevent sample starvation across ~80 pulls)
3. Posterior Updates (Discounted Beta-Bernoulli with gamma = 0.95):
   - alpha_{c, k} <- 0.95 * alpha_{c, k} + success
   - beta_{c, k}  <- 0.95 * beta_{c, k}  + (1 - success)
4. Decision Rule:
   - Sample theta_{c, k} ~ Beta(alpha_{c, k}, beta_{c, k})
   - Select k* = argmax_k [ theta_{c, k} / E[cost_k] ]
   - E[cost_k] initialized to 90s for macro and 45s for micro; updated via EMA.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nas_agent.calibration.budget import sample_random_macro_proposal
from nas_agent.config import BudgetConfig
from nas_agent.memory.queries import (
    compute_violation_ratio,
    get_near_feasible_parents,
    get_pareto_front,
)
from nas_agent.memory.store import CandidateStore
from nas_agent.search.operators import (
    MACRO_OPERATORS,
    MICRO_OPERATORS,
    OPERATOR_NAMES,
    apply_named_operator,
)
from nas_agent.state import CandidateRecord, MacroProposal, Proposal


CONTEXTS = ["feasible", "infeasible"]


@dataclass
class BanditState:
    """Serializable posterior and cost parameters for the Thompson-sampling controller."""
    alpha: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            c: {op: 1.0 for op in OPERATOR_NAMES} for c in CONTEXTS
        }
    )
    beta: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            c: {op: 1.0 for op in OPERATOR_NAMES} for c in CONTEXTS
        }
    )
    expected_cost: Dict[str, float] = field(
        default_factory=lambda: {
            **{op: 90.0 for op in MACRO_OPERATORS},
            **{op: 45.0 for op in MICRO_OPERATORS},
        }
    )


class ContextualBanditProposer:
    """Thompson-sampling contextual bandit over mutation operators (Arm G)."""

    def __init__(
        self,
        seed: int = 42,
        discount: float = 0.95,
        cost_ema_alpha: float = 0.10,
        state: Optional[BanditState] = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.discount = discount
        self.cost_ema_alpha = cost_ema_alpha
        self.state = state or BanditState()

    def update_posterior(
        self,
        context: str,
        operator_name: str,
        success: bool,
        observed_cost: float,
    ) -> None:
        """Apply discounted Beta update and cost moving average.

        alpha <- 0.95 * alpha + success
        beta  <- 0.95 * beta  + (1 - success)
        """
        if context not in CONTEXTS or operator_name not in OPERATOR_NAMES:
            return

        succ_val = 1.0 if success else 0.0
        old_a = self.state.alpha[context][operator_name]
        old_b = self.state.beta[context][operator_name]

        self.state.alpha[context][operator_name] = self.discount * old_a + succ_val
        self.state.beta[context][operator_name] = self.discount * old_b + (1.0 - succ_val)

        # Online cost update via exponential moving average
        if observed_cost > 0.0:
            old_cost = self.state.expected_cost[operator_name]
            self.state.expected_cost[operator_name] = (
                (1.0 - self.cost_ema_alpha) * old_cost + self.cost_ema_alpha * observed_cost
            )

    def select_operator(self, context: str) -> str:
        """Sample theta ~ Beta(alpha, beta) and select operator maximizing theta / E[cost]."""
        if context not in CONTEXTS:
            context = "feasible"

        best_score = -1.0
        best_op = OPERATOR_NAMES[0]

        for op in OPERATOR_NAMES:
            a = self.state.alpha[context][op]
            b = self.state.beta[context][op]
            # Thompson sample
            theta = float(self.np_rng.beta(a, b))
            cost = max(1.0, self.state.expected_cost[op])
            score = theta / cost

            if score > best_score:
                best_score = score
                best_op = op

        return best_op

    def propose(
        self,
        store: CandidateStore,
        run_id: str,
        caps: BudgetConfig,
        min_val_acc: float,
    ) -> Tuple[Proposal, str, str]:
        """Propose candidate: determine context, sample operator via Thompson sampling, emit proposal.

        Returns
        -------
        Tuple[Proposal, context, operator_name]
        """
        all_candidates = store.get_run_candidates(run_id)
        last_candidate = all_candidates[-1] if all_candidates else None

        # 1. Determine Context (feasible vs infeasible)
        if last_candidate is None:
            context = "feasible"
        else:
            v = compute_violation_ratio(
                last_candidate.params_m,
                last_candidate.macs_g,
                last_candidate.peak_vram_mb,
                caps,
            )
            val_acc = last_candidate.val_acc or 0.0
            context = "feasible" if (v == 0.0 and val_acc >= min_val_acc) else "infeasible"

        # 2. Select operator using Thompson sampling
        operator_name = self.select_operator(context)

        # 3. Choose parent if operator is micro or macro mutation
        parent_candidate: Optional[CandidateRecord] = None
        if context == "feasible":
            pareto = get_pareto_front(store, run_id, caps, limit=12)
            if pareto:
                parent_candidate = self.rng.choice(pareto)
        else:
            near_feas = get_near_feasible_parents(store, run_id, caps, min_val_acc, max_violation=0.5)
            if near_feas:
                parent_candidate = self.rng.choice(near_feas)

        # Fallback to last evaluated candidate if available
        if parent_candidate is None and all_candidates:
            parent_candidate = self.rng.choice(all_candidates)

        parent_spec: Optional[MacroProposal] = None
        if parent_candidate and parent_candidate.spec_json:
            try:
                parent_spec = MacroProposal.model_validate(json.loads(parent_candidate.spec_json))
            except Exception:
                parent_spec = None

        parent_id = parent_candidate.candidate_id if parent_candidate else None

        proposal = apply_named_operator(
            operator_name=operator_name,
            parent_spec=parent_spec,
            parent_id=parent_id,
            rng=self.rng,
        )

        return proposal, context, operator_name

    def to_json(self) -> str:
        """Serialize bandit parameters to JSON string."""
        return json.dumps(
            {
                "alpha": self.state.alpha,
                "beta": self.state.beta,
                "expected_cost": self.state.expected_cost,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str, seed: int = 42) -> "ContextualBanditProposer":
        """Deserialize bandit parameters from JSON string."""
        data = json.loads(json_str)
        state = BanditState(
            alpha=data["alpha"],
            beta=data["beta"],
            expected_cost=data["expected_cost"],
        )
        return cls(seed=seed, state=state)
