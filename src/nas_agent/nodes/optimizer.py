"""
optimizer.py
------------
Optimizer Node for NAS-Agent LangGraph (Blueprint Section 5.1).

Dispatches proposal generation based on experimental arm:
- Arm A: Uniform Random Search
- Arm B: Regularized (Aging) Evolution
- Arm C: LLM No-Memory Ablation
- Arm D: Full LLM Agent with Structured Pareto Memory
- Arm E: LLM Raw In-Context History Ablation
- Arm G: Contextual Thompson-Sampling Bandit
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from nas_agent.config import AppConfig, BudgetConfig, load_config
from nas_agent.llm.gateway import LLMGateway
from nas_agent.llm.prompts import (
    SYSTEM_PROMPT,
    format_arm_c_prompt,
    format_arm_d_prompt,
    format_arm_e_prompt,
)
from nas_agent.memory.store import CandidateStore
from nas_agent.search.bandit import ContextualBanditProposer
from nas_agent.search.evolution import RegularizedEvolutionProposer
from nas_agent.search.random_search import RandomSearchProposer
from nas_agent.state import AgentState, MacroProposal, MicroProposal, Proposal


class OptimizerNode:
    """LangGraph node generating architecture proposals for the current iteration."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        store: Optional[CandidateStore] = None,
        gateway: Optional[LLMGateway] = None,
        bandit_proposer: Optional[ContextualBanditProposer] = None,
    ) -> None:
        self.config = config or load_config()
        self.store = store or CandidateStore()
        self.gateway = gateway or LLMGateway()
        self.bandit_proposer = bandit_proposer or ContextualBanditProposer(seed=42)

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        arm = state.get("arm", "D")
        run_id = state.get("run_id", "run_default")
        seed = state.get("seed", 42)
        iteration = state.get("iteration", 1)
        candidate_id = f"cand_{arm}_{run_id[:6]}_{iteration:03d}"

        caps = self.config.budget
        min_val_acc = self.config.calibration.min_val_acc
        sigma_acc = self.config.calibration.sigma_acc
        gpu_seconds_used = state.get("gpu_seconds_used", 0.0)
        gpu_seconds_budget = state.get("gpu_seconds_budget", 7200.0)
        llm_calls_used = state.get("llm_calls_used", 0)

        proposal: Proposal
        proposal_dict: Dict[str, Any]
        technique = "from_scratch"
        new_llm_calls = 0

        # --- Baseline Arms ---
        if arm == "A":
            proposer_a = RandomSearchProposer(seed=seed + iteration)
            proposal = proposer_a.propose(
                store=self.store,
                run_id=run_id,
                caps=caps,
                min_val_acc=min_val_acc,
            )
            proposal_dict = proposal.model_dump()

        elif arm == "B":
            proposer_b = RegularizedEvolutionProposer(seed=seed + iteration)
            proposal = proposer_b.propose(
                store=self.store,
                run_id=run_id,
                caps=caps,
                min_val_acc=min_val_acc,
            )
            proposal_dict = proposal.model_dump()

        elif arm == "G":
            proposal, context, op_name = self.bandit_proposer.propose(
                store=self.store,
                run_id=run_id,
                caps=caps,
                min_val_acc=min_val_acc,
            )
            proposal_dict = proposal.model_dump()
            proposal_dict["_bandit_context"] = context
            proposal_dict["_bandit_operator"] = op_name

        # --- LLM Optimizer Arms (C, D, E) ---
        elif arm in ("C", "D", "E"):
            user_prompt = ""
            if arm == "C":
                user_prompt = format_arm_c_prompt(
                    caps=caps,
                    gpu_seconds_used=gpu_seconds_used,
                    gpu_seconds_budget=gpu_seconds_budget,
                )
            elif arm == "D":
                all_candidates = self.store.get_run_candidates(run_id)
                last_candidate = all_candidates[-1] if all_candidates else None
                user_prompt = format_arm_d_prompt(
                    store=self.store,
                    run_id=run_id,
                    caps=caps,
                    sigma_acc=sigma_acc,
                    min_val_acc=min_val_acc,
                    gpu_seconds_used=gpu_seconds_used,
                    gpu_seconds_budget=gpu_seconds_budget,
                    last_candidate=last_candidate,
                )
            elif arm == "E":
                user_prompt = format_arm_e_prompt(
                    store=self.store,
                    run_id=run_id,
                    caps=caps,
                    gpu_seconds_used=gpu_seconds_used,
                    gpu_seconds_budget=gpu_seconds_budget,
                )

            proposal, audit = self.gateway.query_proposal(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                run_id=run_id,
                store=self.store,
                salt=candidate_id,
            )
            proposal_dict = proposal.model_dump()
            new_llm_calls = 1

        else:
            raise ValueError(f"Unknown experimental arm: {arm}")

        return {
            "proposal": proposal_dict,
            "candidate_id": candidate_id,
            "llm_calls_used": llm_calls_used + new_llm_calls,
        }
