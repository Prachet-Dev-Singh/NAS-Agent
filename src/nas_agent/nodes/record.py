"""
record.py
---------
Record and Experience Memory Node for NAS-Agent LangGraph (Blueprint Section 5.1 & 8).

Responsibilities:
1. Assembles and persists CandidateRecord into SQLite database.
2. Computes decoupled Pareto rewards (r_hv, r_viol, success boolean).
3. Updates Pareto membership flags across the run archive.
4. Updates Arm G Thompson-sampling posterior and cost EMA if applicable.
5. Tracks cumulative GPU-seconds and budget termination.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from nas_agent.config import AppConfig, load_config
from nas_agent.memory.queries import calculate_rewards_and_update
from nas_agent.memory.store import CandidateStore
from nas_agent.search.bandit import ContextualBanditProposer
from nas_agent.state import AgentState, CandidateRecord


logger = logging.getLogger(__name__)


class RecordNode:
    """Persists candidate evaluations to SQLite and updates search statistics."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        store: Optional[CandidateStore] = None,
        bandit_proposer: Optional[ContextualBanditProposer] = None,
    ) -> None:
        self.config = config or load_config()
        self.store = store or CandidateStore()
        self.bandit_proposer = bandit_proposer

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        run_id = state.get("run_id", "run_default")
        arm = state.get("arm", "D")
        seed = state.get("seed", 42)
        iteration = state.get("iteration", 1)
        config_hash = state.get("config_hash", "frozen")
        proposal_dict = state.get("proposal") or {}
        last_res = state.get("last_result") or {}

        candidate_id = state.get("candidate_id") or last_res.get("candidate_id", f"cand_{iteration}")
        spec_hash = last_res.get("spec_hash") or f"hash_{candidate_id}"
        parent_id = last_res.get("parent_id")
        status = last_res.get("status", "crash")

        params_m = last_res.get("params_m")
        macs_g = last_res.get("macs_g")
        peak_vram_mb = last_res.get("peak_vram_mb")
        val_acc = last_res.get("val_acc")
        own_gpu_seconds = float(last_res.get("own_gpu_seconds", 0.0))
        ckpt_ref = last_res.get("ckpt_ref")
        error_tail = last_res.get("error_tail")

        kind = proposal_dict.get("kind", "macro")
        technique = "from_scratch"
        if kind == "micro":
            op = proposal_dict.get("op", {})
            technique = op.get("technique", "micro")
        elif "_bandit_operator" in proposal_dict:
            technique = proposal_dict["_bandit_operator"]

        # Extract block types
        block_types = "unknown"
        if kind == "macro":
            stages = proposal_dict.get("stages", [])
            block_types = ",".join(s.get("block", "conv") for s in stages)
        elif parent_id:
            parent = self.store.get_candidate(parent_id)
            if parent:
                block_types = parent.block_types

        gpu_seconds_used = state.get("gpu_seconds_used", 0.0) + own_gpu_seconds

        # Construct CandidateRecord
        record = CandidateRecord(
            candidate_id=candidate_id,
            run_id=run_id,
            arm=arm,
            seed=seed,
            iteration=iteration,
            config_hash=config_hash,
            spec_hash=spec_hash,
            parent_id=parent_id,
            proposal_type=kind,
            technique=technique,
            distill=bool(proposal_dict.get("distill", {}).get("enabled", False)) if kind == "micro" else False,
            block_types=block_types,
            params_m=params_m,
            macs_g=macs_g,
            peak_vram_mb=peak_vram_mb,
            val_acc=val_acc,
            status=status,
            own_gpu_seconds=own_gpu_seconds,
            cumulative_gpu_seconds=gpu_seconds_used,
            ckpt_ref=ckpt_ref,
            error_tail=error_tail,
            spec_json=json.dumps(proposal_dict),
            rationale=proposal_dict.get("rationale"),
        )

        # 1. Insert into candidate store
        self.store.insert_candidate(record)

        # 2. Compute decoupled Pareto rewards and update DB flags
        r_hv, r_viol, success = calculate_rewards_and_update(
            store=self.store,
            candidate=record,
            caps=self.config.budget,
            min_val_acc=self.config.calibration.min_val_acc,
        )

        # 3. Update Bandit posterior if Arm G
        if arm == "G" and self.bandit_proposer is not None:
            context = proposal_dict.get("_bandit_context", "feasible")
            operator = proposal_dict.get("_bandit_operator", technique)
            self.bandit_proposer.update_posterior(
                context=context,
                operator_name=operator,
                success=success,
                observed_cost=own_gpu_seconds,
            )

        # 4. Compact history summary for graph state
        compact_summary = {
            "iteration": iteration,
            "candidate_id": candidate_id,
            "proposal_type": kind,
            "technique": technique,
            "status": status,
            "val_acc": val_acc,
            "params_m": params_m,
            "macs_g": macs_g,
            "own_gpu_seconds": own_gpu_seconds,
            "r_hv": r_hv,
            "r_viol": r_viol,
            "success": success,
        }

        logger.info(
            f"Candidate {candidate_id} recorded: status={status}, acc={val_acc}, "
            f"gpu_sec={own_gpu_seconds:.1f}s, r_hv={r_hv:.2e}, success={success}"
        )

        return {
            "gpu_seconds_used": gpu_seconds_used,
            "iteration": iteration + 1,
            "last_result": compact_summary,
            "history": [compact_summary],
            "proposal": None,
            "candidate_id": None,
        }
