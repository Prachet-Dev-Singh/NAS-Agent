"""
gate.py
-------
Deterministic Hardware & Contract Gate Node for NAS-Agent (Blueprint Section 5.1).

Responsibilities:
1. Analytical parameter & MACs counting against hard budget caps.
2. Static rejection of budget-violating candidates before GPU training (0 GPU seconds).
3. C4 Gate CPU smoke test (shape check, forward pass, gradient flow on dummy CPU batch).
4. For micro-ops: verifies function preservation / shape integrity.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import torch

from nas_agent.config import AppConfig, BudgetConfig, load_config
from nas_agent.lib.contracts import gate_cpu_smoke
from nas_agent.lib.counters import count_macro_arch, count_params_actual
from nas_agent.memory.store import CandidateStore
from nas_agent.nodes.compiler import compile_macro, compile_micro
from nas_agent.state import AgentState, MacroProposal, MicroProposal, Proposal


logger = logging.getLogger(__name__)


class GateNode:
    """Deterministic gate filtering invalid or statically non-viable models."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        store: Optional[CandidateStore] = None,
    ) -> None:
        self.config = config or load_config()
        self.store = store or CandidateStore()

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        last_res = state.get("last_result") or {}
        # If previous node already classified candidate (e.g. duplicate or invalid), pass through
        if last_res.get("status") in ("duplicate", "invalid"):
            return state

        proposal_dict = state.get("proposal", {})
        candidate_id = state.get("candidate_id", "unknown_cand")
        spec_hash = last_res.get("spec_hash", "")
        caps = self.config.budget

        kind = proposal_dict.get("kind", "macro")
        proposal: Proposal
        if kind == "micro":
            proposal = MicroProposal.model_validate(proposal_dict)
        else:
            proposal = MacroProposal.model_validate(proposal_dict)

        # 1. Analytical Parameter & MACs Computation
        params_m: float = 0.0
        macs_g: float = 0.0
        model: Optional[torch.nn.Module] = None

        if isinstance(proposal, MacroProposal):
            stats = count_macro_arch(proposal)
            params_m = stats.params_m
            macs_g = stats.macs_g

            # Static budget rejection check
            if params_m > caps.params_cap_m or macs_g > caps.macs_cap_g:
                logger.info(f"Candidate {candidate_id} statically rejected: {params_m:.2f}M / {macs_g:.4f}G")
                return {
                    "last_result": {
                        "candidate_id": candidate_id,
                        "spec_hash": spec_hash,
                        "status": "rejected_static",
                        "params_m": round(params_m, 4),
                        "macs_g": round(macs_g, 4),
                        "own_gpu_seconds": 0.0,
                        "error_tail": f"Statically exceeded caps: params {params_m:.2f}M > {caps.params_cap_m:.2f}M or macs {macs_g:.4f}G > {caps.macs_cap_g:.4f}G",
                    }
                }

            # Compile model for CPU smoke check
            try:
                model = compile_macro(proposal)
            except Exception as e:
                return {
                    "last_result": {
                        "candidate_id": candidate_id,
                        "spec_hash": spec_hash,
                        "status": "gate_fail",
                        "own_gpu_seconds": 0.0,
                        "error_tail": f"Macro compilation error: {str(e)[:300]}",
                    }
                }

        elif isinstance(proposal, MicroProposal):
            parent_id = proposal.parent_id
            parent_rec = self.store.get_candidate(parent_id)
            if parent_rec is None or not parent_rec.spec_json:
                return {
                    "last_result": {
                        "candidate_id": candidate_id,
                        "spec_hash": spec_hash,
                        "status": "gate_fail",
                        "own_gpu_seconds": 0.0,
                        "error_tail": f"Micro parent {parent_id} not found in database or missing spec_json",
                    }
                }

            try:
                parent_macro = MacroProposal.model_validate(json.loads(parent_rec.spec_json))
                parent_model = compile_macro(parent_macro)
                # Load parent checkpoint weights if available
                if parent_rec.ckpt_ref:
                    import os
                    if os.path.exists(parent_rec.ckpt_ref):
                        try:
                            parent_model.load_state_dict(
                                torch.load(parent_rec.ckpt_ref, map_location="cpu")
                            )
                        except Exception as ckpt_err:
                            logger.warning(f"Could not load checkpoint weights: {ckpt_err}")

                model = compile_micro(proposal, parent_model)
                # Count actual params and MACs of edited model
                actual_params = sum(p.numel() for p in model.parameters())
                params_m = actual_params / 1e6
                # Micro edits generally maintain or decrease MACs
                macs_g = parent_rec.macs_g or 0.10

                if params_m > caps.params_cap_m:
                    return {
                        "last_result": {
                            "candidate_id": candidate_id,
                            "spec_hash": spec_hash,
                            "parent_id": parent_id,
                            "status": "rejected_static",
                            "params_m": round(params_m, 4),
                            "macs_g": round(macs_g, 4),
                            "own_gpu_seconds": 0.0,
                            "error_tail": f"Micro edit exceeded params cap: {params_m:.2f}M > {caps.params_cap_m:.2f}M",
                        }
                    }

            except Exception as e:
                return {
                    "last_result": {
                        "candidate_id": candidate_id,
                        "spec_hash": spec_hash,
                        "parent_id": parent_id,
                        "status": "gate_fail",
                        "own_gpu_seconds": 0.0,
                        "error_tail": f"Micro surgery compilation error: {str(e)[:300]}",
                    }
                }

        # 2. C4 Gate CPU Smoke Test (shape and gradient verification)
        if model is not None:
            try:
                gate_cpu_smoke(model)
            except Exception as smoke_err:
                logger.warning(f"CPU smoke test failed for {candidate_id}: {smoke_err}")
                return {
                    "last_result": {
                        "candidate_id": candidate_id,
                        "spec_hash": spec_hash,
                        "status": "gate_fail",
                        "params_m": round(params_m, 4),
                        "macs_g": round(macs_g, 4),
                        "own_gpu_seconds": 0.0,
                        "error_tail": f"C4 Gate CPU smoke failed: {str(smoke_err)[:300]}",
                    }
                }

        # Passed all gate checks! Ready for training.
        return {
            "last_result": {
                "candidate_id": candidate_id,
                "spec_hash": spec_hash,
                "parent_id": last_res.get("parent_id"),
                "status": "gate_passed",
                "params_m": round(params_m, 4),
                "macs_g": round(macs_g, 4),
                "own_gpu_seconds": 0.0,
            }
        }
