"""
validate.py
-----------
Validate and Deduplication Node for NAS-Agent LangGraph (Blueprint Section 5.1).

Responsibilities:
1. Pydantic schema validation for MacroProposal and MicroProposal.
2. Spec hash fingerprint computation.
3. Database deduplication query (spec_hash lookup).
4. Duplicates return immediately with previous metrics and status 'duplicate' (0 GPU seconds).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import ValidationError

from nas_agent.memory.store import CandidateStore
from nas_agent.state import AgentState, MacroProposal, MicroProposal, Proposal


logger = logging.getLogger(__name__)


class ValidateNode:
    """Deterministic validation and deduplication node."""

    def __init__(self, store: Optional[CandidateStore] = None) -> None:
        self.store = store or CandidateStore()

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        proposal_dict = state.get("proposal")
        run_id = state.get("run_id", "run_default")
        candidate_id = state.get("candidate_id", "unknown_cand")
        consecutive_invalid = state.get("consecutive_invalid", 0)

        if not proposal_dict or not isinstance(proposal_dict, dict):
            return {
                "consecutive_invalid": consecutive_invalid + 1,
                "last_result": {
                    "candidate_id": candidate_id,
                    "status": "invalid",
                    "error_tail": "Missing or non-dict proposal in state",
                    "own_gpu_seconds": 0.0,
                },
            }

        # 1. Pydantic Schema Validation
        proposal: Proposal
        kind = proposal_dict.get("kind", "macro")
        try:
            if kind == "micro":
                proposal = MicroProposal.model_validate(proposal_dict)
            else:
                proposal = MacroProposal.model_validate(proposal_dict)
        except (ValidationError, ValueError) as e:
            logger.warning(f"Validation failed for candidate {candidate_id}: {e}")
            return {
                "consecutive_invalid": consecutive_invalid + 1,
                "last_result": {
                    "candidate_id": candidate_id,
                    "status": "invalid",
                    "error_tail": f"Pydantic validation error: {str(e)[:300]}",
                    "own_gpu_seconds": 0.0,
                },
            }

        # 2. Spec Hash Computation
        spec_hash: str
        parent_id = None
        if isinstance(proposal, MicroProposal):
            parent_id = proposal.parent_id
            parent_rec = self.store.get_candidate(parent_id)
            parent_spec_hash = parent_rec.spec_hash if parent_rec else "unknown_parent"
            spec_hash = proposal.spec_hash(parent_spec_hash)
        else:
            spec_hash = proposal.spec_hash()

        # 3. Deduplication Check against Store
        duplicate_rec = self.store.find_by_spec_hash(spec_hash, run_id=run_id)
        if duplicate_rec is not None:
            logger.info(f"Duplicate candidate detected: {candidate_id} matches {duplicate_rec.candidate_id}")
            return {
                "consecutive_invalid": 0,
                "last_result": {
                    "candidate_id": candidate_id,
                    "spec_hash": spec_hash,
                    "parent_id": parent_id,
                    "status": "duplicate",
                    "val_acc": duplicate_rec.val_acc,
                    "params_m": duplicate_rec.params_m,
                    "macs_g": duplicate_rec.macs_g,
                    "peak_vram_mb": duplicate_rec.peak_vram_mb,
                    "own_gpu_seconds": 0.0,
                    "error_tail": f"Duplicate of candidate {duplicate_rec.candidate_id}",
                    "spec_json": duplicate_rec.spec_json,
                },
            }

        # Validated and Unique
        return {
            "consecutive_invalid": 0,
            "last_result": {
                "candidate_id": candidate_id,
                "spec_hash": spec_hash,
                "parent_id": parent_id,
                "status": "validated",
                "own_gpu_seconds": 0.0,
            },
        }
