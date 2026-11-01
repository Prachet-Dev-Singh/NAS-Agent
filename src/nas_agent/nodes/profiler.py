"""
profiler.py
-----------
Profiler Node for NAS-Agent LangGraph (Blueprint Section 5.1).

Responsibilities:
1. Validates measured resource consumption against hard hardware caps.
2. Classifies candidates into final status: 'feasible', 'infeasible', or error states.
3. Packages metrics for the Record node.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from nas_agent.config import AppConfig, load_config
from nas_agent.state import AgentState


logger = logging.getLogger(__name__)


class ProfilerNode:
    """Classifies candidate feasibility based on measured accuracy, MACs, params, and VRAM."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or load_config()

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        last_res = state.get("last_result") or {}
        status = last_res.get("status")

        # Non-successful training statuses pass through directly
        if status in ("duplicate", "invalid", "rejected_static", "gate_fail", "oom", "timeout", "nan", "crash"):
            return state

        caps = self.config.budget
        params_m = last_res.get("params_m", 0.0)
        macs_g = last_res.get("macs_g", 0.0)
        peak_vram_mb = last_res.get("peak_vram_mb", 0.0)
        val_acc = last_res.get("val_acc")

        # Feasibility check against caps
        exceeded = []
        if params_m > caps.params_cap_m:
            exceeded.append(f"params {params_m:.2f}M > {caps.params_cap_m:.2f}M")
        if macs_g > caps.macs_cap_g:
            exceeded.append(f"macs {macs_g:.4f}G > {caps.macs_cap_g:.4f}G")
        if peak_vram_mb > caps.peak_vram_cap_mb:
            exceeded.append(f"vram {peak_vram_mb:.0f}MB > {caps.peak_vram_cap_mb:.0f}MB")

        final_status: str
        if not exceeded and val_acc is not None:
            final_status = "feasible"
        else:
            final_status = "infeasible"
            if exceeded:
                logger.info(f"Candidate {last_res.get('candidate_id')} infeasible: {', '.join(exceeded)}")

        updated_last_res = {
            **last_res,
            "status": final_status,
        }

        return {"last_result": updated_last_res}
