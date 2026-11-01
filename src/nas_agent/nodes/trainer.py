"""
trainer.py
----------
Trainer Node for NAS-Agent LangGraph (Blueprint Section 5.1 & 6).

Responsibilities:
1. Skips candidates that were rejected by static caps, gate failure, or duplication.
2. Prepares isolated candidate execution directory with spec.json and checkpoint bindings.
3. Dispatches training worker to LocalExecutor (or ModalExecutor) in a separate process group.
4. Gathers execution results (accuracy, training time, peak VRAM, errors).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from nas_agent.config import AppConfig, load_config
from nas_agent.harness.executor import Executor, LocalExecutor
from nas_agent.memory.store import CandidateStore
from nas_agent.state import AgentState, MacroProposal, MicroProposal


logger = logging.getLogger(__name__)


class TrainerNode:
    """Dispatches candidate training to subprocess-isolated worker."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        store: Optional[CandidateStore] = None,
        executor: Optional[Executor] = None,
        runs_dir: str = "experiments/runs",
    ) -> None:
        self.config = config or load_config()
        self.store = store or CandidateStore()
        self.executor = executor or LocalExecutor()
        self.runs_dir = runs_dir

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        last_res = state.get("last_result") or {}
        status = last_res.get("status")

        # Skip training if candidate was already classified (e.g. rejected_static, gate_fail, duplicate)
        if status != "gate_passed":
            return state

        run_id = state.get("run_id", "run_default")
        candidate_id = state.get("candidate_id", "unknown_cand")
        proposal_dict = state.get("proposal", {})
        cand_dir = os.path.join(self.runs_dir, run_id, "candidates", candidate_id)
        os.makedirs(cand_dir, exist_ok=True)

        spec_path = os.path.join(cand_dir, "spec.json")
        out_json = os.path.join(cand_dir, "metrics.json")
        log_dir = os.path.join(cand_dir, "logs")
        ckpt_out = os.path.join(cand_dir, f"{candidate_id}.pt")

        kind = proposal_dict.get("kind", "macro")
        enriched_spec = dict(proposal_dict)

        if kind == "micro":
            parent_id = proposal_dict.get("parent_id")
            parent_rec = self.store.get_candidate(parent_id) if parent_id else None
            if parent_rec and parent_rec.spec_json:
                enriched_spec["parent_spec"] = json.loads(parent_rec.spec_json)
                if parent_rec.ckpt_ref and os.path.exists(parent_rec.ckpt_ref):
                    enriched_spec["parent_ckpt"] = parent_rec.ckpt_ref

        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(enriched_spec, f, indent=2)

        # Worker parameters
        extra_args = [
            "--ckpt-out", ckpt_out,
            "--config-hash", state.get("config_hash", "frozen"),
        ]

        # Use mock data if specified in config or test environment
        if getattr(self.config, "mock_data", False) or os.getenv("NAS_MOCK_DATA") == "1":
            extra_args.append("--mock-data")

        timeout_sec = 600 if kind == "macro" else 300

        # Execute training subprocess
        logger.info(f"Launching training worker for {candidate_id} ({kind})")
        exec_result = self.executor.run(
            candidate_id=candidate_id,
            spec_path=spec_path,
            out_json=out_json,
            log_dir=log_dir,
            timeout_sec=timeout_sec,
            extra_args=extra_args,
        )

        worker_status = exec_result.get("status", "crash")
        val_acc = exec_result.get("val_acc")
        train_seconds = float(exec_result.get("train_seconds", 0.0))
        peak_vram_mb = float(exec_result.get("peak_vram_mb", 0.0))
        error_tail = exec_result.get("error_tail")

        ckpt_saved = ckpt_out if (worker_status == "ok" and os.path.exists(ckpt_out)) else None

        updated_last_res = {
            **last_res,
            "status": worker_status,
            "val_acc": val_acc,
            "own_gpu_seconds": train_seconds,
            "peak_vram_mb": peak_vram_mb,
            "ckpt_ref": ckpt_saved,
            "error_tail": error_tail,
        }

        return {"last_result": updated_last_res}
