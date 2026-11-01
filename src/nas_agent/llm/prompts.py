"""
prompts.py
----------
LLM prompt templates and SQL context formatters for the NAS Optimizer (Blueprint Sections 7 & 8.3).

Supports:
- Arm D: Full structured Pareto memory (Pareto front, near-feasible parents, technique stats, last result).
- Arm C: No-memory ablation (only budget caps and rules; no candidate history).
- Arm E: Raw in-context history (raw sequential log of last 40 attempts; no SQL aggregation).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from nas_agent.config import BudgetConfig
from nas_agent.memory.queries import (
    compute_violation_ratio,
    get_near_feasible_parents,
    get_pareto_front,
    get_technique_statistics,
)
from nas_agent.memory.store import CandidateStore
from nas_agent.state import CandidateRecord


SYSTEM_PROMPT = """You are an expert Neural Architecture Search (NAS) Optimizer for CIFAR-10.
Your goal is to find Pareto-optimal neural network architectures that maximize validation accuracy while strictly respecting hard resource constraints.

Available Block Primitives (under uniform residual contract):
1. 'conv': Standard 3x3 depthwise/pointwise convolutional block.
2. 'dwsep': Inverted bottleneck depthwise separable block.
3. 'resnet': Classic pre-activation residual block.
4. 'attn_lite': Lightweight single-head self-attention with spatial reduction.
5. 'ssm': State-space sequence mixer (Mamba-style selective SSM).

Search Space Rules:
- A network has 3 to 4 stages.
- Stage 0 downsampling stride MUST be 1 (the stem handles input scale). Subsequent stages use stride 2.
- Stage 0 block type MUST be 'conv', 'dwsep', or 'resnet'.
- Each stage width must be a multiple of 16 (between 16 and 256).
- Each stage expand must be between 1 and 4 (hidden dim = expand * width).
- Each stage depth must be between 1 and 6. Total depth across all stages must NOT exceed 16.

Search Modes:
1. Macro Proposal (explore new topology from random init):
   Emit JSON:
   {
     "kind": "macro",
     "stem_width": 32,
     "stages": [
       {"block": "conv", "width": 32, "expand": 2, "depth": 2, "stride": 1},
       {"block": "resnet", "width": 64, "expand": 2, "depth": 2, "stride": 2},
       {"block": "attn_lite", "width": 128, "expand": 2, "depth": 2, "stride": 2}
     ],
     "norm": "bn",
     "activation": "silu",
     "rationale": "Brief scientific reason (under 400 chars)"
   }

2. Micro Proposal (refine an existing near-feasible checkpoint via surgery):
   Emit JSON:
   {
     "kind": "micro",
     "parent_id": "exact_candidate_id_from_near_feasible_parents",
     "op": {
       "technique": "svd_lowrank" | "prune_internal" | "widen_internal" | "insert_block",
       ... operator-specific fields ...
     },
     "target_stages": [0, 1],
     "distill": {"enabled": false},
     "rationale": "Brief reason for surgical edit"
   }

Operator Fields:
- svd_lowrank: {"technique": "svd_lowrank", "rank_ratio": 0.5, "which": "both"}
- prune_internal: {"technique": "prune_internal", "sparsity": 0.3}
- widen_internal: {"technique": "widen_internal", "factor": 1.5}
- insert_block: {"technique": "insert_block", "block": "conv"}

CRITICAL: Return ONLY valid JSON matching either the MacroProposal or MicroProposal schema. Do NOT include markdown code blocks, conversational text, or commentary outside the JSON object.
"""


def format_arm_d_prompt(
    store: CandidateStore,
    run_id: str,
    caps: BudgetConfig,
    sigma_acc: float,
    min_val_acc: float,
    gpu_seconds_used: float,
    gpu_seconds_budget: float,
    last_candidate: Optional[CandidateRecord] = None,
) -> str:
    """Format structured SQL context for Arm D (Full Optimizer Agent)."""
    # 1. Budget and status
    remaining_gpu = max(0.0, gpu_seconds_budget - gpu_seconds_used)
    budget_lines = [
        "### 1. Hardware Constraints & Budget Status",
        f"- Parameters Cap: <= {caps.params_cap_m:.2f} M",
        f"- MACs Cap: <= {caps.macs_cap_g:.4f} G",
        f"- Peak VRAM Cap: <= {caps.peak_vram_cap_mb:.1f} MB",
        f"- Min Accuracy Floor: >= {min_val_acc:.4f}",
        f"- Measurement Noise Floor (sigma_acc): +/- {sigma_acc:.4f}",
        f"- GPU Budget Used: {gpu_seconds_used:.1f}s / {gpu_seconds_budget:.1f}s ({remaining_gpu:.1f}s remaining)",
        "",
    ]

    # 2. Pareto Front (up to 12)
    pareto_candidates = get_pareto_front(store, run_id, caps, limit=12)
    pareto_lines = ["### 2. Current Feasible Pareto Front (Non-Dominated Archive)"]
    if not pareto_candidates:
        pareto_lines.append("No feasible candidates discovered yet.")
    else:
        pareto_lines.append("| Candidate ID | Val Acc | Params (M) | MACs (G) | Peak VRAM (MB) | Blocks |")
        pareto_lines.append("|---|---|---|---|---|---|")
        for c in pareto_candidates:
            acc_str = f"{c.val_acc:.4f}" if c.val_acc is not None else "N/A"
            p_str = f"{c.params_m:.2f}" if c.params_m is not None else "N/A"
            m_str = f"{c.macs_g:.4f}" if c.macs_g is not None else "N/A"
            v_str = f"{c.peak_vram_mb:.0f}" if c.peak_vram_mb is not None else "N/A"
            pareto_lines.append(f"| {c.candidate_id} | {acc_str} | {p_str} | {m_str} | {v_str} | {c.block_types} |")
    pareto_lines.append("")

    # 3. Near-Feasible Parents (Eligible for Micro Surgery)
    near_parents = get_near_feasible_parents(store, run_id, caps, min_val_acc=min_val_acc, max_violation=0.5)
    parent_lines = ["### 3. Checkpointed Candidates Available for Micro Surgery"]
    if not near_parents:
        parent_lines.append("No checkpointed candidates currently satisfy v(x) <= 0.5 and val_acc >= min_val_acc.")
    else:
        parent_lines.append("| Parent ID | Val Acc | Violation v(x) | Params (M) | MACs (G) | Status |")
        parent_lines.append("|---|---|---|---|---|---|")
        for p in near_parents:
            v_ratio = compute_violation_ratio(p.params_m, p.macs_g, p.peak_vram_mb, caps)
            acc_str = f"{p.val_acc:.4f}" if p.val_acc is not None else "N/A"
            parent_lines.append(
                f"| {p.candidate_id} | {acc_str} | {v_ratio:.3f} | {p.params_m:.2f} | {p.macs_g:.4f} | {p.status} |"
            )
    parent_lines.append("")

    # 4. Technique Statistics (Rolling 15 attempts)
    tech_stats = get_technique_statistics(store, run_id, window=15)
    tech_lines = ["### 4. Rolling Technique Statistics (Last 15 Attempts per Technique)"]
    if not tech_stats:
        tech_lines.append("No technique history yet.")
    else:
        tech_lines.append("| Technique | Attempts | Success Rate | Median r_hv (/s) |")
        tech_lines.append("|---|---|---|---|")
        for t, stat in sorted(tech_stats.items()):
            succ_pct = f"{stat['success_rate'] * 100:.1f}%"
            med_rhv = f"{stat['median_r_hv']:.2e}"
            tech_lines.append(f"| {t} | {stat['attempts']} | {succ_pct} | {med_rhv} |")
    tech_lines.append("")

    # 5. Last Evaluated Candidate
    last_lines = ["### 5. Last Candidate Evaluation Result"]
    if last_candidate is None:
        last_lines.append("This is iteration 1. No candidates have been evaluated yet.")
    else:
        acc_str = f"{last_candidate.val_acc:.4f}" if last_candidate.val_acc is not None else "N/A"
        last_lines.append(f"- ID: {last_candidate.candidate_id}")
        last_lines.append(f"- Type: {last_candidate.proposal_type} ({last_candidate.technique})")
        last_lines.append(f"- Status: {last_candidate.status}")
        last_lines.append(f"- Val Acc: {acc_str} | Params: {last_candidate.params_m}M | MACs: {last_candidate.macs_g}G")
        if last_candidate.error_tail:
            last_lines.append(f"- Failure Reason: {last_candidate.error_tail[:200]}")
    last_lines.append("")

    # 6. Action prompt
    instruction = [
        "### Instruction",
        "Based on the Pareto front gaps, near-feasible checkpoints, and technique performance:",
        "- If near-feasible parents exist with high accuracy but slight violations, prefer micro-surgery (low-rank, pruning) to compress into feasibility.",
        "- If existing Pareto points plateau, explore new macro topologies or widening.",
        "Emit the single JSON proposal for the next candidate now.",
    ]

    return "\n".join(budget_lines + pareto_lines + parent_lines + tech_lines + last_lines + instruction)


def format_arm_c_prompt(
    caps: BudgetConfig,
    gpu_seconds_used: float,
    gpu_seconds_budget: float,
) -> str:
    """Format no-memory ablation context for Arm C (LLM No-Memory)."""
    remaining_gpu = max(0.0, gpu_seconds_budget - gpu_seconds_used)
    lines = [
        "### Hardware Constraints & Search Budget",
        f"- Parameters Cap: <= {caps.params_cap_m:.2f} M",
        f"- MACs Cap: <= {caps.macs_cap_g:.4f} G",
        f"- Peak VRAM Cap: <= {caps.peak_vram_cap_mb:.1f} MB",
        f"- Remaining GPU Budget: {remaining_gpu:.1f}s",
        "",
        "### Instruction",
        "Propose a high-accuracy candidate neural architecture within the specified hardware constraints.",
        "Emit a single valid JSON MacroProposal now.",
    ]
    return "\n".join(lines)


def format_arm_e_prompt(
    store: CandidateStore,
    run_id: str,
    caps: BudgetConfig,
    gpu_seconds_used: float,
    gpu_seconds_budget: float,
) -> str:
    """Format raw in-context history for Arm E (Raw In-Context History Ablation)."""
    remaining_gpu = max(0.0, gpu_seconds_budget - gpu_seconds_used)
    all_candidates = store.get_run_candidates(run_id)
    history_window = all_candidates[-40:] if all_candidates else []

    lines = [
        "### Hardware Constraints & Search Budget",
        f"- Parameters Cap: <= {caps.params_cap_m:.2f} M",
        f"- MACs Cap: <= {caps.macs_cap_g:.4f} G",
        f"- Peak VRAM Cap: <= {caps.peak_vram_cap_mb:.1f} MB",
        f"- Remaining GPU Budget: {remaining_gpu:.1f}s",
        "",
        "### Evaluation History (Last 40 Evaluated Candidates)",
    ]

    if not history_window:
        lines.append("No candidates evaluated yet.")
    else:
        lines.append("| Iter | ID | Type | Tech | Status | Val Acc | Params | MACs | r_hv | r_viol | Success |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in history_window:
            acc_str = f"{c.val_acc:.4f}" if c.val_acc is not None else "N/A"
            p_str = f"{c.params_m:.2f}" if c.params_m is not None else "N/A"
            m_str = f"{c.macs_g:.4f}" if c.macs_g is not None else "N/A"
            lines.append(
                f"| {c.iteration} | {c.candidate_id} | {c.proposal_type} | {c.technique} | {c.status} | "
                f"{acc_str} | {p_str} | {m_str} | {c.r_hv:.1e} | {c.r_viol:.3f} | {int(c.success)} |"
            )

    lines.extend([
        "",
        "### Instruction",
        "Analyze the historical sequence of attempts and propose the next candidate architecture (macro or micro).",
        "Emit a single valid JSON proposal now.",
    ])

    return "\n".join(lines)
