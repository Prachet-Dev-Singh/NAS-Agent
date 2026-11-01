"""
queries.py
----------
Deterministic SQL and Pareto queries for optimizer prompts and reward calculations (Blueprint Section 8).

Implements:
1. Pareto front extraction across (val_acc, params_m, macs_g).
2. Near-feasible parent candidate selection for micro refinement.
3. Technique statistics aggregation over rolling 15-attempt window (median r_hv & success rate).
4. Decoupled reward computation (r_hv per GPU-second, r_viol reduction, and success boolean).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from nas_agent.config import BudgetConfig
from nas_agent.eval.hypervolume import (
    calculate_normalized_hypervolume,
    filter_non_dominated_3d,
)
from nas_agent.memory.store import CandidateStore
from nas_agent.state import CandidateRecord


def compute_violation_ratio(
    params_m: Optional[float],
    macs_g: Optional[float],
    peak_vram_mb: Optional[float],
    caps: BudgetConfig,
) -> float:
    """Compute normalized constraint violation ratio v(x).

    Formula (Blueprint Section 8.1):
        v(x) = max_{m in {params, macs, vram}} max(0, metric_m / cap_m - 1)
    """
    v_params = max(0.0, (params_m / caps.params_cap_m) - 1.0) if params_m is not None else 0.0
    v_macs = max(0.0, (macs_g / caps.macs_cap_g) - 1.0) if macs_g is not None else 0.0
    v_vram = max(0.0, (peak_vram_mb / caps.peak_vram_cap_mb) - 1.0) if peak_vram_mb is not None else 0.0

    return float(max(v_params, v_macs, v_vram))


def get_pareto_front(
    store: CandidateStore,
    run_id: str,
    caps: BudgetConfig,
    limit: int = 12,
) -> List[CandidateRecord]:
    """Retrieve non-dominated feasible candidates for run_id."""
    candidates = store.get_run_candidates(run_id)

    feasible: List[CandidateRecord] = []
    for c in candidates:
        if c.val_acc is None or c.params_m is None or c.macs_g is None:
            continue
        # Feasibility check against caps
        if c.params_m <= caps.params_cap_m and c.macs_g <= caps.macs_cap_g:
            v_ratio = compute_violation_ratio(c.params_m, c.macs_g, c.peak_vram_mb, caps)
            if v_ratio == 0.0 and c.status in ("ok", "feasible"):
                feasible.append(c)

    if not feasible:
        return []

    # Map to 3D points in normalized headroom space: (acc, p_headroom, m_headroom)
    points = [
        (
            float(c.val_acc),
            max(0.0, 1.0 - float(c.params_m) / caps.params_cap_m),
            max(0.0, 1.0 - float(c.macs_g) / caps.macs_cap_g),
        )
        for c in feasible
    ]

    front_points = set(filter_non_dominated_3d(points))

    pareto_candidates = [
        c for c in feasible
        if (
            float(c.val_acc),
            max(0.0, 1.0 - float(c.params_m) / caps.params_cap_m),
            max(0.0, 1.0 - float(c.macs_g) / caps.macs_cap_g),
        ) in front_points
    ]

    # Sort by accuracy descending
    pareto_candidates.sort(key=lambda c: c.val_acc or 0.0, reverse=True)
    return pareto_candidates[:limit]


def get_near_feasible_parents(
    store: CandidateStore,
    run_id: str,
    caps: BudgetConfig,
    min_val_acc: float,
    max_violation: float = 0.5,
) -> List[CandidateRecord]:
    """Find candidates eligible as parents for micro-surgery refinement.

    Requirements (Section 8.3 point 3):
    - Checkpointed (ckpt_ref is not None)
    - val_acc >= min_val_acc
    - Constraint violation v(x) <= max_violation (default 0.5)
    """
    candidates = store.get_run_candidates(run_id)
    parents: List[CandidateRecord] = []

    for c in candidates:
        if c.ckpt_ref is None or c.val_acc is None:
            continue
        if c.val_acc < min_val_acc:
            continue

        v = compute_violation_ratio(c.params_m, c.macs_g, c.peak_vram_mb, caps)
        if v <= max_violation:
            parents.append(c)

    # Sort by accuracy descending
    parents.sort(key=lambda c: c.val_acc or 0.0, reverse=True)
    return parents


def get_technique_statistics(
    store: CandidateStore,
    run_id: str,
    window: int = 15,
) -> Dict[str, Dict[str, float]]:
    """Aggregate success rate and median r_hv for each technique over last 15 attempts.

    Requirements (Section 8.3 point 4):
    - Success rate: share of attempts with success = 1.
    - Median r_hv: using medians (not means) avoids first-feasible candidate spike domination.
    """
    candidates = store.get_run_candidates(run_id)
    # Group by technique
    by_tech: Dict[str, List[CandidateRecord]] = {}
    for c in candidates:
        tech = c.technique
        if tech not in by_tech:
            by_tech[tech] = []
        by_tech[tech].append(c)

    stats: Dict[str, Dict[str, float]] = {}
    for tech, rows in by_tech.items():
        recent = rows[-window:]
        attempts = len(recent)
        if attempts == 0:
            continue

        success_count = sum(1 for r in recent if r.success)
        success_rate = float(success_count) / float(attempts)
        r_hv_vals = [r.r_hv for r in recent]
        median_r_hv = float(np.median(r_hv_vals)) if r_hv_vals else 0.0

        stats[tech] = {
            "attempts": attempts,
            "success_rate": round(success_rate, 3),
            "median_r_hv": round(median_r_hv, 6),
        }

    return stats


def calculate_rewards_and_update(
    store: CandidateStore,
    candidate: CandidateRecord,
    caps: BudgetConfig,
    min_val_acc: float,
) -> Tuple[float, float, bool]:
    """Compute r_hv, r_viol, and success flag, updating SQLite memory."""
    run_id = candidate.run_id
    prior_candidates = [
        c for c in store.get_run_candidates(run_id)
        if c.iteration < candidate.iteration
    ]

    # 1. Hypervolume before
    hv_before = calculate_normalized_hypervolume(
        [c.model_dump() for c in prior_candidates],
        params_cap_m=caps.params_cap_m,
        macs_cap_g=caps.macs_cap_g,
    )

    # 2. Hypervolume after
    current_pool = prior_candidates + [candidate]
    hv_after = calculate_normalized_hypervolume(
        [c.model_dump() for c in current_pool],
        params_cap_m=caps.params_cap_m,
        macs_cap_g=caps.macs_cap_g,
    )

    delta_hv = max(0.0, hv_after - hv_before)
    gpu_secs = max(1.0, candidate.own_gpu_seconds)
    r_hv = float(delta_hv / gpu_secs)

    # 3. Violation ratio reduction
    r_viol = 0.0
    if candidate.proposal_type == "micro" and candidate.parent_id:
        parent = store.get_candidate(candidate.parent_id)
        if parent:
            v_parent = compute_violation_ratio(parent.params_m, parent.macs_g, parent.peak_vram_mb, caps)
            v_child = compute_violation_ratio(candidate.params_m, candidate.macs_g, candidate.peak_vram_mb, caps)
            r_viol = max(0.0, v_parent - v_child)

    # 4. Success definition (Blueprint Section 8.1)
    val_acc = candidate.val_acc or 0.0
    success = (delta_hv > 0.0) or (r_viol > 0.0 and val_acc >= min_val_acc)

    # Update candidate in DB
    updates = {
        "r_hv": r_hv,
        "r_viol": r_viol,
        "success": 1 if success else 0,
    }
    store.update_candidate(candidate.candidate_id, updates)

    # 5. Refresh is_pareto flags for all candidates in run
    pareto_front = get_pareto_front(store, run_id, caps, limit=100)
    pareto_ids = set(c.candidate_id for c in pareto_front)

    for c in store.get_run_candidates(run_id):
        should_be_pareto = 1 if c.candidate_id in pareto_ids else 0
        if (1 if c.is_pareto else 0) != should_be_pareto:
            store.update_candidate(c.candidate_id, {"is_pareto": should_be_pareto})

    return r_hv, r_viol, success
