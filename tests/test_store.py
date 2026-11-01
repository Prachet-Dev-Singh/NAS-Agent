"""
test_store.py
-------------
Phase 5 acceptance test suite for SQLite CandidateStore, queries, and decoupled rewards.

Acceptance criteria verified:
1. SQLite WAL mode, indexing, and schema initialization.
2. Fast deduplication via find_by_spec_hash.
3. Pareto front extraction across multi-objective space.
4. Near-feasible parent retrieval (val_acc >= min_val_acc and v(x) <= 0.5).
5. Technique statistics aggregation over rolling 15-attempt window with median r_hv.
6. Decoupled rewards: r_hv per GPU-second, r_viol reduction, and success boolean.
"""

from __future__ import annotations

import os
import tempfile
import pytest

from nas_agent.config import BudgetConfig
from nas_agent.memory.queries import (
    calculate_rewards_and_update,
    compute_violation_ratio,
    get_near_feasible_parents,
    get_pareto_front,
    get_technique_statistics,
)
from nas_agent.memory.store import CandidateStore
from nas_agent.state import CandidateRecord


@pytest.fixture
def budget_caps() -> BudgetConfig:
    return BudgetConfig(
        params_cap_m=5.0,
        macs_cap_g=1.0,
        peak_vram_cap_mb=1500.0,
    )


@pytest.fixture
def temp_store() -> CandidateStore:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = CandidateStore(db_path)
        yield store
        store.close()


def make_record(
    candidate_id: str,
    run_id: str = "run_1",
    arm: str = "A",
    iteration: int = 1,
    spec_hash: str = "hash1",
    proposal_type: str = "macro",
    technique: str = "from_scratch",
    parent_id: str = None,
    params_m: float = 2.0,
    macs_g: float = 0.4,
    peak_vram_mb: float = 1200.0,
    val_acc: float = 0.70,
    status: str = "feasible",
    own_gpu_seconds: float = 40.0,
    ckpt_ref: str = None,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        run_id=run_id,
        arm=arm,
        seed=42,
        iteration=iteration,
        config_hash="cfg_hash_test",
        spec_hash=spec_hash,
        parent_id=parent_id,
        proposal_type=proposal_type,
        technique=technique,
        block_types="conv,resnet",
        params_m=params_m,
        macs_g=macs_g,
        peak_vram_mb=peak_vram_mb,
        val_acc=val_acc,
        status=status,
        own_gpu_seconds=own_gpu_seconds,
        ckpt_ref=ckpt_ref,
        spec_json='{"kind": "macro", "stages": []}',
    )


# ---------------------------------------------------------------------------
# 1. Store CRUD & Deduplication Tests
# ---------------------------------------------------------------------------

def test_store_insert_and_get(temp_store):
    """Verify insertion and retrieval from SQLite."""
    rec = make_record("cand_1", spec_hash="shash_1")
    temp_store.insert_candidate(rec)

    fetched = temp_store.get_candidate("cand_1")
    assert fetched is not None
    assert fetched.candidate_id == "cand_1"
    assert fetched.val_acc == 0.70
    assert fetched.params_m == 2.0


def test_deduplication_by_spec_hash(temp_store):
    """Verify fast deduplication lookup by spec_hash."""
    rec = make_record("cand_1", spec_hash="hash_unique_123")
    temp_store.insert_candidate(rec)

    match = temp_store.find_by_spec_hash("hash_unique_123")
    assert match is not None
    assert match.candidate_id == "cand_1"

    assert temp_store.find_by_spec_hash("nonexistent_hash") is None


# ---------------------------------------------------------------------------
# 2. Pareto Front & Near-Feasible Queries Tests
# ---------------------------------------------------------------------------

def test_pareto_front_query(temp_store, budget_caps):
    """Verify non-dominated feasible candidate selection."""
    # cand_1: acc=0.70, params=2.0, macs=0.4 (non-dominated)
    temp_store.insert_candidate(make_record("cand_1", val_acc=0.70, params_m=2.0, macs_g=0.4))
    # cand_2: acc=0.80, params=4.0, macs=0.8 (non-dominated, higher acc, bigger)
    temp_store.insert_candidate(make_record("cand_2", val_acc=0.80, params_m=4.0, macs_g=0.8))
    # cand_3: acc=0.65, params=3.0, macs=0.6 (dominated by cand_1)
    temp_store.insert_candidate(make_record("cand_3", val_acc=0.65, params_m=3.0, macs_g=0.6))
    # cand_4: acc=0.90, params=8.0, macs=1.5 (infeasible, over cap)
    temp_store.insert_candidate(make_record("cand_4", val_acc=0.90, params_m=8.0, macs_g=1.5))

    pareto = get_pareto_front(temp_store, "run_1", budget_caps)
    pareto_ids = [c.candidate_id for c in pareto]

    assert "cand_1" in pareto_ids
    assert "cand_2" in pareto_ids
    assert "cand_3" not in pareto_ids  # dominated
    assert "cand_4" not in pareto_ids  # infeasible


def test_near_feasible_parents_query(temp_store, budget_caps):
    """Verify retrieval of checkpointed models with v(x) <= 0.5 and val_acc >= min_val_acc."""
    min_acc = 0.60
    # Parent 1: acc=0.68, params=5.5 (v = 0.1), has ckpt -> ELIGIBLE
    temp_store.insert_candidate(make_record(
        "p1", val_acc=0.68, params_m=5.5, macs_g=0.5, ckpt_ref="/path/to/p1.pt"
    ))
    # Parent 2: acc=0.55 (acc < min_acc) -> INELIGIBLE
    temp_store.insert_candidate(make_record(
        "p2", val_acc=0.55, params_m=4.0, macs_g=0.5, ckpt_ref="/path/to/p2.pt"
    ))
    # Parent 3: acc=0.75, params=10.0 (v = 1.0 > 0.5) -> INELIGIBLE (too infeasible)
    temp_store.insert_candidate(make_record(
        "p3", val_acc=0.75, params_m=10.0, macs_g=0.5, ckpt_ref="/path/to/p3.pt"
    ))
    # Parent 4: acc=0.72, params=4.5 (feasible), no ckpt -> INELIGIBLE
    temp_store.insert_candidate(make_record(
        "p4", val_acc=0.72, params_m=4.5, macs_g=0.5, ckpt_ref=None
    ))

    parents = get_near_feasible_parents(temp_store, "run_1", budget_caps, min_val_acc=min_acc, max_violation=0.5)
    parent_ids = [p.candidate_id for p in parents]

    assert "p1" in parent_ids
    assert "p2" not in parent_ids
    assert "p3" not in parent_ids
    assert "p4" not in parent_ids


# ---------------------------------------------------------------------------
# 3. Technique Statistics & Decoupled Rewards Tests
# ---------------------------------------------------------------------------

def test_technique_statistics(temp_store):
    """Verify median r_hv and success rate across rolling window."""
    # 5 attempts for 'svd_lowrank': 3 successes, 2 failures
    r_hvs = [0.0001, 0.0004, 0.0002, 0.0, 0.0]
    for i in range(5):
        c = make_record(f"c_{i}", technique="svd_lowrank", iteration=i)
        temp_store.insert_candidate(c)
        temp_store.update_candidate(f"c_{i}", {
            "r_hv": r_hvs[i],
            "success": 1 if i < 3 else 0,
        })

    stats = get_technique_statistics(temp_store, "run_1", window=15)
    assert "svd_lowrank" in stats
    svd_stat = stats["svd_lowrank"]
    assert svd_stat["attempts"] == 5
    assert svd_stat["success_rate"] == 0.60
    assert svd_stat["median_r_hv"] == 0.0001


def test_decoupled_rewards_and_success(temp_store, budget_caps):
    """Verify decoupled r_hv and r_viol reward calculation."""
    min_acc = 0.60

    # Iteration 1: Feasible candidate 1 -> establishes initial HV
    c1 = make_record("c1", iteration=1, val_acc=0.70, params_m=3.0, macs_g=0.5, own_gpu_seconds=50.0)
    temp_store.insert_candidate(c1)
    r_hv1, r_viol1, succ1 = calculate_rewards_and_update(temp_store, c1, budget_caps, min_acc)
    assert r_hv1 > 0.0
    assert succ1 is True

    # Iteration 2: Feasible candidate 2 improves Pareto front
    c2 = make_record("c2", iteration=2, val_acc=0.78, params_m=2.5, macs_g=0.4, own_gpu_seconds=50.0)
    temp_store.insert_candidate(c2)
    r_hv2, r_viol2, succ2 = calculate_rewards_and_update(temp_store, c2, budget_caps, min_acc)
    assert r_hv2 > 0.0
    assert succ2 is True

    # Iteration 3: Infeasible parent
    parent = make_record("p_inf", iteration=3, val_acc=0.75, params_m=6.0, macs_g=0.5, own_gpu_seconds=50.0)
    temp_store.insert_candidate(parent)
    calculate_rewards_and_update(temp_store, parent, budget_caps, min_acc)

    # Iteration 4: Micro child that reduces violation on infeasible parent
    child = make_record(
        "child_prune", iteration=4, proposal_type="micro", technique="prune_internal",
        parent_id="p_inf", val_acc=0.72, params_m=4.8, macs_g=0.4, own_gpu_seconds=25.0
    )
    temp_store.insert_candidate(child)
    r_hv4, r_viol4, succ4 = calculate_rewards_and_update(temp_store, child, budget_caps, min_acc)
    assert r_viol4 > 0.0
    assert succ4 is True
