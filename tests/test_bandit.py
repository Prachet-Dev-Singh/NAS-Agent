"""
test_bandit.py
--------------
Unit and integration tests for Search Baselines:
- Contextual Thompson-Sampling Bandit (Arm G)
- Uniform Random Search (Arm A)
- Regularized Aging Evolution (Arm B)
"""

import json
import os
import tempfile
import pytest

from nas_agent.config import BudgetConfig
from nas_agent.memory.store import CandidateStore
from nas_agent.search.bandit import BanditState, ContextualBanditProposer
from nas_agent.search.evolution import RegularizedEvolutionProposer
from nas_agent.search.operators import MACRO_OPERATORS, MICRO_OPERATORS, OPERATOR_NAMES
from nas_agent.search.random_search import RandomSearchProposer
from nas_agent.state import CandidateRecord, MacroProposal, MicroProposal


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


from nas_agent.calibration.budget import sample_random_macro_proposal
import random

def make_dummy_record(
    candidate_id: str,
    val_acc: float = 0.70,
    params_m: float = 3.0,
    macs_g: float = 0.5,
    status: str = "feasible",
    ckpt_ref: str = None,
) -> CandidateRecord:
    dummy_spec = sample_random_macro_proposal(random.Random(42))
    return CandidateRecord(
        candidate_id=candidate_id,
        run_id="run_1",
        arm="G",
        seed=42,
        iteration=1,
        config_hash="cfg_hash",
        spec_hash=f"hash_{candidate_id}",
        parent_id=None,
        proposal_type="macro",
        technique="from_scratch",
        block_types="conv",
        params_m=params_m,
        macs_g=macs_g,
        peak_vram_mb=1000.0,
        val_acc=val_acc,
        status=status,
        own_gpu_seconds=40.0,
        ckpt_ref=ckpt_ref,
        spec_json=dummy_spec.model_dump_json(),
    )


# ---------------------------------------------------------------------------
# 1. Arm G (Contextual Bandit) Tests
# ---------------------------------------------------------------------------

def test_bandit_posterior_and_cost_updates():
    """Verify discounted Beta-Bernoulli update rule (gamma = 0.95) and cost EMA."""
    bandit = ContextualBanditProposer(seed=42, discount=0.95, cost_ema_alpha=0.10)
    op = "svd_lowrank"
    ctx = "feasible"

    assert bandit.state.alpha[ctx][op] == 1.0
    assert bandit.state.beta[ctx][op] == 1.0
    assert bandit.state.expected_cost[op] == 45.0  # micro op default

    # 1. Update on success=True with observed cost = 50.0
    bandit.update_posterior(
        context=ctx,
        operator_name=op,
        success=True,
        observed_cost=50.0,
    )
    # alpha <- 0.95 * 1.0 + 1.0 = 1.95
    # beta  <- 0.95 * 1.0 + 0.0 = 0.95
    # cost  <- 0.90 * 45.0 + 0.10 * 50.0 = 45.5
    assert abs(bandit.state.alpha[ctx][op] - 1.95) < 1e-5
    assert abs(bandit.state.beta[ctx][op] - 0.95) < 1e-5
    assert abs(bandit.state.expected_cost[op] - 45.5) < 1e-5

    # 2. Update on success=False with observed cost = 40.0
    bandit.update_posterior(
        context=ctx,
        operator_name=op,
        success=False,
        observed_cost=40.0,
    )
    # alpha <- 0.95 * 1.95 + 0.0 = 1.8525
    # beta  <- 0.95 * 0.95 + 1.0 = 1.9025
    # cost  <- 0.90 * 45.5 + 0.10 * 40.0 = 40.95 + 4.0 = 44.95
    assert abs(bandit.state.alpha[ctx][op] - 1.8525) < 1e-5
    assert abs(bandit.state.beta[ctx][op] - 1.9025) < 1e-5
    assert abs(bandit.state.expected_cost[op] - 44.95) < 1e-5


def test_bandit_serialization_roundtrip():
    """Verify BanditState can be converted to/from JSON without drift."""
    bandit = ContextualBanditProposer(seed=123)
    bandit.update_posterior("feasible", "mutate_width", True, 85.0)
    bandit.update_posterior("infeasible", "prune_internal", False, 42.0)

    json_str = bandit.to_json()
    restored = ContextualBanditProposer.from_json(json_str, seed=123)

    assert restored.state.alpha == bandit.state.alpha
    assert restored.state.beta == bandit.state.beta
    assert restored.state.expected_cost == bandit.state.expected_cost


def test_bandit_propose_flow(temp_store, budget_caps):
    """Verify bandit proposal selection in feasible and infeasible contexts."""
    bandit = ContextualBanditProposer(seed=42)

    # 1. Propose with empty store -> defaults to context 'feasible'
    prop1, ctx1, op1 = bandit.propose(temp_store, "run_1", budget_caps, min_val_acc=0.60)
    assert ctx1 == "feasible"
    assert op1 in OPERATOR_NAMES
    assert isinstance(prop1, (MacroProposal, MicroProposal))

    # 2. Insert an infeasible candidate (macs = 2.0 > cap 1.0)
    temp_store.insert_candidate(make_dummy_record(
        "cand_inf", val_acc=0.72, params_m=3.0, macs_g=2.0, status="infeasible", ckpt_ref="/path/ckpt.pt"
    ))
    prop2, ctx2, op2 = bandit.propose(temp_store, "run_1", budget_caps, min_val_acc=0.60)
    assert ctx2 == "infeasible"
    assert op2 in OPERATOR_NAMES

    # 3. Insert a feasible high-acc candidate
    temp_store.insert_candidate(make_dummy_record(
        "cand_feas", val_acc=0.75, params_m=2.5, macs_g=0.4, status="feasible", ckpt_ref="/path/ckpt2.pt"
    ))
    prop3, ctx3, op3 = bandit.propose(temp_store, "run_1", budget_caps, min_val_acc=0.60)
    assert ctx3 == "feasible"
    assert op3 in OPERATOR_NAMES


# ---------------------------------------------------------------------------
# 2. Arm A (Random Search) Tests
# ---------------------------------------------------------------------------

def test_random_search_proposer(temp_store, budget_caps):
    """Verify Arm A proposes valid macro proposals and occasional micro surgeries when parents exist."""
    proposer = RandomSearchProposer(seed=42, micro_prob=0.0)
    # With micro_prob=0.0, strictly generates random MacroProposal
    prop = proposer.propose(temp_store, "run_1", budget_caps, min_val_acc=0.60)
    assert isinstance(prop, MacroProposal)
    assert len(prop.stages) == 3

    # With eligible parents and micro_prob=1.0, should propose micro surgery
    temp_store.insert_candidate(make_dummy_record(
        "p1", val_acc=0.70, params_m=3.0, macs_g=0.5, ckpt_ref="/path/p1.pt"
    ))
    proposer_micro = RandomSearchProposer(seed=42, micro_prob=1.0)
    prop_micro = proposer_micro.propose(temp_store, "run_1", budget_caps, min_val_acc=0.60)
    assert isinstance(prop_micro, MicroProposal)
    assert prop_micro.parent_id == "p1"
    assert prop_micro.op.technique in ("svd_lowrank", "prune_internal", "widen_internal", "insert_block")


# ---------------------------------------------------------------------------
# 3. Arm B (Regularized Evolution) Tests
# ---------------------------------------------------------------------------

def test_regularized_evolution_proposer(temp_store, budget_caps):
    """Verify Arm B starts with random initialization then uses tournament selection."""
    evo = RegularizedEvolutionProposer(population_size=3, tournament_size=2, seed=42)

    # Population < 3: emits random macro proposal
    prop_init = evo.propose(temp_store, "run_1", budget_caps, min_val_acc=0.60)
    assert isinstance(prop_init, MacroProposal)

    # Insert 3 candidates to fill population
    temp_store.insert_candidate(make_dummy_record("c1", val_acc=0.65, params_m=2.0, macs_g=0.3))
    temp_store.insert_candidate(make_dummy_record("c2", val_acc=0.75, params_m=2.5, macs_g=0.4))
    temp_store.insert_candidate(make_dummy_record("c3", val_acc=0.50, params_m=3.0, macs_g=0.5))

    # Population >= 3: selects tournament winner and mutates
    prop_mut = evo.propose(temp_store, "run_1", budget_caps, min_val_acc=0.60)
    if isinstance(prop_mut, MicroProposal):
        assert prop_mut.parent_id in ("c1", "c2", "c3")
    else:
        assert len(prop_mut.stages) == 3
