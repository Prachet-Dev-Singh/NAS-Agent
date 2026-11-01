"""
test_graph.py
-------------
Integration tests for the LangGraph search loop across multiple arms:
- Routing rules (validate -> record, gate -> record, record -> end/loop).
- End-to-end multi-iteration run on Arm A (Random Search).
- End-to-end multi-iteration run on Arm G (Thompson-Sampling Bandit).
- End-to-end multi-iteration run on Arm D (LLM Optimizer with Mock Gateway).
- Static rejection bypass (0 GPU seconds).
- Deduplication bypass (0 GPU seconds).
- Graph checkpointing and resumption.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional
import pytest

from langgraph.checkpoint.memory import MemorySaver

from nas_agent.config import AppConfig, BudgetConfig
from nas_agent.graph import (
    build_search_graph,
    route_after_gate,
    route_after_record,
    route_after_validate,
)
from nas_agent.harness.executor import Executor
from nas_agent.llm.gateway import LLMGateway
from nas_agent.memory.store import CandidateStore
from nas_agent.search.bandit import ContextualBanditProposer
from nas_agent.state import AgentState


class MockExecutor(Executor):
    """Fast deterministic executor simulating training outcomes without launching subprocesses."""

    def __init__(self, val_acc: float = 0.72, train_seconds: float = 20.0, peak_vram_mb: float = 800.0) -> None:
        self.val_acc = val_acc
        self.train_seconds = train_seconds
        self.peak_vram_mb = peak_vram_mb
        self.call_count = 0

    def run(
        self,
        candidate_id: str,
        spec_path: str,
        out_json: str,
        log_dir: str,
        timeout_sec: int = 300,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.call_count += 1
        return {
            "status": "ok",
            "val_acc": self.val_acc + (self.call_count * 0.01),
            "train_seconds": self.train_seconds,
            "peak_vram_mb": self.peak_vram_mb,
            "latency_ms_median": 1.5,
            "latency_ms_p95": 2.0,
            "latency_mode": "eager",
        }


from nas_agent.config import load_config


@pytest.fixture
def test_config() -> AppConfig:
    return load_config()


def test_routing_functions():
    """Verify routing decisions at conditional branch points."""
    # Validate router
    assert route_after_validate({"last_result": {"status": "invalid"}}) == "record"
    assert route_after_validate({"last_result": {"status": "duplicate"}}) == "record"
    assert route_after_validate({"last_result": {"status": "validated"}}) == "gate"

    # Gate router
    assert route_after_gate({"last_result": {"status": "rejected_static"}}) == "record"
    assert route_after_gate({"last_result": {"status": "gate_fail"}}) == "record"
    assert route_after_gate({"last_result": {"status": "gate_passed"}}) == "trainer"

    # Record router
    assert route_after_record({"gpu_seconds_used": 60.0, "gpu_seconds_budget": 50.0}) == "end"
    assert route_after_record({"llm_calls_used": 601}) == "end"
    assert route_after_record({"consecutive_invalid": 5}) == "end"
    assert route_after_record({"iteration": 10, "max_iterations": 5}) == "end"
    assert route_after_record({"gpu_seconds_used": 10.0, "gpu_seconds_budget": 50.0, "iteration": 2, "max_iterations": 10}) == "optimizer"


def test_graph_arm_a_run(test_config):
    """Verify Arm A (Random Search) loops and terminates upon reaching budget."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = CandidateStore(db_path)
        executor = MockExecutor(train_seconds=25.0)

        graph = build_search_graph(
            config=test_config,
            store=store,
            executor=executor,
        )

        initial_state: AgentState = {
            "run_id": "run_arm_a",
            "arm": "A",
            "seed": 42,
            "config_hash": "cfg_test",
            "gpu_seconds_used": 0.0,
            "gpu_seconds_budget": 50.0,  # 2 iterations at 25s each
            "llm_calls_used": 0,
            "iteration": 1,
            "consecutive_invalid": 0,
            "history": [],
        }

        config_thread = {"configurable": {"thread_id": "thread_arm_a"}}
        final_state = graph.invoke(initial_state, config_thread)

        assert final_state["gpu_seconds_used"] >= 50.0
        assert final_state["iteration"] >= 3
        assert executor.call_count >= 2

        candidates = store.get_run_candidates("run_arm_a")
        assert len(candidates) >= 2
        for c in candidates:
            assert c.status in ("feasible", "infeasible", "rejected_static")

        store.close()


def test_graph_arm_g_run(test_config):
    """Verify Arm G (Contextual Bandit) updates posteriors through graph execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = CandidateStore(db_path)
        executor = MockExecutor(train_seconds=30.0)
        bandit = ContextualBanditProposer(seed=42)

        graph = build_search_graph(
            config=test_config,
            store=store,
            executor=executor,
            bandit_proposer=bandit,
        )

        initial_state: AgentState = {
            "run_id": "run_arm_g",
            "arm": "G",
            "seed": 42,
            "config_hash": "cfg_test",
            "gpu_seconds_used": 0.0,
            "gpu_seconds_budget": 50.0,
            "llm_calls_used": 0,
            "iteration": 1,
            "consecutive_invalid": 0,
            "history": [],
        }

        config_thread = {"configurable": {"thread_id": "thread_arm_g"}}
        final_state = graph.invoke(initial_state, config_thread)

        assert final_state["gpu_seconds_used"] >= 50.0
        # Check that bandit posterior was updated
        alphas = [
            bandit.state.alpha[c][op]
            for c in ("feasible", "infeasible")
            for op in bandit.state.alpha[c]
        ]
        assert any(a != 1.0 for a in alphas)

        store.close()


def test_graph_arm_d_run(test_config):
    """Verify Arm D (LLM Agent with Mock Gateway) steps, tracks Pareto front, and terminates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache_dir = os.path.join(tmpdir, "llm_cache")
        store = CandidateStore(db_path)
        gateway = LLMGateway(provider="mock", cache_dir=cache_dir)
        executor = MockExecutor(train_seconds=25.0)

        graph = build_search_graph(
            config=test_config,
            store=store,
            gateway=gateway,
            executor=executor,
        )

        initial_state: AgentState = {
            "run_id": "run_arm_d",
            "arm": "D",
            "seed": 42,
            "config_hash": "cfg_test",
            "gpu_seconds_used": 0.0,
            "gpu_seconds_budget": 50.0,
            "llm_calls_used": 0,
            "iteration": 1,
            "consecutive_invalid": 0,
            "history": [],
        }

        config_thread = {"configurable": {"thread_id": "thread_arm_d"}, "recursion_limit": 100}
        final_state = graph.invoke(initial_state, config_thread)

        assert final_state["llm_calls_used"] >= 2
        assert final_state["gpu_seconds_used"] >= 50.0

        candidates = store.get_run_candidates("run_arm_d")
        assert len(candidates) >= 2
        pareto_flags = [c.is_pareto for c in candidates]
        assert any(pareto_flags)

        store.close()


def test_graph_static_rejection_bypass(test_config):
    """Verify that a candidate violating static caps is rejected at Gate with 0 GPU seconds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = CandidateStore(db_path)
        executor = MockExecutor(train_seconds=25.0)

        # Tiny cap forces static rejection
        strict_config = load_config()
        strict_config.budget.params_cap_m = 0.001

        graph = build_search_graph(
            config=strict_config,
            store=store,
            executor=executor,
        )

        initial_state: AgentState = {
            "run_id": "run_reject",
            "arm": "A",
            "seed": 42,
            "config_hash": "cfg_test",
            "gpu_seconds_used": 0.0,
            "gpu_seconds_budget": 50.0,
            "llm_calls_used": 0,
            "iteration": 1,
            "consecutive_invalid": 0,
            "max_iterations": 1,
            "history": [],
        }

        config_thread = {"configurable": {"thread_id": "thread_reject"}, "recursion_limit": 50}
        final_state = graph.invoke(initial_state, config_thread)

        assert executor.call_count == 0
        assert final_state["gpu_seconds_used"] == 0.0

        candidates = store.get_run_candidates("run_reject")
        assert len(candidates) >= 1
        assert candidates[0].status == "rejected_static"
        assert candidates[0].own_gpu_seconds == 0.0

        store.close()


def test_graph_deduplication_bypass(test_config):
    """Verify that proposing an existing spec_hash is caught at Validate and skips Gate/Trainer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = CandidateStore(db_path)
        executor = MockExecutor(train_seconds=25.0)

        from nas_agent.calibration.budget import sample_random_macro_proposal
        import random
        macro = sample_random_macro_proposal(random.Random(42))
        spec_hash = macro.spec_hash()

        from nas_agent.state import CandidateRecord
        store.insert_candidate(
            CandidateRecord(
                candidate_id="cand_existing",
                run_id="run_dedup",
                arm="A",
                seed=42,
                iteration=1,
                config_hash="cfg_test",
                spec_hash=spec_hash,
                proposal_type="macro",
                technique="from_scratch",
                block_types="conv",
                params_m=2.0,
                macs_g=0.3,
                peak_vram_mb=800.0,
                val_acc=0.74,
                status="feasible",
                own_gpu_seconds=25.0,
                spec_json=macro.model_dump_json(),
            )
        )

        class FixedGateway(LLMGateway):
            def _call_provider_raw(self, sys_p, user_p):
                return macro.model_dump_json(), 50, 50

        gw = FixedGateway(provider="mock", cache_dir=os.path.join(tmpdir, "cache"))

        graph = build_search_graph(
            config=test_config,
            store=store,
            gateway=gw,
            executor=executor,
        )

        initial_state: AgentState = {
            "run_id": "run_dedup",
            "arm": "D",
            "seed": 42,
            "config_hash": "cfg_test",
            "gpu_seconds_used": 0.0,
            "gpu_seconds_budget": 50.0,
            "llm_calls_used": 0,
            "iteration": 2,
            "consecutive_invalid": 0,
            "max_iterations": 2,
            "history": [],
        }

        config_thread = {"configurable": {"thread_id": "thread_dedup"}, "recursion_limit": 50}
        final_state = graph.invoke(initial_state, config_thread)

        assert executor.call_count == 0
        assert final_state["gpu_seconds_used"] == 0.0

        candidates = store.get_run_candidates("run_dedup")
        dup_candidate = [c for c in candidates if c.candidate_id != "cand_existing"][0]
        assert dup_candidate.status == "duplicate"
        assert dup_candidate.val_acc == 0.74
        assert dup_candidate.own_gpu_seconds == 0.0

        store.close()


def test_graph_checkpoint_resumption(test_config):
    """Verify that graph execution can pause and resume from checkpointer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = CandidateStore(db_path)
        executor = MockExecutor(train_seconds=10.0)
        checkpointer = MemorySaver()

        graph = build_search_graph(
            config=test_config,
            store=store,
            executor=executor,
            checkpointer=checkpointer,
        )

        config_thread = {"configurable": {"thread_id": "resume_thread"}, "recursion_limit": 50}

        # Step 1: Run with budget = 10s (terminates after 1 candidate)
        state_1: AgentState = {
            "run_id": "run_resume",
            "arm": "A",
            "seed": 42,
            "config_hash": "cfg_test",
            "gpu_seconds_used": 0.0,
            "gpu_seconds_budget": 10.0,
            "llm_calls_used": 0,
            "iteration": 1,
            "consecutive_invalid": 0,
            "history": [],
        }
        res_1 = graph.invoke(state_1, config_thread)
        assert res_1["gpu_seconds_used"] >= 10.0
        assert res_1["iteration"] == 2

        # Step 2: Resume with extended budget = 25s
        res_1["gpu_seconds_budget"] = 25.0
        res_2 = graph.invoke(res_1, config_thread)
        assert res_2["gpu_seconds_used"] >= 20.0
        assert res_2["iteration"] >= 3

        store.close()
