"""
graph.py
--------
LangGraph closed-loop architecture search assembly (Blueprint Section 5.1).

Defines the complete closed loop:
Optimizer -> Validate -> Gate -> Trainer -> Profiler -> Record -> Budget Check -> END / Loop
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from nas_agent.config import AppConfig, load_config
from nas_agent.harness.executor import Executor, LocalExecutor
from nas_agent.llm.gateway import LLMGateway
from nas_agent.memory.store import CandidateStore
from nas_agent.nodes.gate import GateNode
from nas_agent.nodes.optimizer import OptimizerNode
from nas_agent.nodes.profiler import ProfilerNode
from nas_agent.nodes.record import RecordNode
from nas_agent.nodes.trainer import TrainerNode
from nas_agent.nodes.validate import ValidateNode
from nas_agent.search.bandit import ContextualBanditProposer
from nas_agent.state import AgentState


logger = logging.getLogger(__name__)


def route_after_validate(state: AgentState) -> str:
    """Route invalid or duplicate candidates directly to Record, bypassing Gate and Trainer."""
    status = state.get("last_result", {}).get("status")
    if status in ("invalid", "duplicate"):
        return "record"
    return "gate"


def route_after_gate(state: AgentState) -> str:
    """Route statically rejected or smoke-failing candidates directly to Record, bypassing Trainer."""
    status = state.get("last_result", {}).get("status")
    if status in ("rejected_static", "gate_fail"):
        return "record"
    return "trainer"


def route_after_record(state: AgentState) -> str:
    """Check budget termination condition against GPU-seconds and LLM call caps."""
    gpu_seconds_used = state.get("gpu_seconds_used", 0.0)
    gpu_seconds_budget = state.get("gpu_seconds_budget", 7200.0)
    llm_calls_used = state.get("llm_calls_used", 0)
    consecutive_invalid = state.get("consecutive_invalid", 0)
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 1000)

    # 1. GPU budget exhausted (7,200 own GPU-seconds)
    if gpu_seconds_used >= gpu_seconds_budget:
        logger.info(f"Terminating search: GPU budget reached ({gpu_seconds_used:.1f}s >= {gpu_seconds_budget:.1f}s)")
        return "end"

    # 2. Hard LLM call cap (600 calls)
    if llm_calls_used >= 600:
        logger.info(f"Terminating search: LLM call cap reached ({llm_calls_used} >= 600)")
        return "end"

    # 3. Consecutive invalid threshold (5 failures in a row)
    if consecutive_invalid >= 5:
        logger.warning(f"Terminating search: 5 consecutive invalid proposals encountered.")
        return "end"

    # 4. Max iterations safeguard
    if iteration > max_iterations:
        logger.info(f"Terminating search: max iterations reached ({iteration} > {max_iterations})")
        return "end"

    return "optimizer"


def build_search_graph(
    config: Optional[AppConfig] = None,
    store: Optional[CandidateStore] = None,
    gateway: Optional[LLMGateway] = None,
    executor: Optional[Executor] = None,
    bandit_proposer: Optional[ContextualBanditProposer] = None,
    checkpointer: Optional[Any] = None,
) -> Any:
    """Assemble and compile the complete NAS-Agent LangGraph workflow."""
    cfg = config or load_config()
    db_store = store or CandidateStore()
    llm_gw = gateway or LLMGateway()
    exec_engine = executor or LocalExecutor()
    bandit = bandit_proposer or ContextualBanditProposer(seed=42)
    saver = checkpointer or MemorySaver()

    # Instantiate Nodes
    optimizer_node = OptimizerNode(config=cfg, store=db_store, gateway=llm_gw, bandit_proposer=bandit)
    validate_node = ValidateNode(store=db_store)
    gate_node = GateNode(config=cfg, store=db_store)
    trainer_node = TrainerNode(config=cfg, store=db_store, executor=exec_engine)
    profiler_node = ProfilerNode(config=cfg)
    record_node = RecordNode(config=cfg, store=db_store, bandit_proposer=bandit)

    # Build Graph
    builder = StateGraph(AgentState)

    builder.add_node("optimizer", optimizer_node)
    builder.add_node("validate", validate_node)
    builder.add_node("gate", gate_node)
    builder.add_node("trainer", trainer_node)
    builder.add_node("profiler", profiler_node)
    builder.add_node("record", record_node)

    # Entry point
    builder.set_entry_point("optimizer")

    # Edges
    builder.add_edge("optimizer", "validate")

    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "record": "record",
            "gate": "gate",
        },
    )

    builder.add_conditional_edges(
        "gate",
        route_after_gate,
        {
            "record": "record",
            "trainer": "trainer",
        },
    )

    builder.add_edge("trainer", "profiler")
    builder.add_edge("profiler", "record")

    builder.add_conditional_edges(
        "record",
        route_after_record,
        {
            "end": END,
            "optimizer": "optimizer",
        },
    )

    return builder.compile(checkpointer=saver)
