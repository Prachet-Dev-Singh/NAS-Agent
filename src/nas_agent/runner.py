"""
runner.py
---------
Experiment runner CLI for NAS-Agent (Blueprint Section 11 Phase 7).

Executes a complete search run for a specified experimental arm and seed:
- Sets up run directories and SQLite structured experience database.
- Binds configuration with frozen config_hash.
- Executes closed-loop search on LangGraph until the 7,200 GPU-second budget or call limit is reached.
- Logs anytime progress and final Pareto front summary.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import yaml
from langgraph.checkpoint.memory import MemorySaver

from nas_agent.config import AppConfig, load_config
from nas_agent.graph import build_search_graph
from nas_agent.harness.executor import LocalExecutor
from nas_agent.llm.gateway import LLMGateway
from nas_agent.memory.queries import get_pareto_front
from nas_agent.memory.store import CandidateStore
from nas_agent.search.bandit import ContextualBanditProposer
from nas_agent.state import AgentState


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nas_agent.runner")


def load_arm_config(arm: str) -> Dict[str, Any]:
    """Load arm-specific configuration from configs/arms/<arm>.yaml."""
    path = os.path.join("configs", "arms", f"{arm.upper()}.yaml")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_experiment(
    arm: str = "D",
    seed: int = 42,
    run_id: Optional[str] = None,
    budget_seconds: float = 7200.0,
    max_iterations: int = 150,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    device: Optional[str] = None,
    mock_data: bool = False,
    config_path: str = "configs/v1.yaml",
) -> Dict[str, Any]:
    """Execute a single search run under equal GPU-seconds budget."""
    cfg = load_config(config_path)
    arm = arm.upper()

    if run_id is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{arm}_s{seed}_{timestamp}"

    # Output directory layout
    run_dir = os.path.join("experiments", "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    db_path = os.path.join(run_dir, "nas_agent.db")

    # Load arm specifics
    arm_cfg = load_arm_config(arm)
    llm_provider = provider or arm_cfg.get("provider", "mock")
    if llm_provider == "groq":
        llm_model = model or "llama-3.3-70b-versatile"
    else:
        llm_model = model or arm_cfg.get("model", "google/gemini-2.0-flash-exp:free")

    # If no API key is present in environment for non-mock provider, warn and fall back
    if llm_provider != "mock":
        key_var = f"{llm_provider.upper()}_API_KEY"
        if not os.environ.get(key_var):
            logger.warning(f"No {key_var} found in environment. Falling back to 'mock' provider.")
            llm_provider = "mock"

    logger.info("=" * 70)
    logger.info(f"STARTING NAS-AGENT EXPERIMENT: {run_id}")
    logger.info(f"  Arm: {arm} ({arm_cfg.get('name', 'Arm ' + arm)})")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  GPU Budget: {budget_seconds:.1f}s")
    logger.info(f"  LLM Provider: {llm_provider} (Model: {llm_model})")
    logger.info(f"  Database: {db_path}")
    logger.info(f"  Config Hash: {cfg.config_hash}")
    logger.info("=" * 70)

    # Initialize components
    store = CandidateStore(db_path)
    gateway = LLMGateway(
        provider=llm_provider,
        model=llm_model,
        cache_dir=os.path.join("experiments", "llm_cache"),
    )

    executor_extra_args = []
    if mock_data:
        executor_extra_args.append("--mock-data")

    executor = LocalExecutor(device=device, default_extra_args=executor_extra_args)
    bandit = ContextualBanditProposer(
        seed=seed,
        discount=arm_cfg.get("discount", 0.95),
        cost_ema_alpha=arm_cfg.get("cost_ema_alpha", 0.10),
    )

    checkpointer = MemorySaver()

    graph = build_search_graph(
        config=cfg,
        store=store,
        gateway=gateway,
        executor=executor,
        bandit_proposer=bandit,
        checkpointer=checkpointer,
    )

    # Check if resuming previous run
    existing_candidates = store.get_run_candidates(run_id)
    start_iter = len(existing_candidates) + 1
    initial_gpu_used = sum(c.own_gpu_seconds for c in existing_candidates)

    initial_state: AgentState = {
        "run_id": run_id,
        "arm": arm,
        "seed": seed,
        "config_hash": cfg.config_hash or "frozen",
        "gpu_seconds_used": initial_gpu_used,
        "gpu_seconds_budget": budget_seconds,
        "llm_calls_used": 0,
        "iteration": start_iter,
        "max_iterations": max_iterations,
        "consecutive_invalid": 0,
        "history": [],
    }

    config_thread = {
        "configurable": {"thread_id": run_id},
        "recursion_limit": max(500, max_iterations * 5),
    }

    try:
        final_state = graph.invoke(initial_state, config_thread)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Cleaning up and finalizing database...")
        final_state = initial_state

    # Summary report
    pareto_front = get_pareto_front(store, run_id, cfg.budget)
    all_evaluated = store.get_run_candidates(run_id)
    feasible = [c for c in all_evaluated if c.status == "feasible"]
    best_acc = max((c.val_acc for c in feasible if c.val_acc is not None), default=0.0)

    logger.info("=" * 70)
    logger.info(f"EXPERIMENT FINISHED: {run_id}")
    logger.info(f"  Total Candidates Evaluated: {len(all_evaluated)}")
    logger.info(f"  Feasible Candidates: {len(feasible)}")
    logger.info(f"  Pareto Front Size: {len(pareto_front)}")
    logger.info(f"  Best Feasible Validation Accuracy: {best_acc * 100:.2f}%")
    logger.info(f"  Total GPU Seconds Consumed: {final_state.get('gpu_seconds_used', 0.0):.1f}s")
    logger.info("=" * 70)

    store.close()
    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(description="NAS-Agent Experiment Runner")
    parser.add_argument("--arm", type=str, default="D", choices=["A", "B", "C", "D", "E", "G"], help="Experimental arm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run ID")
    parser.add_argument("--budget-seconds", type=float, default=7200.0, help="Total GPU seconds budget")
    parser.add_argument("--max-iterations", type=int, default=120, help="Max iterations cutoff")
    parser.add_argument("--provider", type=str, default=None, choices=["mock", "openrouter", "openai", "gemini", "groq"], help="LLM provider (mock, openrouter, openai, gemini, groq)")
    parser.add_argument("--model", type=str, default=None, help="Model identifier")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--mock-data", action="store_true", help="Use synthetic data for fast local testing")
    parser.add_argument("--config", type=str, default="configs/v1.yaml", help="Path to frozen config YAML")

    args = parser.parse_args()

    run_experiment(
        arm=args.arm,
        seed=args.seed,
        run_id=args.run_id,
        budget_seconds=args.budget_seconds,
        max_iterations=args.max_iterations,
        provider=args.provider,
        model=args.model,
        device=args.device,
        mock_data=args.mock_data,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
