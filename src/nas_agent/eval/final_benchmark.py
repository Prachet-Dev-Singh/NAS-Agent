"""
final_benchmark.py
------------------
Post-hoc benchmarking on final Pareto-front candidates (Blueprint Section 6.3 & 11 Phase 8).

Key Functions:
1. Evaluates untouched CIFAR-10 test split (10,000 images).
2. Measures precise inference latency via CUDA events (50 warmup, 200 repetitions, BS=1).
3. Updates SQLite candidates table with latency and test metrics.
4. Produces markdown and structured tables for publications/analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from nas_agent.config import AppConfig, load_config
from nas_agent.harness.data import FastBatchLoader, get_cifar10_data
from nas_agent.harness.metrics import benchmark_latency
from nas_agent.memory.queries import get_pareto_front
from nas_agent.memory.store import CandidateStore
from nas_agent.nodes.compiler import compile_macro, compile_micro
from nas_agent.state import CandidateRecord, MacroProposal, MicroProposal


logger = logging.getLogger("nas_agent.final_benchmark")


def load_candidate_model(
    candidate: CandidateRecord,
    store: CandidateStore,
    device: str = "cuda",
) -> nn.Module:
    """Instantiate and restore candidate PyTorch model from spec_json and checkpoint."""
    if not candidate.spec_json:
        raise ValueError(f"Candidate {candidate.candidate_id} has no spec_json")

    spec_dict = json.loads(candidate.spec_json)
    kind = spec_dict.get("kind", "macro")

    if kind == "macro":
        proposal = MacroProposal.model_validate(spec_dict)
        model = compile_macro(proposal)
    elif kind == "micro":
        micro_proposal = MicroProposal.model_validate(spec_dict)
        parent = store.get_candidate(micro_proposal.parent_id)
        if not parent or not parent.spec_json:
            raise ValueError(f"Parent {micro_proposal.parent_id} not found in store")
        parent_macro = MacroProposal.model_validate(json.loads(parent.spec_json))
        parent_model = compile_macro(parent_macro)
        model = compile_micro(micro_proposal, parent_model)
    else:
        raise ValueError(f"Unknown proposal kind: {kind}")

    if candidate.ckpt_ref and os.path.exists(candidate.ckpt_ref):
        try:
            state_dict = torch.load(candidate.ckpt_ref, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info(f"Loaded weights from checkpoint: {candidate.ckpt_ref}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint for {candidate.candidate_id}: {e}")

    model.to(device)
    model.eval()
    return model


def evaluate_test_accuracy(
    model: nn.Module,
    data_dir: str = "data",
    device: str = "cuda",
    batch_size: int = 128,
    mock_data: bool = False,
) -> float:
    """Evaluate accuracy on the untouched CIFAR-10 test set."""
    actual_device = "cuda" if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    model = model.to(actual_device)
    model.eval()

    # Load CIFAR-10 test split
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    if mock_data:
        x = torch.randn(256, 3, 32, 32)
        y = torch.randint(0, 10, (256,))
        test_dataset = torch.utils.data.TensorDataset(x, y)
    else:
        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(actual_device), by.to(actual_device)
            logits = model(bx)
            preds = logits.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

    return float(correct / max(1, total))


def benchmark_final_pareto(
    db_path: str,
    run_id: str,
    device: str = "cuda",
    mock_data: bool = False,
    data_dir: str = "data",
) -> List[Dict[str, Any]]:
    """Benchmark all Pareto-front candidates for a given run and update database."""
    cfg = load_config()
    store = CandidateStore(db_path)
    pareto_candidates = get_pareto_front(store, run_id, cfg.budget)

    logger.info(f"Found {len(pareto_candidates)} Pareto-optimal candidates for run {run_id}")
    results = []

    for c in pareto_candidates:
        logger.info(f"Benchmarking candidate: {c.candidate_id} ({c.block_types})")
        model = load_candidate_model(c, store, device=device)

        # 1. Latency benchmark (BS=1, 50 warmup, 200 repetitions)
        lat_res = benchmark_latency(
            model=model,
            input_shape=(1, 3, 32, 32),
            device=device,
            warmup=50,
            repetitions=200,
        )

        # 2. Test accuracy
        test_acc = evaluate_test_accuracy(
            model=model,
            data_dir=data_dir,
            device=device,
            mock_data=mock_data,
        )

        # 3. Update candidate in database
        updates = {
            "latency_ms_median": lat_res["latency_ms_median"],
            "latency_ms_p95": lat_res["latency_ms_p95"],
            "latency_mode": lat_res["latency_mode"],
        }
        store.update_candidate(c.candidate_id, updates)

        res_dict = {
            "candidate_id": c.candidate_id,
            "val_acc": c.val_acc,
            "test_acc": round(test_acc, 4),
            "params_m": c.params_m,
            "macs_g": c.macs_g,
            "peak_vram_mb": c.peak_vram_mb,
            "latency_ms_median": lat_res["latency_ms_median"],
            "latency_ms_p95": lat_res["latency_ms_p95"],
            "latency_mode": lat_res["latency_mode"],
            "block_types": c.block_types,
        }
        results.append(res_dict)

    store.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="NAS-Agent Post-Hoc Pareto Benchmark")
    parser.add_argument("--db", type=str, required=True, help="Path to SQLite database")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID to benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    parser.add_argument("--mock-data", action="store_true", help="Use mock test data")
    parser.add_argument("--data-dir", type=str, default="data", help="CIFAR-10 data dir")

    args = parser.parse_args()

    results = benchmark_final_pareto(
        db_path=args.db,
        run_id=args.run_id,
        device=args.device,
        mock_data=args.mock_data,
        data_dir=args.data_dir,
    )

    # Print markdown table
    print("\n### Final Pareto Front Benchmark Results\n")
    print("| Candidate ID | Val Acc | Test Acc | Params (M) | MACs (G) | VRAM (MB) | Latency (ms) | Mode | Blocks |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['candidate_id']} | {r['val_acc']:.4f} | {r['test_acc']:.4f} | "
            f"{r['params_m']:.2f} | {r['macs_g']:.4f} | {r['peak_vram_mb']:.0f} | "
            f"{r['latency_ms_median']:.2f} | {r['latency_mode']} | {r['block_types']} |"
        )


if __name__ == "__main__":
    main()
