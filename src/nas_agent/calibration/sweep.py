"""
sweep.py
--------
Learning rate multiplier sweep and noise floor estimation (Phase 3, Steps 2 & 3).

Protocol (Blueprint Section 7.2 & 7.3):
1. Block Family LR Multipliers:
   - For each block family (conv, resnet, dwsep, attn_lite, ssm), train a homogeneous
     reference network across multipliers {0.5, 1.0, 2.0} x base_lr.
   - Select and freeze the best multiplier for each block family.
2. Noise Floor (sigma_acc):
   - Train a fixed reference architecture across 5 seeds.
   - Compute the standard deviation of validation accuracy (sigma_acc).
   - In search prompts, any accuracy difference below 2 * sigma_acc is treated as noise.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from nas_agent.state import Block, MacroProposal, StageSpec


def build_reference_network(block_type: Block, stem_width: int = 32) -> MacroProposal:
    """Build a standard homogeneous 3-stage reference architecture for calibration sweeps.

    Note: Stage 0 uses a conv-compatible block (conv for attn/ssm) to respect early-stage contracts.
    """
    s0_block: Block = "conv" if block_type in ("attn_lite", "ssm") else block_type

    return MacroProposal(
        stem_width=stem_width,
        stages=[
            StageSpec(block=s0_block, width=32, expand=2, depth=2, stride=1),
            StageSpec(block=block_type, width=64, expand=2, depth=2, stride=2),
            StageSpec(block=block_type, width=128, expand=2, depth=2, stride=2),
        ],
        norm="bn",
        activation="silu",
        rationale=f"Calibration reference architecture for {block_type}",
    )


def compute_noise_floor(accuracies: List[float]) -> float:
    """Calculate the empirical accuracy standard deviation across random seeds."""
    if len(accuracies) < 2:
        raise ValueError(f"At least 2 seed runs required to compute noise floor, got {len(accuracies)}")
    return float(np.std(accuracies, ddof=1))


def select_best_multipliers(
    sweep_results: Dict[str, Dict[float, float]],
) -> Dict[str, float]:
    """Select the multiplier yielding the highest validation accuracy for each block family.

    Parameters
    ----------
    sweep_results : Dict[block_family, Dict[multiplier, val_acc]]

    Returns
    -------
    Dict[block_family, best_multiplier]
    """
    best_multipliers: Dict[str, float] = {}

    for block_type, mult_accs in sweep_results.items():
        if not mult_accs:
            best_multipliers[block_type] = 1.0
            continue
        best_mult = max(mult_accs.keys(), key=lambda m: mult_accs[m])
        best_multipliers[block_type] = float(best_mult)

    return best_multipliers
