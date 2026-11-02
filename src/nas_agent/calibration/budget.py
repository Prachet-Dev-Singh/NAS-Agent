"""
budget.py
---------
Budget sampling and resource constraint calibration (Phase 3, Step 1).

Protocol (Blueprint Section 7.1 & Study Guide Part 3.8):
- Samples 200 random valid macro specs from the search space on CPU.
- Computes analytical parameter counts (params_m) and multiply-accumulate operations (macs_g).
- Calculates the joint CDF and sets binding caps so that 30% to 50% of architectures
  are infeasible (violating at least one constraint).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from nas_agent.lib.counters import count_macro_arch
from nas_agent.state import EARLY_STAGE_BLOCKS, Block, MacroProposal, StageSpec

ALL_BLOCKS: list[Block] = ["conv", "dwsep", "resnet", "attn_lite", "ssm"]
VALID_WIDTHS: list[int] = [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]


def sample_random_macro_proposal(rng: random.Random) -> MacroProposal:
    """Generate a single random structurally valid MacroProposal within search space bounds."""
    num_stages = rng.choice([3, 4])
    stem_width = rng.choice([16, 32, 48, 64])
    norm = rng.choice(["bn", "gn"])
    activation = rng.choice(["silu", "gelu", "relu"])

    # Stage 0 must use an early-stage block (conv, dwsep, resnet) and stride 1
    s0_block = rng.choice(list(EARLY_STAGE_BLOCKS))
    s0_width = rng.choice(VALID_WIDTHS[:6])  # 32 to 112
    s0_depth = rng.randint(1, 3)

    stages: list[StageSpec] = [
        StageSpec(block=s0_block, width=s0_width, expand=rng.choice([1, 2, 3, 4]), depth=s0_depth, stride=1)
    ]

    remaining_depth_budget = 16 - s0_depth
    current_width = s0_width

    for s_idx in range(1, num_stages):
        # Stage 1 resolution is 16x16: use conv, dwsep, resnet, or attn_lite
        # Stage 2+ resolution is downsampled (8x8, 4x4): all blocks including ssm are allowed
        if s_idx < 2:
            allowed_blocks = ["conv", "dwsep", "resnet", "attn_lite"]
        else:
            allowed_blocks = list(ALL_BLOCKS)
        block = rng.choice(allowed_blocks)
        # Ensure monotonic or non-decreasing width trend typical of vision networks
        avail_widths = [w for w in VALID_WIDTHS if w >= current_width]
        if not avail_widths:
            avail_widths = [current_width]
        width = rng.choice(avail_widths[:4])
        current_width = width

        # Ensure total depth across all stages <= 16
        max_d = min(5, remaining_depth_budget - (num_stages - 1 - s_idx))
        max_d = max(1, max_d)
        depth = rng.randint(1, max_d)

        expand = rng.choice([1, 2, 3, 4])
        if block == "ssm":
            depth = min(depth, 2)
            expand = min(expand, 2)
        elif block == "attn_lite":
            depth = min(depth, 3)

        remaining_depth_budget -= depth
        stride = 2  # Standard downsampling transition

        stages.append(
            StageSpec(block=block, width=width, expand=expand, depth=depth, stride=stride)
        )

    return MacroProposal(
        stem_width=stem_width,
        stages=stages,
        norm=norm,
        activation=activation,
        rationale="Calibrated budget random sample architecture",
    )


@dataclass
class BudgetSamplingResult:
    """Outcomes from sampling the architecture space."""
    num_samples: int
    params_m: list[float]
    macs_g: list[float]
    params_cap_m: float
    macs_cap_g: float
    peak_vram_cap_mb: float
    infeasible_rate: float
    percentiles: dict[str, float]


def sample_budget(
    num_samples: int = 200,
    seed: int = 42,
    target_infeasible_ratio: float = 0.40,
) -> BudgetSamplingResult:
    """Sample random valid specs and determine binding constraint caps.

    Parameters
    ----------
    num_samples : int
        Number of valid architectures to sample (default 200 per Section 7.1).
    seed : int
        Seed for deterministic architecture generation.
    target_infeasible_ratio : float
        Target fraction of the distribution to exceed caps (target 30% - 50%, default 40%).

    Returns
    -------
    BudgetSamplingResult
        Calculated caps and empirical distribution metrics.
    """
    rng = random.Random(seed)
    params_list: list[float] = []
    macs_list: list[float] = []

    for _ in range(num_samples):
        spec = sample_random_macro_proposal(rng)
        stats = count_macro_arch(spec)
        params_list.append(stats.params_m)
        macs_list.append(stats.macs_g)

    params_arr = np.array(params_list)
    macs_arr = np.array(macs_list)

    # Compute empirical percentiles
    p_median = float(np.median(params_arr))
    p_70 = float(np.percentile(params_arr, 70))
    p_80 = float(np.percentile(params_arr, 80))

    m_median = float(np.median(macs_arr))
    m_70 = float(np.percentile(macs_arr, 70))
    m_80 = float(np.percentile(macs_arr, 80))

    # Grid search for joint percentile cap that yields target infeasibility in [0.30, 0.50]
    best_cap_p = p_70
    best_cap_m = m_70
    best_diff = float("inf")
    best_rate = 0.0

    # Search percentiles between 60th and 85th
    for pct in np.linspace(60, 85, 26):
        cp = float(np.percentile(params_arr, pct))
        cm = float(np.percentile(macs_arr, pct))
        infeasible_count = np.sum((params_arr > cp) | (macs_arr > cm))
        rate = float(infeasible_count) / float(num_samples)

        diff = abs(rate - target_infeasible_ratio)
        if diff < best_diff and 0.30 <= rate <= 0.50:
            best_diff = diff
            best_cap_p = cp
            best_cap_m = cm
            best_rate = rate

    # If exact range wasn't hit, use the 70th percentile default
    if best_rate == 0.0:
        best_cap_p = float(np.percentile(params_arr, 70))
        best_cap_m = float(np.percentile(macs_arr, 70))
        best_rate = float(np.sum((params_arr > best_cap_p) | (macs_arr > best_cap_m))) / float(num_samples)

    # Peak VRAM cap (derived from CIFAR-10 batch-size 128 proxy runs, typically capped around 1500 MB)
    peak_vram_cap_mb = 1500.0

    percentiles = {
        "params_p50": round(p_median, 3),
        "params_p70": round(p_70, 3),
        "params_p80": round(p_80, 3),
        "macs_p50": round(m_median, 4),
        "macs_p70": round(m_70, 4),
        "macs_p80": round(m_80, 4),
    }

    return BudgetSamplingResult(
        num_samples=num_samples,
        params_m=params_list,
        macs_g=macs_list,
        params_cap_m=round(best_cap_p, 3),
        macs_cap_g=round(best_cap_m, 4),
        peak_vram_cap_mb=peak_vram_cap_mb,
        infeasible_rate=round(best_rate, 3),
        percentiles=percentiles,
    )
