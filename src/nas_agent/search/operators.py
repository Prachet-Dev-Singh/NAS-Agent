"""
operators.py
------------
The eight mutation operators shared across Arms B, G, and search baselines (Blueprint Section 8.4).

Operators:
1. Macro:
   - mutate_block: change stage block type
   - mutate_width: mutate stage channel width (multiple of 16)
   - mutate_depth: increment or decrement stage depth (total depth <= 16)
   - mutate_expand: mutate internal expansion ratio H / C in {1, 2, 3, 4}
2. Micro:
   - svd_lowrank: SVD rank reduction on in_proj/out_proj
   - prune_internal: L1-norm coupled pruning of internal H channels
   - widen_internal: zero-init widening of internal H channels
   - insert_block: append zero-init identity block to stage
"""

from __future__ import annotations

import copy
import random
from typing import List, Optional, Tuple, Union

from nas_agent.calibration.budget import sample_random_macro_proposal
from nas_agent.state import (
    EARLY_STAGE_BLOCKS,
    Block,
    InsertBlock,
    MacroProposal,
    MicroProposal,
    PruneInternal,
    Proposal,
    SVDLowRank,
    StageSpec,
    WidenInternal,
)

OPERATOR_NAMES = [
    "mutate_block",
    "mutate_width",
    "mutate_depth",
    "mutate_expand",
    "svd_lowrank",
    "prune_internal",
    "widen_internal",
    "insert_block",
]

MACRO_OPERATORS = ["mutate_block", "mutate_width", "mutate_depth", "mutate_expand"]
MICRO_OPERATORS = ["svd_lowrank", "prune_internal", "widen_internal", "insert_block"]
ALL_BLOCK_TYPES: list[Block] = ["conv", "dwsep", "resnet", "attn_lite", "ssm"]
VALID_WIDTHS: list[int] = [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]


def mutate_macro_block(spec: MacroProposal, rng: random.Random) -> MacroProposal:
    """Mutate one stage's block type respecting placement rules."""
    new_spec = copy.deepcopy(spec)
    stage_idx = rng.randint(0, len(new_spec.stages) - 1)

    allowed = list(EARLY_STAGE_BLOCKS) if stage_idx == 0 else ALL_BLOCK_TYPES
    current_block = new_spec.stages[stage_idx].block
    candidates = [b for b in allowed if b != current_block]
    if not candidates:
        candidates = allowed

    new_spec.stages[stage_idx].block = rng.choice(candidates)
    new_spec.rationale = f"Mutated block in stage {stage_idx} from {current_block} to {new_spec.stages[stage_idx].block}"
    return new_spec


def mutate_macro_width(spec: MacroProposal, rng: random.Random) -> MacroProposal:
    """Mutate one stage's width to an adjacent valid multiple of 16."""
    new_spec = copy.deepcopy(spec)
    stage_idx = rng.randint(0, len(new_spec.stages) - 1)
    current_width = new_spec.stages[stage_idx].width

    idx = VALID_WIDTHS.index(current_width) if current_width in VALID_WIDTHS else 2
    delta = rng.choice([-1, 1])
    new_idx = max(0, min(len(VALID_WIDTHS) - 1, idx + delta))
    new_width = VALID_WIDTHS[new_idx]

    new_spec.stages[stage_idx].width = new_width
    new_spec.rationale = f"Mutated width in stage {stage_idx} from {current_width} to {new_width}"
    return new_spec


def mutate_macro_depth(spec: MacroProposal, rng: random.Random) -> MacroProposal:
    """Increment or decrement one stage's depth while keeping total depth <= 16."""
    new_spec = copy.deepcopy(spec)
    stage_idx = rng.randint(0, len(new_spec.stages) - 1)
    current_d = new_spec.stages[stage_idx].depth
    total_d = sum(s.depth for s in new_spec.stages)

    options = []
    if current_d > 1:
        options.append(-1)
    if current_d < 6 and total_d < 16:
        options.append(1)

    if not options:
        options = [-1] if current_d > 1 else [0]

    delta = rng.choice(options)
    new_spec.stages[stage_idx].depth = current_d + delta
    new_spec.rationale = f"Mutated depth in stage {stage_idx} from {current_d} to {current_d + delta}"
    return new_spec


def mutate_macro_expand(spec: MacroProposal, rng: random.Random) -> MacroProposal:
    """Mutate stage expand multiplier H / C in {1, 2, 3, 4}."""
    new_spec = copy.deepcopy(spec)
    stage_idx = rng.randint(0, len(new_spec.stages) - 1)
    current_expand = new_spec.stages[stage_idx].expand

    choices = [e for e in [1, 2, 3, 4] if e != current_expand]
    new_expand = rng.choice(choices)

    new_spec.stages[stage_idx].expand = new_expand
    new_spec.rationale = f"Mutated expand in stage {stage_idx} from {current_expand} to {new_expand}"
    return new_spec


def create_micro_proposal(
    parent_id: str,
    operator_name: str,
    num_stages: int,
    rng: random.Random,
) -> MicroProposal:
    """Construct a MicroProposal with randomly sampled hyperparameters for a selected operator."""
    target_stage = rng.randint(0, max(0, num_stages - 1))

    if operator_name == "svd_lowrank":
        rank_ratio = round(rng.uniform(0.3, 0.7), 2)
        which = rng.choice(["in_proj", "out_proj", "both"])
        op = SVDLowRank(technique="svd_lowrank", rank_ratio=rank_ratio, which=which)
        rationale = f"Applied SVD low-rank (ratio {rank_ratio}) on stage {target_stage}"

    elif operator_name == "prune_internal":
        sparsity = round(rng.uniform(0.2, 0.5), 2)
        op = PruneInternal(technique="prune_internal", sparsity=sparsity)
        rationale = f"Applied L1 pruning (sparsity {sparsity}) on stage {target_stage}"

    elif operator_name == "widen_internal":
        factor = round(rng.uniform(1.25, 1.75), 2)
        op = WidenInternal(technique="widen_internal", factor=factor)
        rationale = f"Applied zero-init widening (factor {factor}) on stage {target_stage}"

    elif operator_name == "insert_block":
        block = rng.choice(ALL_BLOCK_TYPES)
        op = InsertBlock(technique="insert_block", block=block)
        rationale = f"Inserted zero-init {block} block into stage {target_stage}"

    else:
        raise ValueError(f"Unknown micro operator: {operator_name}")

    return MicroProposal(
        parent_id=parent_id,
        op=op,
        target_stages=[target_stage],
        rationale=rationale,
    )


def apply_named_operator(
    operator_name: str,
    parent_spec: Optional[MacroProposal],
    parent_id: Optional[str],
    rng: random.Random,
) -> Proposal:
    """Dispatch to one of the 8 canonical search operators."""
    if operator_name in MACRO_OPERATORS:
        if parent_spec is None:
            return sample_random_macro_proposal(rng)

        if operator_name == "mutate_block":
            return mutate_macro_block(parent_spec, rng)
        elif operator_name == "mutate_width":
            return mutate_macro_width(parent_spec, rng)
        elif operator_name == "mutate_depth":
            return mutate_macro_depth(parent_spec, rng)
        elif operator_name == "mutate_expand":
            return mutate_macro_expand(parent_spec, rng)

    elif operator_name in MICRO_OPERATORS:
        if parent_id is None or parent_spec is None:
            # Fallback to random macro if no parent available
            return sample_random_macro_proposal(rng)

        num_stages = len(parent_spec.stages)
        return create_micro_proposal(parent_id, operator_name, num_stages, rng)

    raise ValueError(f"Unknown operator name: {operator_name}")
