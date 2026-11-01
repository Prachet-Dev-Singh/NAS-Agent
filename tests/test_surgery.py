"""
test_surgery.py
---------------
Phase 4 acceptance test suite for surgical micro-operators (Blueprint Section 2.5 & 11).

Acceptance criteria verified:
1. Zero-init Widening Function Preservation: max |y_widened - y_orig| < 1e-4 across all 5 blocks.
2. Block Insertion Function Preservation: max |y_inserted - y_parent| < 1e-4.
3. SVD Low-Rank Full-Rank Equivalence: max |y_svd - y_orig| < 1e-4, plus parameter reduction.
4. Coupled Pruning Invariant: residual addition shapes never mismatch; stage boundary widths preserved.
5. Distillation Loss: computes combined CE + KL divergence and gradients flow to student.
6. Micro Compiler End-to-End: compile_micro correctly parses and applies all 4 operator schemas.
"""

from __future__ import annotations

import copy
import pytest
import torch
import torch.nn as nn

from nas_agent.lib.blocks import (
    AttnLiteBlock,
    ConvBlock,
    DWSepBlock,
    ResNetBlock,
    SSMBlock,
    build_block,
)
from nas_agent.lib.distill import DistillationLoss
from nas_agent.lib.insert import apply_insert_block, insert_block_into_stage
from nas_agent.lib.lowrank import (
    apply_svd_lowrank,
    factorize_conv2d,
    factorize_linear,
    svd_split_matrix,
)
from nas_agent.lib.prune import apply_pruning, prune_block
from nas_agent.lib.widen import apply_widening, widen_block
from nas_agent.nodes.compiler import compile_macro, compile_micro
from nas_agent.state import (
    InsertBlock,
    MacroProposal,
    MicroProposal,
    PruneInternal,
    SVDLowRank,
    StageSpec,
    WidenInternal,
)

ALL_BLOCKS = ["conv", "resnet", "dwsep", "attn_lite", "ssm"]


@pytest.fixture
def base_macro_proposal() -> MacroProposal:
    """Standard 3-stage parent architecture for surgery testing."""
    return MacroProposal(
        stem_width=32,
        stages=[
            StageSpec(block="conv", width=32, expand=2, depth=2, stride=1),
            StageSpec(block="resnet", width=48, expand=2, depth=2, stride=2),
            StageSpec(block="ssm", width=64, expand=2, depth=2, stride=2),
        ],
        norm="bn",
        activation="silu",
        rationale="Parent model for surgery tests",
    )


# ---------------------------------------------------------------------------
# 1. Zero-Init Widening Function Preservation Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_type", ALL_BLOCKS)
def test_widen_function_preservation(block_type: str):
    """Verify that zero-init widening produces max |y_widened - y_orig| < 1e-4."""
    width = 32
    expand = 2
    block = build_block(block_type=block_type, width=width, expand=expand, norm="bn", activation="silu")
    block.eval()

    Hs, Ws = (16, 16) if block_type in ("attn_lite", "ssm") else (32, 32)
    x = torch.randn(2, width, Hs, Ws)

    with torch.no_grad():
        y_orig = block(x)

    widen_block(block, factor=1.5)
    block.eval()
    assert block.hidden_dim > expand * width
    assert block.stage_width == width

    with torch.no_grad():
        y_widened = block(x)

    max_diff = (y_widened - y_orig).abs().max().item()
    assert max_diff < 1e-4, (
        f"Widening failed function preservation for {block_type}: max diff = {max_diff:.6f}"
    )


# ---------------------------------------------------------------------------
# 2. Block Insertion Function Preservation Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_type", ALL_BLOCKS)
def test_insert_block_function_preservation(base_macro_proposal, block_type: str):
    """Verify that inserting a zero-init block preserves output: max |y_child - y_parent| < 1e-4."""
    model = compile_macro(base_macro_proposal)
    model.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y_parent = model(x)

    # Insert into stage 1 (C=48, spatial=16x16)
    target_stage = model.stages[1]
    orig_blocks = len(target_stage.blocks)
    insert_block_into_stage(target_stage, block_type=block_type, expand=2, norm="bn", activation="silu")
    assert len(target_stage.blocks) == orig_blocks + 1

    model.eval()
    with torch.no_grad():
        y_child = model(x)

    max_diff = (y_child - y_parent).abs().max().item()
    assert max_diff < 1e-4, (
        f"Block insertion failed function preservation for {block_type}: max diff = {max_diff:.6f}"
    )


# ---------------------------------------------------------------------------
# 3. SVD Low-Rank Full-Rank Equivalence & Compression Tests
# ---------------------------------------------------------------------------

def test_svd_split_matrix_exactness():
    """Verify SVD split exact reconstruction at full rank."""
    W = torch.randn(64, 32)
    A, B = svd_split_matrix(W, rank=32)
    W_rec = B @ A
    diff = (W - W_rec).abs().max().item()
    assert diff < 1e-4


def test_factorize_linear_full_rank():
    """Verify factorize_linear reproduces output at rank_ratio=1.0."""
    linear = nn.Linear(32, 64, bias=True)
    linear.eval()
    x = torch.randn(4, 32)
    y_orig = linear(x)

    fact = factorize_linear(linear, rank_ratio=1.0)
    fact.eval()
    y_fact = fact(x)

    assert (y_fact - y_orig).abs().max().item() < 1e-4


def test_factorize_conv2d_full_rank():
    """Verify factorize_conv2d reproduces output at rank_ratio=1.0 for 1x1 and 3x3 convs."""
    # 1x1 Conv
    conv1 = nn.Conv2d(32, 64, kernel_size=1, bias=True)
    conv1.eval()
    x = torch.randn(2, 32, 8, 8)
    y1_orig = conv1(x)
    fact1 = factorize_conv2d(conv1, rank_ratio=1.0)
    fact1.eval()
    assert (fact1(x) - y1_orig).abs().max().item() < 1e-4

    # 3x3 Conv
    conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
    conv3.eval()
    y3_orig = conv3(x)
    fact3 = factorize_conv2d(conv3, rank_ratio=1.0)
    fact3.eval()
    assert (fact3(x) - y3_orig).abs().max().item() < 1e-4


def test_svd_parameter_reduction(base_macro_proposal):
    """Verify SVD low rank surgery reduces parameter count at low rank ratios."""
    model = compile_macro(base_macro_proposal)
    orig_params = sum(p.numel() for p in model.parameters())

    apply_svd_lowrank(model, target_stages=[0, 1], rank_ratio=0.25, which="both")
    new_params = sum(p.numel() for p in model.parameters())

    assert new_params < orig_params
    # Forward pass works cleanly
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


# ---------------------------------------------------------------------------
# 4. Coupled Pruning Invariant Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_type", ALL_BLOCKS)
def test_pruning_shape_invariants(block_type: str):
    """Verify that pruning preserves stage boundary widths and residual addition shapes."""
    width = 32
    expand = 4
    block = build_block(block_type=block_type, width=width, expand=expand, norm="bn", activation="silu")

    Hs, Ws = (16, 16) if block_type in ("attn_lite", "ssm") else (32, 32)
    x = torch.randn(2, width, Hs, Ws)

    orig_H = block.hidden_dim
    prune_block(block, sparsity=0.5)

    assert block.hidden_dim < orig_H
    assert block.stage_width == width  # Stage boundary untouched!

    # Forward pass must execute cleanly without residual shape mismatch
    y = block(x)
    assert y.shape == (2, width, Hs, Ws)


# ---------------------------------------------------------------------------
# 5. Distillation Loss Tests
# ---------------------------------------------------------------------------

def test_distillation_loss_gradient_flow():
    """Verify DistillationLoss computes correct scalar loss and propagates gradients."""
    distill = DistillationLoss(alpha=0.6, temperature=2.5)

    child_logits = torch.randn(4, 10, requires_grad=True)
    parent_logits = torch.randn(4, 10)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    loss = distill(child_logits, parent_logits, targets)
    assert loss.item() > 0.0
    loss.backward()

    assert child_logits.grad is not None
    assert (child_logits.grad.abs().sum() > 0).item()


# ---------------------------------------------------------------------------
# 6. Micro Compiler End-to-End Tests
# ---------------------------------------------------------------------------

def test_compile_micro_all_operators(base_macro_proposal):
    """Verify compile_micro applies each operator schema and updates spec_hash."""
    parent_model = compile_macro(base_macro_proposal)
    x = torch.randn(2, 3, 32, 32)

    # 1. SVD Low-Rank
    prop_svd = MicroProposal(
        parent_id="parent_0",
        op=SVDLowRank(technique="svd_lowrank", rank_ratio=0.5, which="both"),
        target_stages=[1],
        rationale="Compress stage 1 via SVD",
    )
    m_svd = compile_micro(prop_svd, parent_model)
    assert m_svd.spec_hash != parent_model.spec_hash
    assert m_svd(x).shape == (2, 10)

    # 2. Prune Internal
    prop_prune = MicroProposal(
        parent_id="parent_0",
        op=PruneInternal(technique="prune_internal", sparsity=0.3),
        target_stages=[0],
        rationale="Prune stage 0 internal channels",
    )
    m_prune = compile_micro(prop_prune, parent_model)
    assert m_prune.spec_hash != parent_model.spec_hash
    assert m_prune(x).shape == (2, 10)

    # 3. Widen Internal
    prop_widen = MicroProposal(
        parent_id="parent_0",
        op=WidenInternal(technique="widen_internal", factor=1.5),
        target_stages=[0],
        rationale="Widen stage 0 capacity",
    )
    m_widen = compile_micro(prop_widen, parent_model)
    assert m_widen.spec_hash != parent_model.spec_hash
    assert m_widen(x).shape == (2, 10)

    # 4. Insert Block
    prop_insert = MicroProposal(
        parent_id="parent_0",
        op=InsertBlock(technique="insert_block", block="conv"),
        target_stages=[2],
        rationale="Insert conv block into stage 2",
    )
    m_insert = compile_micro(prop_insert, parent_model)
    assert m_insert.spec_hash != parent_model.spec_hash
    assert m_insert(x).shape == (2, 10)
