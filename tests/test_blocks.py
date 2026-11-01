"""
test_blocks.py
--------------
Phase 1 acceptance test suite.

Coverage maps directly to the blueprint roadmap:
  ✓ All 5 blocks pass shape contracts     (C1)
  ✓ All 5 blocks pass gradient contracts  (C2)
  ✓ Zero-init out_proj is identity        (C3 — function preservation)
  ✓ Blocks work at every valid stage spatial size (32, 16, 8, 4)
  ✓ compile_macro works for 200 random MacroProposal specs
  ✓ Analytical counters are within 10% of torch param count
  ✓ Gate CPU smoke passes for all compiled models
  ✓ spec_hash is deterministic and unique

Run with:
    pip install -e ".[dev]"
    pytest tests/test_blocks.py -v
"""

from __future__ import annotations

import hashlib
import json
import random

import pytest
import torch

from nas_agent.state import (
    MacroProposal,
    StageSpec,
    EARLY_STAGE_BLOCKS,
)
from nas_agent.lib.blocks import (
    ConvBlock,
    ResNetBlock,
    DWSepBlock,
    AttnLiteBlock,
    SSMBlock,
    build_block,
    BLOCK_CLASSES,
)
from nas_agent.lib.counters import count_macro_arch, count_params_actual
from nas_agent.lib.contracts import (
    check_block_shape,
    check_gradients_flow,
    check_zero_init_out_proj,
    gate_cpu_smoke,
)
from nas_agent.nodes.compiler import compile_macro


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

ALL_BLOCK_TYPES = list(BLOCK_CLASSES.keys())
ALL_NORMS = ["bn", "gn"]
ALL_ACTS = ["relu", "gelu", "silu"]
SPATIAL_SIZES = [32, 16, 8, 4]


def _make_block(block_type: str, width: int = 32, expand: int = 2,
                norm: str = "bn", activation: str = "silu",
                zero_init: bool = False) -> torch.nn.Module:
    return build_block(block_type, width, expand, norm, activation,  # type: ignore
                       zero_init_out=zero_init)


def _make_minimal_proposal(
    block_type: str = "conv",
    stages: int = 3,
    norm: str = "bn",
    activation: str = "silu",
) -> MacroProposal:
    """A valid minimal MacroProposal for quick compilation tests."""
    stage_specs = []
    for i in range(stages):
        btype = block_type if (i > 0 or block_type in EARLY_STAGE_BLOCKS) else "conv"
        stage_specs.append(StageSpec(
            block=btype,  # type: ignore
            width=32,
            expand=2,
            depth=1,
            stride=1 if i == 0 else 2,
        ))
    return MacroProposal(
        stem_width=32,
        stages=stage_specs,
        norm=norm,  # type: ignore
        activation=activation,  # type: ignore
        rationale="test",
    )


def _random_proposal(rng: random.Random, seed: int | None = None) -> MacroProposal:
    """Sample a random valid MacroProposal for stress tests."""
    if seed is not None:
        rng.seed(seed)
    n_stages = rng.choice([3, 4])
    widths = [rng.choice([16, 32, 48, 64]) for _ in range(n_stages)]
    blocks_allowed = list(EARLY_STAGE_BLOCKS)
    late_blocks = list(BLOCK_CLASSES.keys())

    stages = []
    for i in range(n_stages):
        block = (rng.choice(blocks_allowed) if i == 0
                 else rng.choice(late_blocks))
        width = widths[i]
        expand = rng.choice([1, 2])
        depth = rng.randint(1, 3)
        stride = 1 if i == 0 else 2
        stages.append(StageSpec(
            block=block,  # type: ignore
            width=width,
            expand=expand,
            depth=depth,
            stride=stride,
        ))

    # Ensure total depth <= 16
    total_depth = sum(s.depth for s in stages)
    if total_depth > 16:
        # Scale down uniformly
        for s in stages:
            object.__setattr__(s, "depth", max(1, s.depth - 1))

    return MacroProposal(
        stem_width=rng.choice([16, 32, 48]),
        stages=stages,
        norm=rng.choice(["bn", "gn"]),  # type: ignore
        activation=rng.choice(["relu", "gelu", "silu"]),  # type: ignore
        rationale="random stress test",
    )


# ---------------------------------------------------------------------------
# C1: Shape contracts — all 5 blocks at all spatial sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_type", ALL_BLOCK_TYPES)
@pytest.mark.parametrize("spatial", SPATIAL_SIZES)
@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("expand", [1, 2])
def test_block_shape_contract(block_type: str, spatial: int, width: int, expand: int) -> None:
    """Block output must have the same shape as its input."""
    block = _make_block(block_type, width=width, expand=expand)
    check_block_shape(block, width=width, Hs=spatial, Ws=spatial)


@pytest.mark.parametrize("block_type", ALL_BLOCK_TYPES)
@pytest.mark.parametrize("norm", ALL_NORMS)
@pytest.mark.parametrize("activation", ALL_ACTS)
def test_block_norm_act_combinations(block_type: str, norm: str, activation: str) -> None:
    """Verify all norm × activation combinations construct and forward without error."""
    block = _make_block(block_type, width=32, expand=2, norm=norm, activation=activation)
    check_block_shape(block, width=32)


# ---------------------------------------------------------------------------
# C2: Gradient contracts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_type", ALL_BLOCK_TYPES)
def test_gradient_flow(block_type: str) -> None:
    """Gradients must reach both in_proj and out_proj."""
    block = _make_block(block_type, width=32, expand=2)
    check_gradients_flow(block, width=32)


@pytest.mark.parametrize("block_type", ALL_BLOCK_TYPES)
def test_gradient_flow_small_spatial(block_type: str) -> None:
    """Gradient flow at 4x4 spatial (smallest stage size)."""
    block = _make_block(block_type, width=32, expand=2)
    check_gradients_flow(block, width=32, Hs=4, Ws=4)


# ---------------------------------------------------------------------------
# C3: Function-preservation (zero-init out_proj)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_type", ALL_BLOCK_TYPES)
def test_zero_init_is_identity(block_type: str) -> None:
    """A block with zero-init out_proj must be a perfect identity at init."""
    block = _make_block(block_type, width=32, expand=2, zero_init=True)
    check_zero_init_out_proj(block, width=32, tol=1e-4)


@pytest.mark.parametrize("block_type", ALL_BLOCK_TYPES)
@pytest.mark.parametrize("spatial", [32, 8, 4])
def test_zero_init_is_identity_various_spatial(block_type: str, spatial: int) -> None:
    """Zero-init identity holds at all valid spatial sizes."""
    block = _make_block(block_type, width=32, expand=2, zero_init=True)
    check_zero_init_out_proj(block, width=32, Hs=spatial, Ws=spatial, tol=1e-4)


# ---------------------------------------------------------------------------
# Compilation: compile_macro produces a valid NASModel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_type", ALL_BLOCK_TYPES)
def test_compile_macro_single_block_type(block_type: str) -> None:
    """Compiling a homogeneous proposal completes and the model is callable."""
    proposal = _make_minimal_proposal(block_type=block_type)
    model = compile_macro(proposal)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 10), f"Expected (2, 10), got {tuple(y.shape)}"


@pytest.mark.parametrize("norm", ALL_NORMS)
@pytest.mark.parametrize("activation", ALL_ACTS)
def test_compile_macro_norm_act(norm: str, activation: str) -> None:
    """All norm × activation combinations produce a callable model."""
    proposal = _make_minimal_proposal(norm=norm, activation=activation)
    model = compile_macro(proposal)
    gate_cpu_smoke(model)


def test_compile_macro_4_stages() -> None:
    """A 4-stage proposal compiles and the head output is (B, 10)."""
    proposal = _make_minimal_proposal(stages=4)
    model = compile_macro(proposal)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 10)


def test_compile_200_random_proposals() -> None:
    """200 uniformly sampled random valid proposals all compile successfully."""
    rng = random.Random(42)
    failures = []
    for i in range(200):
        try:
            proposal = _random_proposal(rng, seed=i)
            model = compile_macro(proposal)
            x = torch.randn(1, 3, 32, 32)
            with torch.no_grad():
                _ = model(x)
        except Exception as e:
            failures.append((i, str(e)))

    assert not failures, (
        f"{len(failures)}/200 proposals failed compilation:\n"
        + "\n".join(f"  seed={s}: {e}" for s, e in failures[:5])
    )


# ---------------------------------------------------------------------------
# Gate: CPU smoke test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_type", ALL_BLOCK_TYPES)
def test_gate_cpu_smoke(block_type: str) -> None:
    """gate_cpu_smoke passes for every single-block-type compiled model."""
    proposal = _make_minimal_proposal(block_type=block_type)
    model = compile_macro(proposal)
    gate_cpu_smoke(model)  # raises RuntimeError on failure


# ---------------------------------------------------------------------------
# Counters: analytical params within 10% of actual torch count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_type", ALL_BLOCK_TYPES)
def test_param_counter_accuracy(block_type: str) -> None:
    """Analytical parameter count is within 10% of torch's actual count."""
    proposal = _make_minimal_proposal(block_type=block_type)
    stats = count_macro_arch(proposal)
    model = compile_macro(proposal)
    actual_m = count_params_actual(model)

    rel_err = abs(stats.params_m - actual_m) / max(actual_m, 1e-9)
    assert rel_err < 0.10, (
        f"Counter for '{block_type}' block: analytical={stats.params_m:.4f}M, "
        f"actual={actual_m:.4f}M, rel_err={rel_err:.2%}"
    )


def test_counter_macs_positive() -> None:
    """MACs count must be strictly positive for any valid proposal."""
    rng = random.Random(0)
    for i in range(20):
        proposal = _random_proposal(rng, seed=i)
        stats = count_macro_arch(proposal)
        assert stats.macs_g > 0, f"MACs is 0 for seed {i}: {stats}"
        assert stats.params_m > 0, f"Params is 0 for seed {i}: {stats}"


# ---------------------------------------------------------------------------
# spec_hash: determinism and uniqueness
# ---------------------------------------------------------------------------

def test_spec_hash_deterministic() -> None:
    """The same proposal always produces the same hash."""
    proposal = _make_minimal_proposal()
    h1 = proposal.spec_hash()
    h2 = proposal.spec_hash()
    assert h1 == h2


def test_spec_hash_unique_across_proposals() -> None:
    """Different proposals produce different hashes (no collision in 50 samples)."""
    rng = random.Random(99)
    hashes = set()
    for i in range(50):
        p = _random_proposal(rng, seed=i)
        h = p.spec_hash()
        assert h not in hashes, f"Hash collision at seed {i}"
        hashes.add(h)


def test_spec_hash_excludes_rationale() -> None:
    """Changing only rationale must not change spec_hash."""
    p1 = _make_minimal_proposal()
    # Build identical proposal with different rationale
    p2 = MacroProposal(
        stem_width=p1.stem_width,
        stages=p1.stages,
        norm=p1.norm,
        activation=p1.activation,
        rationale="completely different rationale",
    )
    assert p1.spec_hash() == p2.spec_hash()


# ---------------------------------------------------------------------------
# Pydantic schema validation
# ---------------------------------------------------------------------------

def test_macro_proposal_stage0_block_validation() -> None:
    """Stage 0 must not use attn_lite or ssm."""
    with pytest.raises(Exception):
        MacroProposal(
            stem_width=32,
            stages=[
                StageSpec(block="ssm", width=32, expand=2, depth=1, stride=1),
                StageSpec(block="conv", width=32, expand=2, depth=1, stride=2),
                StageSpec(block="conv", width=64, expand=2, depth=1, stride=2),
            ],
            norm="bn",
            activation="silu",
            rationale="invalid stage 0",
        )


def test_macro_proposal_too_few_stages() -> None:
    """Fewer than 3 stages must be rejected."""
    with pytest.raises(Exception):
        MacroProposal(
            stem_width=32,
            stages=[
                StageSpec(block="conv", width=32, expand=2, depth=1, stride=1),
                StageSpec(block="conv", width=64, expand=2, depth=1, stride=2),
            ],
            norm="bn",
            activation="silu",
            rationale="only 2 stages",
        )


def test_macro_proposal_total_depth_limit() -> None:
    """Total depth > 16 must be rejected."""
    with pytest.raises(Exception):
        MacroProposal(
            stem_width=32,
            stages=[
                StageSpec(block="conv", width=32, expand=2, depth=6, stride=1),
                StageSpec(block="conv", width=64, expand=2, depth=6, stride=2),
                StageSpec(block="conv", width=128, expand=2, depth=6, stride=2),
            ],
            norm="bn",
            activation="silu",
            rationale="too deep",
        )


def test_stage_width_must_be_mult_of_16() -> None:
    """Non-multiple-of-16 width must be rejected."""
    with pytest.raises(Exception):
        StageSpec(block="conv", width=30, expand=2, depth=1, stride=1)
