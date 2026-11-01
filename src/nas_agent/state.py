"""
state.py
--------
All Pydantic models and TypedDict state for the NAS-Agent search loop.

Sections:
  1. Proposal schemas  (MacroProposal, MicroProposal, Proposal union)
  2. Metrics / result models
  3. LangGraph AgentState
"""

from __future__ import annotations

import operator
import hashlib
import json
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# 1.  Search-space types
# ---------------------------------------------------------------------------

Block = Literal["conv", "dwsep", "resnet", "attn_lite", "ssm"]
EARLY_STAGE_BLOCKS: frozenset[str] = frozenset({"conv", "dwsep", "resnet"})


class StageSpec(BaseModel):
    """One stage in the macro layout.

    width  : output channel count for this stage (C), multiple of 16.
    expand : H = expand * width  (internal hidden dim of each block).
    depth  : number of blocks in the stage (>= 1).
    stride : spatial downsampling applied by the *transition* layer at the
             start of the stage (1 for stage 0, 2 for stages 1-3).
    """

    block: Block
    width: int = Field(..., ge=16, le=256)
    expand: int = Field(2, ge=1, le=4)
    depth: int = Field(..., ge=1, le=6)
    stride: int = Field(2, ge=1, le=2)

    @field_validator("width")
    @classmethod
    def _must_be_mult16(cls, v: int) -> int:
        if v % 16:
            raise ValueError(f"width must be a multiple of 16, got {v}")
        return v


class MacroProposal(BaseModel):
    """New topology, compiled from scratch.

    The LLM emits this when exploring a new region of the search space.
    Compiled → model → harness trains from random init.
    """

    kind: Literal["macro"] = "macro"
    stem_width: int = Field(32, ge=16, le=64)
    stages: list[StageSpec]
    norm: Literal["bn", "gn"] = "bn"
    activation: Literal["relu", "gelu", "silu"] = "silu"
    rationale: str = Field(..., max_length=400)

    @model_validator(mode="after")
    def _structural_checks(self) -> "MacroProposal":
        n = len(self.stages)
        if not 3 <= n <= 4:
            raise ValueError(f"MacroProposal must have 3-4 stages, got {n}")
        if self.stages[0].block not in EARLY_STAGE_BLOCKS:
            raise ValueError(
                f"Stage 0 must use conv/dwsep/resnet, got '{self.stages[0].block}'"
            )
        total_depth = sum(s.depth for s in self.stages)
        if total_depth > 16:
            raise ValueError(
                f"Total depth across all stages must be <= 16, got {total_depth}"
            )
        # Force stage 0 stride = 1 (stem already handles spatial resolution)
        if self.stages[0].stride != 1:
            raise ValueError("Stage 0 stride must be 1 (stem handles the first spatial scale)")
        return self

    def spec_hash(self) -> str:
        """Deterministic SHA-256 fingerprint of the architecture spec."""
        payload = self.model_dump(exclude={"rationale"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1b. Micro operator sub-models
# ---------------------------------------------------------------------------

class SVDLowRank(BaseModel):
    """Truncated-SVD factorisation of in_proj / out_proj weights."""

    technique: Literal["svd_lowrank"]
    rank_ratio: float = Field(..., ge=0.1, le=0.9)
    which: Literal["in_proj", "out_proj", "both"] = "both"


class PruneInternal(BaseModel):
    """L1-structured pruning of internal H channels.

    Coupled: in_proj out-channels, mixer params (where applicable),
    out_proj in-channels, and norms are all trimmed together.
    Stage boundary widths are never touched.
    """

    technique: Literal["prune_internal"]
    sparsity: float = Field(..., ge=0.1, le=0.6)


class WidenInternal(BaseModel):
    """Zero-init width extension of H channels.

    New out_proj weights are zero-init → identity-preserving at init.
    """

    technique: Literal["widen_internal"]
    factor: float = Field(..., ge=1.25, le=2.0)


class InsertBlock(BaseModel):
    """Append a new block with zero-init out_proj to a stage.

    Identity at initialisation: y = x + 0 = x.
    """

    technique: Literal["insert_block"]
    block: Block


Operator = Annotated[
    Union[SVDLowRank, PruneInternal, WidenInternal, InsertBlock],
    Field(discriminator="technique"),
]


class Distill(BaseModel):
    """Optional knowledge-distillation flag on micro fine-tune."""

    enabled: bool = False
    alpha: float = Field(0.5, ge=0.3, le=0.9)
    temperature: float = Field(2.0, ge=1.0, le=4.0)


class MicroProposal(BaseModel):
    """Surgery on an existing checkpoint, followed by short fine-tune.

    parent_id    : candidate_id of the checkpoint to inherit.
    op           : one of the four micro operators.
    target_stages: list of stage indices (0-indexed) the op touches.
    distill      : optional KD from parent as teacher.
    """

    kind: Literal["micro"] = "micro"
    parent_id: str
    op: Operator
    target_stages: list[int]
    distill: Distill = Field(default_factory=Distill)
    rationale: str = Field(..., max_length=400)

    @model_validator(mode="after")
    def _validate_stages(self) -> "MicroProposal":
        if not self.target_stages:
            raise ValueError("target_stages must list at least one stage index")
        if any(s < 0 or s > 3 for s in self.target_stages):
            raise ValueError("Stage indices must be in [0, 3]")
        return self

    def spec_hash(self, parent_spec_hash: str) -> str:
        payload = {
            "parent": parent_spec_hash,
            "op": self.op.model_dump(),
            "target_stages": sorted(self.target_stages),
            "distill": self.distill.model_dump(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# Discriminated union the LLM (or baselines) emit
Proposal = Annotated[
    Union[MacroProposal, MicroProposal],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# 2.  Metrics / result models
# ---------------------------------------------------------------------------

Status = Literal[
    "feasible",
    "infeasible",
    "rejected_static",
    "gate_fail",
    "duplicate",
    "invalid",
    "oom",
    "timeout",
    "nan",
    "crash",
]


class CandidateMetrics(BaseModel):
    """Parsed + validated view of the worker's metrics.json."""

    status: Status
    val_acc: Optional[float] = None
    params_m: Optional[float] = None
    macs_g: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    train_seconds: Optional[float] = None
    epochs_done: Optional[int] = None
    epoch_log: list[dict] = Field(default_factory=list)
    error_tail: Optional[str] = None
    spec_hash: Optional[str] = None
    config_hash: Optional[str] = None
    seed: Optional[int] = None
    env: dict = Field(default_factory=dict)


class CandidateRecord(BaseModel):
    """Full row written to the SQLite candidates table by the Record node."""

    candidate_id: str
    run_id: str
    arm: Literal["A", "B", "C", "D", "E", "G"]
    seed: int
    iteration: int
    config_hash: str
    spec_hash: str
    parent_id: Optional[str] = None
    proposal_type: Literal["macro", "micro"]
    technique: str                        # 'from_scratch' | operator name
    distill: bool = False
    block_types: str                      # comma-separated, e.g. "conv,resnet,ssm"
    params_m: Optional[float] = None
    macs_g: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    val_acc: Optional[float] = None
    latency_ms_median: Optional[float] = None
    latency_ms_p95: Optional[float] = None
    latency_mode: Optional[str] = None
    status: Status
    own_gpu_seconds: float = 0.0
    cumulative_gpu_seconds: float = 0.0
    r_hv: float = 0.0
    r_viol: float = 0.0
    success: bool = False
    fallback: bool = False
    is_pareto: bool = False
    ckpt_ref: Optional[str] = None
    error_tail: Optional[str] = None
    spec_json: Optional[str] = None
    rationale: Optional[str] = None


# ---------------------------------------------------------------------------
# 3.  LangGraph AgentState
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """Compact mutable state threaded through the LangGraph nodes.

    Blobs (model weights, full logs, spec files) are stored on disk / Modal
    Volume by reference. Only short summaries and scalars live here so that
    the graph state stays serialisable for the SQLite checkpointer.
    """

    run_id: str
    arm: Literal["A", "B", "C", "D", "E", "G"]
    seed: int
    config_hash: str

    # Budget tracking (GPU-seconds, not wall-clock or candidate count)
    gpu_seconds_used: float        # running sum of own_gpu_seconds
    gpu_seconds_budget: float      # 7200 for reported runs
    llm_calls_used: int

    # Loop counters
    iteration: int
    max_iterations: int
    consecutive_invalid: int

    # Current iteration transient data
    proposal: Optional[dict]       # raw JSON emitted by Optimizer / baseline
    candidate_id: Optional[str]
    last_result: Optional[dict]    # compact summary {status, metrics, reasons}

    # Append-only list of per-iteration compact summaries
    history: Annotated[list[dict], operator.add]
