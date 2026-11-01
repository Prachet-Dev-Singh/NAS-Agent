"""
compiler.py  (Phase 1 scope: MacroProposal only)
------------------------------------------------
Deterministic compiler: MacroProposal → nn.Module

No LLM involved. Takes a validated MacroProposal JSON, instantiates
the PyTorch model using the primitive library, and returns it.

Architecture layout:
    NASModel
    ├── stem:   TransitionLayer-like stem (3x3 conv, stride 1)
    ├── stages: nn.ModuleList of StageModule
    │   └── Each StageModule:
    │       ├── transition: TransitionLayer (non-prunable)
    │       └── blocks:     nn.ModuleList of {ConvBlock|ResNetBlock|...}
    └── head:   nn.Sequential(AdaptiveAvgPool2d, Flatten, Linear → 10)

MicroProposal compilation (surgery on checkpoint) is implemented in
Phase 4 (lowrank.py / prune.py / widen.py).
"""

import copy
import torch
import torch.nn as nn

from nas_agent.state import (
    MacroProposal,
    MicroProposal,
    StageSpec,
    SVDLowRank,
    PruneInternal,
    WidenInternal,
    InsertBlock,
)
from nas_agent.lib.blocks import (
    TransitionLayer,
    build_block,
)
from nas_agent.lib.lowrank import apply_svd_lowrank
from nas_agent.lib.prune import apply_pruning
from nas_agent.lib.widen import apply_widening
from nas_agent.lib.insert import apply_insert_block


# ---------------------------------------------------------------------------
# Internal sub-modules
# ---------------------------------------------------------------------------

class StageModule(nn.Module):
    """One stage: a non-prunable TransitionLayer followed by depth blocks.

    Attributes exposed for surgery:
        transition : TransitionLayer  — widths locked; never pruned
        blocks     : nn.ModuleList    — each block is prunable on H
    """

    def __init__(
        self,
        in_channels: int,
        stage_spec: StageSpec,
        norm: str,
        activation: str,
    ) -> None:
        super().__init__()
        C = stage_spec.width
        self.transition = TransitionLayer(
            in_channels=in_channels,
            out_channels=C,
            stride=stage_spec.stride,
            norm=norm,  # type: ignore[arg-type]
            activation=activation,  # type: ignore[arg-type]
        )
        self.blocks = nn.ModuleList(
            [
                build_block(
                    block_type=stage_spec.block,
                    width=C,
                    expand=stage_spec.expand,
                    norm=norm,  # type: ignore[arg-type]
                    activation=activation,  # type: ignore[arg-type]
                    zero_init_out=False,
                )
                for _ in range(stage_spec.depth)
            ]
        )
        self._out_channels = C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transition(x)
        for block in self.blocks:
            x = block(x)
        return x

    @property
    def out_channels(self) -> int:
        return self._out_channels


class NASModel(nn.Module):
    """Top-level model compiled from a MacroProposal.

    Attributes
    ----------
    stem      : nn.Sequential  — 3x3 conv + norm + act
    stages    : nn.ModuleList  — list of StageModule
    head      : nn.Sequential  — GAP → Flatten → Linear(C_last, 10)
    spec_hash : str            — fingerprint of the source proposal
    """

    NUM_CLASSES: int = 10   # CIFAR-10

    def __init__(
        self,
        stem: nn.Sequential,
        stages: nn.ModuleList,
        head: nn.Sequential,
        spec_hash: str,
    ) -> None:
        super().__init__()
        self.stem = stem
        self.stages = stages
        self.head = head
        self.spec_hash = spec_hash

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Public compiler entry point
# ---------------------------------------------------------------------------

def _build_norm(name: str, channels: int) -> nn.Module:
    if name == "bn":
        return nn.BatchNorm2d(channels)
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _build_act(name: str) -> nn.Module:
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[name]()


def compile_macro(proposal: MacroProposal) -> NASModel:
    """Compile a MacroProposal into a trainable NASModel.

    Parameters
    ----------
    proposal : MacroProposal
        A fully validated MacroProposal (already passed Pydantic validation).

    Returns
    -------
    NASModel
        A randomly initialised PyTorch model ready for training.

    Raises
    ------
    ValueError
        If the proposal fails any structural invariant (should not occur
        after Pydantic validation, but guards against misuse).
    """
    norm = proposal.norm
    act_name = proposal.activation

    # ------------------------------------------------------------------
    # Stem: 3x3 conv  (input 3 × 32 × 32  →  C_stem × 32 × 32)
    # ------------------------------------------------------------------
    C_stem = proposal.stem_width
    stem = nn.Sequential(
        nn.Conv2d(3, C_stem, kernel_size=3, stride=1, padding=1, bias=False),
        _build_norm(norm, C_stem),
        _build_act(act_name),
    )

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    stages: list[StageModule] = []
    C_in = C_stem
    for stage_spec in proposal.stages:
        stage = StageModule(
            in_channels=C_in,
            stage_spec=stage_spec,
            norm=norm,
            activation=act_name,
        )
        stages.append(stage)
        C_in = stage.out_channels

    # ------------------------------------------------------------------
    # Head: Global Average Pool → Flatten → Linear(C_last, 10)
    # ------------------------------------------------------------------
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(C_in, NASModel.NUM_CLASSES, bias=True),
    )

    return NASModel(
        stem=stem,
        stages=nn.ModuleList(stages),
        head=head,
        spec_hash=proposal.spec_hash(),
    )


def compile_micro(proposal: MicroProposal, parent_model: NASModel) -> NASModel:
    """Compile a MicroProposal by applying surgical edits to a parent NASModel.

    Parameters
    ----------
    proposal : MicroProposal
        Validated micro proposal specifying operator, targets, and parameters.
    parent_model : NASModel
        Trained parent model to inherit weights and topology from.

    Returns
    -------
    NASModel
        Edited child model ready for fine-tuning.
    """
    child_model = copy.deepcopy(parent_model)
    op = proposal.op

    if isinstance(op, SVDLowRank):
        apply_svd_lowrank(
            child_model,
            target_stages=proposal.target_stages,
            rank_ratio=op.rank_ratio,
            which=op.which,
        )
    elif isinstance(op, PruneInternal):
        apply_pruning(
            child_model,
            target_stages=proposal.target_stages,
            sparsity=op.sparsity,
        )
    elif isinstance(op, WidenInternal):
        apply_widening(
            child_model,
            target_stages=proposal.target_stages,
            factor=op.factor,
        )
    elif isinstance(op, InsertBlock):
        norm = "bn"
        act = "silu"
        # Inherit norm and activation from parent stages if available
        if hasattr(parent_model.stem[1], "__class__"):
            norm = "bn" if "BatchNorm" in parent_model.stem[1].__class__.__name__ else "gn"
        apply_insert_block(
            child_model,
            target_stages=proposal.target_stages,
            block_type=op.block,
            norm=norm,
            activation=act,
        )
    else:
        raise TypeError(f"Unknown micro operator: {type(op)}")

    # Update child fingerprint
    child_model.spec_hash = proposal.spec_hash(parent_model.spec_hash)
    return child_model
