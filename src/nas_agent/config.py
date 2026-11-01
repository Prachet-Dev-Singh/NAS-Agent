"""
config.py
---------
Configuration schema, validation, serialization, and cryptographic freezing for NAS-Agent.

Design Contract (Blueprint Section 6.2, 7 & Appendix 13):
- All search loop parameters and calibration outcomes are consolidated in a typed config.
- A deterministic 16-character SHA-256 fingerprint (`config_hash`) is computed from the
  canonical settings and stamped onto every row in SQLite to prevent silent configuration drift.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, model_validator


class BudgetConfig(BaseModel):
    """Resource constraints established during Phase 3 budget sampling."""
    params_cap_m: float = Field(..., gt=0.0, description="Max model parameters in Millions")
    macs_cap_g: float = Field(..., gt=0.0, description="Max multiply-accumulate operations in Giga-MACs")
    peak_vram_cap_mb: float = Field(..., gt=0.0, description="Max peak allocated VRAM in Megabytes")


class HarnessConfig(BaseModel):
    """Fixed, subprocess-isolated training harness hyperparameters."""
    base_lr: float = Field(2e-3, gt=0.0)
    min_lr: float = Field(1e-5, ge=0.0)
    weight_decay: float = Field(1e-2, ge=0.0)
    betas: Tuple[float, float] = (0.9, 0.98)
    macro_epochs: int = Field(8, ge=1)
    macro_warmup_epochs: int = Field(2, ge=0)
    macro_subset_size: int = Field(10000, ge=100)
    micro_epochs: int = Field(4, ge=1)
    micro_warmup_epochs: int = Field(1, ge=0)
    micro_lr_factor: float = Field(0.3, gt=0.0)
    batch_size: int = Field(128, ge=1)
    grad_clip: float = Field(1.0, gt=0.0)
    val_size: int = Field(5000, ge=100)
    dataset: str = "cifar10"
    num_classes: int = 10


class CalibrationConfig(BaseModel):
    """Empirically calibrated constants frozen in Phase 3."""
    min_val_acc: float = Field(..., ge=0.0, le=1.0, description="25th percentile of feasible proxy accuracy")
    tau: float = Field(..., ge=0.0, le=1.0, description="75th percentile of feasible proxy accuracy")
    sigma_acc: float = Field(..., ge=0.0, description="Noise floor standard deviation across seeds")
    block_multipliers: Dict[str, float] = Field(
        default_factory=lambda: {
            "conv": 1.0,
            "resnet": 1.0,
            "dwsep": 1.0,
            "attn_lite": 1.0,
            "ssm": 1.0,
        }
    )


class AppConfig(BaseModel):
    """Root configuration for NAS-Agent runs."""
    version: str = "v1.1"
    budget: BudgetConfig
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    calibration: CalibrationConfig
    config_hash: Optional[str] = None

    def canonical_dict(self) -> dict:
        """Return a sorted, normalized dictionary representation excluding config_hash."""
        data = self.model_dump(mode="json", exclude={"config_hash"})
        return data

    def compute_hash(self) -> str:
        """Compute deterministic 16-character SHA-256 fingerprint."""
        raw_json = json.dumps(self.canonical_dict(), sort_keys=True)
        return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()[:16]


def compute_config_hash(cfg: AppConfig) -> str:
    """Compute the deterministic configuration fingerprint."""
    return cfg.compute_hash()


def load_config(path: str = "configs/v1.yaml", verify_hash: bool = True) -> AppConfig:
    """Load and validate an AppConfig from a YAML file.

    Parameters
    ----------
    path : str
        Path to YAML config file.
    verify_hash : bool
        If True and config_hash is present in file, verifies that it matches the content.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cfg = AppConfig.model_validate(data)

    if verify_hash and cfg.config_hash is not None:
        expected = cfg.compute_hash()
        if cfg.config_hash != expected:
            raise ValueError(
                f"Config hash mismatch in {path}: expected {expected}, found {cfg.config_hash}. "
                "The configuration has been modified after freeze!"
            )

    return cfg


def save_config(cfg: AppConfig, path: str) -> str:
    """Save configuration to YAML file with embedded cryptographic fingerprint."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Compute and stamp hash
    cfg_hash = cfg.compute_hash()
    cfg.config_hash = cfg_hash

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)

    return cfg_hash

