"""
calibration package
-------------------
Calibration protocol and freeze utilities for NAS-Agent (Phase 3).
"""

from nas_agent.calibration.budget import (
    BudgetSamplingResult,
    sample_budget,
    sample_random_macro_proposal,
)
from nas_agent.calibration.protocol import (
    CalibrationReport,
    run_calibration_protocol,
)
from nas_agent.calibration.proxy import (
    ProxyValidationResult,
    evaluate_proxy_validity,
    partial_correlation,
    spearman_rho,
)
from nas_agent.calibration.sweep import (
    build_reference_network,
    compute_noise_floor,
    select_best_multipliers,
)

__all__ = [
    "BudgetSamplingResult",
    "sample_budget",
    "sample_random_macro_proposal",
    "CalibrationReport",
    "run_calibration_protocol",
    "ProxyValidationResult",
    "evaluate_proxy_validity",
    "partial_correlation",
    "spearman_rho",
    "build_reference_network",
    "compute_noise_floor",
    "select_best_multipliers",
]
