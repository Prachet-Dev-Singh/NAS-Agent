"""
proxy.py
--------
Proxy validity testing, rank correlation, and partial correlation analysis (Phase 3, Step 4).

Protocol (Blueprint Section 7.4 & Study Guide Part 3.8):
- Compares 8-epoch / 10k-image proxy accuracy against 25-epoch / 20k-image reference accuracy.
- Computes Spearman rank correlation (rho) with bootstrap confidence intervals.
- Computes Partial Correlation controlling for model parameters (params_m) to ensure the proxy
  evaluates architectural quality beyond parameter scaling.
- Evaluates against acceptance threshold: rho >= 0.5 and partial_corr >= 0.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def rank_data(arr: np.ndarray) -> np.ndarray:
    """Compute fractional ranks for 1D array."""
    temp = np.argsort(arr)
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(arr), dtype=float) + 1.0

    # Handle ties
    for val in np.unique(arr):
        mask = (arr == val)
        if np.sum(mask) > 1:
            ranks[mask] = np.mean(ranks[mask])

    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman's rank correlation coefficient."""
    rx = rank_data(x)
    ry = rank_data(y)
    vx = rx - np.mean(rx)
    vy = ry - np.mean(ry)
    denom = np.sqrt(np.sum(vx**2) * np.sum(vy**2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(vx * vy) / denom)


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for Spearman's rho."""
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        rhos.append(spearman_rho(x[idx], y[idx]))

    lower_p = (1.0 - ci) / 2.0 * 100.0
    upper_p = (1.0 + ci) / 2.0 * 100.0
    return float(np.percentile(rhos, lower_p)), float(np.percentile(rhos, upper_p))


def partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
    """Compute partial correlation between x and y controlling for covariate z.

    Formula:
        r_xy_z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))

    When applied to ranks, yields the partial Spearman correlation.
    """
    rx = rank_data(x)
    ry = rank_data(y)
    rz = rank_data(z)

    r_xy = spearman_rho(rx, ry)
    r_xz = spearman_rho(rx, rz)
    r_yz = spearman_rho(ry, rz)

    denom_sq = (1.0 - r_xz**2) * (1.0 - r_yz**2)
    if denom_sq <= 0.0:
        return 0.0

    return float((r_xy - r_xz * r_yz) / np.sqrt(denom_sq))


@dataclass
class ProxyValidationResult:
    """Outcomes from the Phase 3 proxy validity assessment."""
    num_models: int
    spearman_rho: float
    ci_lower: float
    ci_upper: float
    partial_corr: float
    passed: bool
    remediation_needed: bool
    summary: str


def evaluate_proxy_validity(
    proxy_accs: List[float],
    ref_accs: List[float],
    params_m: List[float],
    min_rho: float = 0.5,
    min_partial: float = 0.3,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> ProxyValidationResult:
    """Verify that proxy rankings faithfully reflect full reference performance.

    Acceptance Criteria (Section 7.4):
    - Spearman rho >= 0.5
    - Partial correlation (controlling for params) >= 0.3
    """
    x = np.array(proxy_accs, dtype=float)
    y = np.array(ref_accs, dtype=float)
    z = np.array(params_m, dtype=float)

    if len(x) < 5:
        raise ValueError(f"At least 5 models required for proxy validation, got {len(x)}")

    rho = spearman_rho(x, y)
    ci_low, ci_high = bootstrap_spearman_ci(x, y, n_bootstraps=n_bootstraps, seed=seed)
    p_corr = partial_correlation(x, y, z)

    passed = (rho >= min_rho) and (p_corr >= min_partial)

    summary = (
        f"Proxy Validity: Spearman rho={rho:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}]), "
        f"Partial corr (controlling for params)={p_corr:.3f}. "
        f"Status: {'PASS' if passed else 'FAIL (remediation required)'}"
    )

    return ProxyValidationResult(
        num_models=len(x),
        spearman_rho=round(rho, 4),
        ci_lower=round(ci_low, 4),
        ci_upper=round(ci_high, 4),
        partial_corr=round(p_corr, 4),
        passed=passed,
        remediation_needed=not passed,
        summary=summary,
    )
