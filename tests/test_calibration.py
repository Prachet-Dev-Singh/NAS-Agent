"""
test_calibration.py
-------------------
Phase 3 acceptance test suite for the NAS-Agent calibration protocol and configuration freeze.

Acceptance criteria verified:
1. Budget sampling on CPU produces binding caps with 30-50% infeasibility.
2. Proxy validity testing calculates Spearman rho and partial correlation controlling for params.
3. LR sweep constructs valid homogeneous reference models for all 5 block families.
4. Noise floor calculation accurately determines empirical sigma_acc.
5. Cryptographic config_hash is deterministic and detects any modification to frozen settings.
6. End-to-end run produces valid, verified configs/v1.yaml.
"""

from __future__ import annotations

import os
import tempfile
import pytest
import numpy as np

from nas_agent.calibration.budget import sample_budget
from nas_agent.calibration.proxy import (
    evaluate_proxy_validity,
    partial_correlation,
    spearman_rho,
)
from nas_agent.calibration.sweep import (
    build_reference_network,
    compute_noise_floor,
    select_best_multipliers,
)
from nas_agent.calibration.protocol import run_calibration_protocol
from nas_agent.config import (
    AppConfig,
    BudgetConfig,
    CalibrationConfig,
    HarnessConfig,
    compute_config_hash,
    load_config,
    save_config,
)
from nas_agent.nodes.compiler import compile_macro
from nas_agent.state import Block


# ---------------------------------------------------------------------------
# 1. Budget Sampling Tests
# ---------------------------------------------------------------------------

def test_budget_sampling_binding_caps():
    """Verify that budget sampling produces 30-50% infeasible architectures."""
    res = sample_budget(num_samples=60, seed=42, target_infeasible_ratio=0.40)

    assert res.num_samples == 60
    assert 0.30 <= res.infeasible_rate <= 0.50, (
        f"Infeasible rate must be binding between 30% and 50%, got {res.infeasible_rate:.1%}"
    )
    assert res.params_cap_m > 0.0
    assert res.macs_cap_g > 0.0
    assert res.peak_vram_cap_mb > 0.0
    assert "params_p50" in res.percentiles
    assert "macs_p50" in res.percentiles


# ---------------------------------------------------------------------------
# 2. Proxy Validity & Partial Correlation Tests
# ---------------------------------------------------------------------------

def test_spearman_rho():
    """Verify standard monotonic ranking properties."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert abs(spearman_rho(x, y) - 1.0) < 1e-6

    y_rev = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
    assert abs(spearman_rho(x, y_rev) - (-1.0)) < 1e-6


def test_partial_correlation_controls_for_confounder():
    """Verify partial correlation isolates true quality from parameter scale confounder."""
    # Case A: x and y correlate strongly only because both scale with z (model size)
    z = np.linspace(1.0, 10.0, 50)  # parameters
    x = 2.0 * z + np.random.normal(0, 0.1, 50)  # proxy accuracy
    y = 3.0 * z + np.random.normal(0, 0.1, 50)  # reference accuracy

    p_corr = partial_correlation(x, y, z)
    # Controlling for z, the partial correlation drops to near zero
    assert abs(p_corr) < 0.25

    # Case B: x and y share genuine architectural quality independent of model size
    q = np.random.normal(0, 1.0, 50)  # genuine architectural merit
    x = q + 0.1 * z
    y = q + 0.1 * z
    p_corr_sig = partial_correlation(x, y, z)
    assert p_corr_sig > 0.85


def test_evaluate_proxy_validity_pass_and_fail():
    """Verify acceptance threshold checking."""
    # Passing dataset
    proxy_accs = [0.60, 0.62, 0.65, 0.68, 0.71, 0.73, 0.75, 0.78, 0.80, 0.82]
    ref_accs =   [0.61, 0.63, 0.64, 0.69, 0.70, 0.74, 0.76, 0.77, 0.81, 0.83]
    params =     [1.0, 1.1, 1.2, 1.0, 1.3, 1.1, 1.4, 1.2, 1.5, 1.3]

    res_pass = evaluate_proxy_validity(proxy_accs, ref_accs, params, min_rho=0.5, min_partial=0.3)
    assert res_pass.passed is True
    assert res_pass.remediation_needed is False
    assert res_pass.spearman_rho >= 0.5
    assert res_pass.partial_corr >= 0.3

    # Failing dataset (uncorrelated)
    proxy_noisy = [0.82, 0.60, 0.75, 0.62, 0.70, 0.80, 0.65, 0.73, 0.68, 0.71]
    res_fail = evaluate_proxy_validity(proxy_noisy, ref_accs, params, min_rho=0.5, min_partial=0.3)
    assert res_fail.passed is False
    assert res_fail.remediation_needed is True


# ---------------------------------------------------------------------------
# 3. Sweep & Reference Architecture Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_type", ["conv", "resnet", "dwsep", "attn_lite", "ssm"])
def test_reference_networks_compilation(block_type: Block):
    """Verify that reference homogeneous networks compile for all 5 block families."""
    proposal = build_reference_network(block_type, stem_width=32)
    model = compile_macro(proposal)
    assert model is not None
    assert len(model.stages) == 3


def test_noise_floor_computation():
    """Verify empirical standard deviation calculation."""
    accs = [0.680, 0.685, 0.675, 0.682, 0.678]
    sigma = compute_noise_floor(accs)
    assert 0.001 < sigma < 0.010


# ---------------------------------------------------------------------------
# 4. Config & Cryptographic Freeze Tests
# ---------------------------------------------------------------------------

def test_config_hash_sensitivity():
    """Verify that any modification to frozen hyperparameters alters the config_hash."""
    cfg = AppConfig(
        version="v1.1",
        budget=BudgetConfig(params_cap_m=8.0, macs_cap_g=1.2, peak_vram_cap_mb=1500.0),
        harness=HarnessConfig(base_lr=2e-3),
        calibration=CalibrationConfig(
            min_val_acc=0.60,
            tau=0.72,
            sigma_acc=0.005,
            block_multipliers={"conv": 1.0, "resnet": 1.0, "dwsep": 1.0, "attn_lite": 1.25, "ssm": 1.25},
        ),
    )

    base_hash = cfg.compute_hash()
    assert len(base_hash) == 16

    # 1. Modify base_lr -> hash must change
    cfg_mutated1 = cfg.model_copy(deep=True)
    cfg_mutated1.harness.base_lr = 1e-3
    assert cfg_mutated1.compute_hash() != base_hash

    # 2. Modify cap -> hash must change
    cfg_mutated2 = cfg.model_copy(deep=True)
    cfg_mutated2.budget.params_cap_m = 7.5
    assert cfg_mutated2.compute_hash() != base_hash

    # 3. Modify block multiplier -> hash must change
    cfg_mutated3 = cfg.model_copy(deep=True)
    cfg_mutated3.calibration.block_multipliers["conv"] = 1.1
    assert cfg_mutated3.compute_hash() != base_hash


def test_config_tamper_detection():
    """Verify that load_config rejects modified YAML files with stale hashes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "test_config.yaml")

        cfg = AppConfig(
            budget=BudgetConfig(params_cap_m=5.0, macs_cap_g=1.0, peak_vram_cap_mb=1200.0),
            calibration=CalibrationConfig(min_val_acc=0.60, tau=0.70, sigma_acc=0.004),
        )
        saved_hash = save_config(cfg, yaml_path)

        # Clean load must succeed
        loaded = load_config(yaml_path, verify_hash=True)
        assert loaded.config_hash == saved_hash

        # Tamper with the file (change params_cap_m without updating config_hash)
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()

        tampered_content = content.replace("params_cap_m: 5.0", "params_cap_m: 9.9")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(tampered_content)

        # Verification must raise ValueError
        with pytest.raises(ValueError, match="Config hash mismatch"):
            load_config(yaml_path, verify_hash=True)


def test_end_to_end_protocol_execution():
    """Verify complete Phase 3 execution and file generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_yaml = os.path.join(tmpdir, "v1_test.yaml")
        report = run_calibration_protocol(
            output_yaml_path=out_yaml,
            num_budget_samples=50,
            seed=42,
        )

        assert os.path.exists(out_yaml)
        assert report.proxy_result.passed is True
        assert 0.30 <= report.budget_result.infeasible_rate <= 0.50
        assert report.config_hash is not None

        # Verify loaded config matches report
        cfg = load_config(out_yaml, verify_hash=True)
        assert cfg.config_hash == report.config_hash
        assert cfg.budget.params_cap_m == report.budget_result.params_cap_m
        assert cfg.calibration.min_val_acc == report.min_val_acc
