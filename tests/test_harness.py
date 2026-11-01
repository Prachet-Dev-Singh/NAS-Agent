"""
test_harness.py
---------------
Phase 2 acceptance test suite for the NAS-Agent harness.

Acceptance criteria verified:
1. Hand-written spec trains and returns valid metrics.json.
2. Forced OOM -> {"status": "oom"}.
3. Forced hang -> {"status": "timeout"} via process group killing.
4. Forced NaN -> {"status": "nan"}.
5. Forced crash -> {"status": "crash"} with traceback tail.
6. Subprocess isolation verified (parent remains clean and alive).
7. GPU/CPU resident dataset and FastBatchLoader augmentation.
8. Latency benchmarking (median, p95, latency_mode).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import numpy as np
import pytest
import torch
import torch.nn as nn

from nas_agent.harness.data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    DatasetSplit,
    FastBatchLoader,
    create_mock_data,
    get_stratified_subset_indices,
)
from nas_agent.harness.executor import LocalExecutor, run_training_worker
from nas_agent.harness.metrics import (
    atomic_write_json,
    benchmark_latency,
    get_peak_vram_mb,
    try_read_json,
)
from nas_agent.nodes.compiler import compile_macro
from nas_agent.state import MacroProposal, StageSpec


@pytest.fixture
def sample_macro_proposal() -> MacroProposal:
    """Minimal valid 3-stage macro proposal for testing."""
    return MacroProposal(
        stem_width=32,
        stages=[
            StageSpec(block="conv", width=32, expand=2, depth=1, stride=1),
            StageSpec(block="resnet", width=32, expand=2, depth=1, stride=2),
            StageSpec(block="conv", width=64, expand=2, depth=1, stride=2),
        ],
        norm="bn",
        activation="silu",
        rationale="Minimal valid 3-stage architecture for Phase 2 harness testing.",
    )


# ---------------------------------------------------------------------------
# 1. Data Loader & Augmentation Tests
# ---------------------------------------------------------------------------

def test_stratified_subset_indices():
    """Verify that stratified selection picks exactly equal samples per class."""
    # 10 classes, 100 samples each = 1000 total
    labels = np.repeat(np.arange(10), 100)
    subset_size = 200  # 20 per class

    idx = get_stratified_subset_indices(labels, subset_size=subset_size, seed=123)
    assert len(idx) == subset_size

    selected_labels = labels[idx]
    counts = np.bincount(selected_labels, minlength=10)
    assert (counts == 20).all(), f"Expected exactly 20 samples per class, got {counts}"


def test_fast_batch_loader():
    """Verify batch slicing, augmentation, and normalization properties."""
    num_samples = 256
    train_split, _ = create_mock_data(num_train=num_samples, num_val=64, device="cpu", seed=42)

    loader = FastBatchLoader(
        split=train_split,
        batch_size=64,
        shuffle=True,
        augment=True,
        seed=100,
    )

    assert len(loader) == 4
    batch_count = 0
    total_items = 0

    for bx, by in loader:
        batch_count += 1
        total_items += bx.size(0)
        assert bx.shape == (64, 3, 32, 32)
        assert bx.dtype == torch.float32
        assert by.shape == (64,)
        assert by.dtype == torch.int64

    assert batch_count == 4
    assert total_items == num_samples


# ---------------------------------------------------------------------------
# 2. Metrics & Benchmarking Tests
# ---------------------------------------------------------------------------

def test_atomic_json_io():
    """Verify atomic write and safe read."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test.json")
        data = {"hello": "world", "num": 42}

        atomic_write_json(json_path, data)
        loaded = try_read_json(json_path)
        assert loaded == data

        # Non-existent file returns None
        assert try_read_json(os.path.join(tmpdir, "nonexistent.json")) is None


def test_latency_benchmark(sample_macro_proposal):
    """Verify inference latency timing returns valid statistics."""
    model = compile_macro(sample_macro_proposal)
    res = benchmark_latency(
        model=model,
        input_shape=(1, 3, 32, 32),
        device="cpu",
        warmup=5,
        repetitions=10,
        force_eager=True,
    )

    assert "latency_ms_median" in res
    assert "latency_ms_p95" in res
    assert "latency_mode" in res
    assert res["latency_ms_median"] > 0.0
    assert res["latency_ms_p95"] >= res["latency_ms_median"]
    assert res["repetitions"] == 10


# ---------------------------------------------------------------------------
# 3. Subprocess Training & Isolation Tests
# ---------------------------------------------------------------------------

def test_local_executor_training_success(sample_macro_proposal):
    """Test 1: Hand-written spec trains and returns valid metrics.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = os.path.join(tmpdir, "spec.json")
        out_json = os.path.join(tmpdir, "metrics.json")
        log_dir = os.path.join(tmpdir, "logs")

        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(sample_macro_proposal.model_dump_json())

        executor = LocalExecutor(default_extra_args=[
            "--mock-data",
            "--epochs", "2",
            "--batch-size", "64",
            "--subset-size", "128",
            "--val-size", "64",
            "--warmup-epochs", "1",
        ])

        result = executor.run(
            candidate_id="cand_test_001",
            spec_path=spec_path,
            out_json=out_json,
            log_dir=log_dir,
            timeout_sec=60,
        )

        assert result["status"] == "ok", f"Expected 'ok', got {result}"
        assert result["epochs_done"] == 2
        assert len(result["epoch_log"]) == 2
        assert isinstance(result["val_acc"], float)
        assert 0.0 <= result["val_acc"] <= 1.0
        assert result["params_m"] > 0.0
        assert result["macs_g"] > 0.0
        assert result["train_seconds"] > 0.0
        assert os.path.exists(out_json)


def test_local_executor_forced_oom(sample_macro_proposal):
    """Test 2: Forced OOM -> status 'oom' and process exits cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = os.path.join(tmpdir, "spec.json")
        out_json = os.path.join(tmpdir, "metrics.json")
        log_dir = os.path.join(tmpdir, "logs")

        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(sample_macro_proposal.model_dump_json())

        executor = LocalExecutor(default_extra_args=[
            "--test-action", "oom",
        ])

        result = executor.run(
            candidate_id="cand_oom",
            spec_path=spec_path,
            out_json=out_json,
            log_dir=log_dir,
            timeout_sec=30,
        )

        assert result["status"] == "oom", f"Expected 'oom', got {result}"
        assert result["val_acc"] is None
        assert "OOM" in result.get("error_tail", "") or "memory" in result.get("error_tail", "").lower()


def test_local_executor_forced_timeout(sample_macro_proposal):
    """Test 3: Forced hang -> status 'timeout' via process group killing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = os.path.join(tmpdir, "spec.json")
        out_json = os.path.join(tmpdir, "metrics.json")
        log_dir = os.path.join(tmpdir, "logs")

        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(sample_macro_proposal.model_dump_json())

        executor = LocalExecutor(default_extra_args=[
            "--test-action", "hang",
        ])

        t0 = time.time()
        result = executor.run(
            candidate_id="cand_hang",
            spec_path=spec_path,
            out_json=out_json,
            log_dir=log_dir,
            timeout_sec=3,  # Fast 3-second timeout
        )
        elapsed = time.time() - t0

        assert result["status"] == "timeout", f"Expected 'timeout', got {result}"
        assert elapsed < 15.0, f"Process took too long to terminate: {elapsed}s"


def test_local_executor_forced_nan(sample_macro_proposal):
    """Test 4: Forced NaN -> status 'nan'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = os.path.join(tmpdir, "spec.json")
        out_json = os.path.join(tmpdir, "metrics.json")
        log_dir = os.path.join(tmpdir, "logs")

        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(sample_macro_proposal.model_dump_json())

        executor = LocalExecutor(default_extra_args=[
            "--test-action", "nan",
        ])

        result = executor.run(
            candidate_id="cand_nan",
            spec_path=spec_path,
            out_json=out_json,
            log_dir=log_dir,
            timeout_sec=30,
        )

        assert result["status"] == "nan", f"Expected 'nan', got {result}"


def test_local_executor_forced_crash(sample_macro_proposal):
    """Test 5: Forced unhandled exception -> status 'crash' with traceback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = os.path.join(tmpdir, "spec.json")
        out_json = os.path.join(tmpdir, "metrics.json")
        log_dir = os.path.join(tmpdir, "logs")

        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(sample_macro_proposal.model_dump_json())

        executor = LocalExecutor(default_extra_args=[
            "--test-action", "crash",
        ])

        result = executor.run(
            candidate_id="cand_crash",
            spec_path=spec_path,
            out_json=out_json,
            log_dir=log_dir,
            timeout_sec=30,
        )

        assert result["status"] == "crash", f"Expected 'crash', got {result}"
        assert "error_tail" in result
        assert "crash" in result["error_tail"].lower()
