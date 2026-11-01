"""
metrics.py
----------
Measurement and benchmarking utilities for the NAS-Agent harness.

Features:
1. Post-Hoc Latency Benchmarking (Blueprint Section 6.3):
   - 50 warmup iterations, 200 repetitions timed with CUDA events (or high-res CPU timer).
   - Reports median and p95 latency in milliseconds.
   - CUDA-graph capture with automatic fallback to eager mode; records `latency_mode`.
2. Peak VRAM Tracking:
   - Measures `torch.cuda.max_memory_allocated` in MB.
3. Robust Incremental JSON I/O:
   - Atomic file writing (temp file + rename) for incremental `metrics.json` updates.
   - Safe parsing with error recovery.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def get_env_info() -> Dict[str, Any]:
    """Capture runtime environment metadata for the metrics payload."""
    cuda_avail = torch.cuda.is_available()
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda if cuda_avail else None,
        "gpu": torch.cuda.get_device_name(0) if cuda_avail else "cpu",
        "platform": sys.platform,
    }


def get_peak_vram_mb(device: str = "cuda") -> float:
    """Return peak allocated GPU memory in Megabytes (excluding CUDA context)."""
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024))
    return 0.0


def reset_peak_vram(device: str = "cuda") -> None:
    """Reset peak memory tracking counter."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)


def atomic_write_json(filepath: str, data: Dict[str, Any]) -> None:
    """Atomically write data as JSON using a temp file and rename.

    Prevents partial/corrupted JSON reads if a process is killed or interrupted.
    """
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as tf:
        json.dump(data, tf, indent=2)
        temp_name = tf.name

    # Atomic rename / replace
    os.replace(temp_name, filepath)


def try_read_json(filepath: str) -> Optional[Dict[str, Any]]:
    """Attempt to read and parse a JSON file, returning None on failure."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_log_tail(filepath: str, max_lines: int = 40) -> Optional[str]:
    """Read the last N lines of a log file."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return "".join(lines[-max_lines:])
    except Exception:
        return None


def benchmark_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    device: str = "cuda",
    warmup: int = 50,
    repetitions: int = 200,
    force_eager: bool = False,
) -> Dict[str, Any]:
    """Benchmark inference latency using CUDA events (or high-res CPU perf counter).

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model in eval mode.
    input_shape : Tuple[int, ...]
        Batch shape for inference (default (1, 3, 32, 32)).
    device : str
        Target device ('cuda' or 'cpu').
    warmup : int
        Number of warmup iterations before timing (default 50).
    repetitions : int
        Number of timed repetitions (default 200).
    force_eager : bool
        If True, skip CUDA Graph capture attempt and benchmark in eager mode.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'latency_ms_median', 'latency_ms_p95', 'latency_mode', 'repetitions'.
    """
    actual_device = "cuda" if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    model = model.to(actual_device)
    model.eval()

    sample_input = torch.randn(*input_shape, device=actual_device)
    latency_mode = "eager"
    cuda_graph = None

    # Attempt CUDA Graph capture if on CUDA and not force_eager
    if actual_device == "cuda" and not force_eager:
        try:
            # Warm up stream
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _ = model(sample_input)
            torch.cuda.current_stream().wait_stream(stream)

            # Capture graph
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph):
                _ = model(sample_input)
            latency_mode = "graph"
        except Exception:
            # Fall back cleanly to eager execution if CUDA Graph fails (e.g. dynamic shapes/SSM)
            cuda_graph = None
            latency_mode = "eager"

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if cuda_graph is not None:
                cuda_graph.replay()
            else:
                _ = model(sample_input)
        if actual_device == "cuda":
            torch.cuda.synchronize()

    # Repetitions timing
    times_ms: list[float] = []

    if actual_device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(repetitions):
                start_event.record()
                if cuda_graph is not None:
                    cuda_graph.replay()
                else:
                    _ = model(sample_input)
                end_event.record()
                torch.cuda.synchronize()
                times_ms.append(float(start_event.elapsed_time(end_event)))
    else:
        with torch.no_grad():
            for _ in range(repetitions):
                t0 = time.perf_counter_ns()
                _ = model(sample_input)
                t1 = time.perf_counter_ns()
                times_ms.append((t1 - t0) / 1_000_000.0)

    median_ms = float(np.median(times_ms))
    p95_ms = float(np.percentile(times_ms, 95))

    return {
        "latency_ms_median": round(median_ms, 4),
        "latency_ms_p95": round(p95_ms, 4),
        "latency_mode": latency_mode,
        "repetitions": repetitions,
    }
