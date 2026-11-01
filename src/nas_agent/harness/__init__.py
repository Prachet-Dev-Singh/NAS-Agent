"""
harness package
---------------
Fixed training harness, subprocess worker isolation, and benchmarking utilities.
"""

from nas_agent.harness.data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    DatasetSplit,
    FastBatchLoader,
    create_mock_data,
    get_cifar10_data,
)
from nas_agent.harness.executor import (
    Executor,
    LocalExecutor,
    ModalExecutor,
    run_training_worker,
)
from nas_agent.harness.metrics import (
    atomic_write_json,
    benchmark_latency,
    get_env_info,
    get_peak_vram_mb,
    reset_peak_vram,
    try_read_json,
)

__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "DatasetSplit",
    "FastBatchLoader",
    "create_mock_data",
    "get_cifar10_data",
    "Executor",
    "LocalExecutor",
    "ModalExecutor",
    "run_training_worker",
    "atomic_write_json",
    "benchmark_latency",
    "get_env_info",
    "get_peak_vram_mb",
    "reset_peak_vram",
    "try_read_json",
]
