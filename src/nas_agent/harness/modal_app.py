"""
modal_app.py
------------
Modal cloud deployment specification for NAS-Agent workers.

Architecture (Blueprint Section 1.3 & 6.1):
- The orchestrator (LangGraph loop) runs locally or in a controller container.
- When evaluating a candidate, ModalExecutor invokes run_remote_worker.
- INSIDE the Modal container, run_training_worker spawns the isolated subprocess worker.
  This guarantees that OOMs, hangs, and memory fragmentation are isolated from the container worker.
- Persistent Volume holds cached CIFAR-10 datasets and candidate checkpoints.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict

try:
    import modal

    app = modal.App("nas-agent")

    # Volume for shared datasets and model checkpoints
    volume = modal.Volume.from_name("nas-agent-volume", create_if_missing=True)

    # Base GPU container image
    worker_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.5.0",
            "torchvision>=0.20.0",
            "pydantic>=2.0.0",
            "numpy",
            "pyyaml",
        )
        .add_local_python_source("nas_agent")
    )

    @app.function(
        image=worker_image,
        gpu="T4",
        volumes={"/vol": volume},
        timeout=600,
        scaledown_window=300,  # Keep container warm to avoid per-candidate cold start
    )
    def run_remote_worker(
        candidate_id: str,
        spec_dict: Dict[str, Any],
        timeout_sec: int = 300,
    ) -> Dict[str, Any]:
        """Execute candidate evaluation inside an isolated subprocess within the Modal container."""
        from nas_agent.harness.executor import run_training_worker

        data_dir = "/vol/data"
        log_dir = "/vol/logs"
        ckpt_dir = "/vol/checkpoints"
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = os.path.join(tmpdir, f"{candidate_id}_spec.json")
            out_json = os.path.join(tmpdir, f"{candidate_id}_metrics.json")
            ckpt_path = os.path.join(ckpt_dir, f"{candidate_id}.pt")

            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(spec_dict, f)

            extra_args = [
                "--data-dir", data_dir,
                "--ckpt-out", ckpt_path,
            ]

            result = run_training_worker(
                candidate_id=candidate_id,
                spec_path=spec_path,
                out_json=out_json,
                log_dir=log_dir,
                timeout_sec=timeout_sec,
                device="cuda",
                extra_args=extra_args,
            )

            # Persist volume changes
            volume.commit()
            return result

except ImportError:
    # modal is optional when running strictly locally
    app = None  # type: ignore[assignment]
    volume = None  # type: ignore[assignment]
    run_remote_worker = None  # type: ignore[assignment]
