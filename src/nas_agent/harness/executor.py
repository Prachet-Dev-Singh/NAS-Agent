"""
executor.py
-----------
Execution engines for the NAS-Agent training worker.

Implements:
1. Executor Interface:
   - Abstract contract: run(candidate_id, spec_path, out_json, log_dir, timeout_sec) -> dict
2. LocalExecutor:
   - Subprocess worker execution with full process group isolation and timeout killing.
   - Cross-platform support (Windows CREATE_NEW_PROCESS_GROUP / taskkill and Linux killpg).
   - Captures stdout/stderr into candidate-specific log files.
   - Incremental JSON recovery on timeout/crash.
3. ModalExecutor:
   - Remote execution on Modal GPU cloud with Volume storage.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from nas_agent.harness.metrics import read_log_tail, try_read_json


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill process and all child processes across both Windows and POSIX systems."""
    if proc.poll() is not None:
        return

    pid = proc.pid
    if sys.platform == "win32":
        try:
            # Forcefully terminate process and all child processes created by it
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def run_training_worker(
    candidate_id: str,
    spec_path: str,
    out_json: str,
    log_dir: str,
    timeout_sec: int = 300,
    device: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Launch training worker as an isolated subprocess with timeout and process-group reaping.

    Blueprint Section 6.1 contract:
    - Runs in a new process session/group.
    - Captures stdout and stderr to candidate-specific files.
    - Kills entire process group on timeout.
    - Worker-reported JSON status wins if written; otherwise reports crash with stderr tail.
    """
    os.makedirs(log_dir, exist_ok=True)
    out_log = os.path.join(log_dir, f"{candidate_id}.out")
    err_log = os.path.join(log_dir, f"{candidate_id}.err")

    cmd = [
        sys.executable,
        "-m",
        "nas_agent.harness.worker",
        "--spec",
        spec_path,
        "--out",
        out_json,
    ]
    if device is not None:
        cmd.extend(["--device", device])
    if extra_args:
        cmd.extend(extra_args)

    popen_kwargs: Dict[str, Any] = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    with open(out_log, "w", encoding="utf-8") as fo, open(err_log, "w", encoding="utf-8") as fe:
        proc = subprocess.Popen(
            cmd,
            stdout=fo,
            stderr=fe,
            **popen_kwargs,
        )

        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

            # Check if worker wrote partial metrics before timeout
            result = try_read_json(out_json)
            if result is not None:
                result["status"] = "timeout"
                result["error_tail"] = "exceeded timeout"
                return result
            return {"status": "timeout", "error_tail": "exceeded timeout"}
        finally:
            _kill_group(proc)

    # Worker-reported status wins if JSON was successfully emitted
    result = try_read_json(out_json)
    if result is not None:
        return result

    # Fallback to crash status with stderr tail
    return {
        "status": "crash",
        "error_tail": read_log_tail(err_log, max_lines=40) or "Process terminated without metrics output",
        "returncode": proc.returncode,
    }


class Executor(ABC):
    """Abstract interface for executing candidate evaluation."""

    @abstractmethod
    def run(
        self,
        candidate_id: str,
        spec_path: str,
        out_json: str,
        log_dir: str,
        timeout_sec: int = 300,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute evaluation of candidate specification and return metrics dictionary."""
        pass


class LocalExecutor(Executor):
    """Executes the training worker as a local isolated subprocess."""

    def __init__(self, device: Optional[str] = None, default_extra_args: Optional[List[str]] = None) -> None:
        self.device = device
        self.default_extra_args = default_extra_args or []

    def run(
        self,
        candidate_id: str,
        spec_path: str,
        out_json: str,
        log_dir: str,
        timeout_sec: int = 300,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        extra_args = list(self.default_extra_args)
        if "extra_args" in kwargs and kwargs["extra_args"]:
            extra_args.extend(kwargs["extra_args"])

        device = kwargs.get("device", self.device)
        return run_training_worker(
            candidate_id=candidate_id,
            spec_path=spec_path,
            out_json=out_json,
            log_dir=log_dir,
            timeout_sec=timeout_sec,
            device=device,
            extra_args=extra_args,
        )


class ModalExecutor(Executor):
    """Executes training remotely on Modal cloud infrastructure with attached Volume."""

    def __init__(self, app_name: str = "nas-agent", function_name: str = "run_remote_worker") -> None:
        self.app_name = app_name
        self.function_name = function_name

    def run(
        self,
        candidate_id: str,
        spec_path: str,
        out_json: str,
        log_dir: str,
        timeout_sec: int = 300,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            import modal
        except ImportError:
            raise RuntimeError(
                "modal package is not installed. Install with `pip install modal` to use ModalExecutor, "
                "or use LocalExecutor for local/Kaggle/Colab runs."
            )

        try:
            from nas_agent.harness.modal_app import run_remote_worker
            with open(spec_path, "r", encoding="utf-8") as f:
                import json
                spec_dict = json.load(f)

            # Call remote Modal function
            result = run_remote_worker.remote(
                candidate_id=candidate_id,
                spec_dict=spec_dict,
                timeout_sec=timeout_sec,
            )
            # Write out_json locally for downstream consumers
            from nas_agent.harness.metrics import atomic_write_json
            atomic_write_json(out_json, result)
            return result
        except Exception as e:
            return {
                "status": "crash",
                "error_tail": f"ModalExecutor error: {str(e)}",
            }
