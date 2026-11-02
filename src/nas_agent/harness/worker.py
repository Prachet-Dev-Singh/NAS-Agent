"""
worker.py
---------
Subprocess training worker for the NAS-Agent fixed harness.

Design Contract (Blueprint Section 6 & Study Guide):
1. Subprocess Isolation:
   - Runs as a standalone process launched by LocalExecutor / ModalExecutor.
   - Enforces deterministic training using fixed seeds derived from run_seed + spec_hash.
2. Error Catching at Top Level:
   - Catches torch.cuda.OutOfMemoryError / MemoryError -> writes {"status": "oom"} and exits 0.
   - Catches NaN/Inf loss -> writes {"status": "nan"} and exits 0.
   - Catches unhandled exceptions -> writes {"status": "crash", "error_tail": ...} and exits 0.
3. Incremental metrics.json Writing:
   - Writes metrics.json incrementally after each completed epoch so partial data survives
     hangs or external termination.
4. Fixed Training Configuration:
   - Optimizer: AdamW, betas=(0.9, 0.98), weight_decay=1e-2 (none on norm/bias).
   - Parameter groups: per-block-family LR multipliers (stem/transition/head = 1.0x).
   - Schedule: linear warmup then cosine decay to 1e-5.
   - Precision: fp16 AMP + GradScaler on CUDA; norms and SSM scan in fp32; grad clip 1.0.
   - Data: GPU-resident tensors with GPU-side augmentation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from nas_agent.harness.data import FastBatchLoader, get_cifar10_data
from nas_agent.harness.metrics import (
    atomic_write_json,
    get_env_info,
    get_peak_vram_mb,
    reset_peak_vram,
)
from nas_agent.lib.counters import count_macro_arch
from nas_agent.lib.distill import compute_distillation_loss
from nas_agent.nodes.compiler import compile_macro, compile_micro
from nas_agent.state import MacroProposal, MicroProposal


def build_optimizer_and_scheduler(
    model: nn.Module,
    base_lr: float = 2e-3,
    min_lr: float = 1e-5,
    total_epochs: int = 8,
    warmup_epochs: int = 2,
    weight_decay: float = 1e-2,
    block_multipliers: Optional[Dict[str, float]] = None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Construct AdamW optimizer with per-block LR multipliers and warmup-cosine scheduler.

    Rules (Blueprint 6.2):
    - Weight decay 1e-2 on 2D+ weights (conv, linear).
    - Weight decay 0.0 on 1D parameters (biases, normalization weights/biases).
    - Per-block family LR multipliers (stem, transition, head = 1.0x).
    """
    block_mults = block_multipliers or {}

    # Separate parameters into groups
    decay_params: list[dict] = []
    no_decay_params: list[dict] = []

    # Map named modules to block types
    for name, module in model.named_modules():
        # Check if module has a known block type attribute
        pass

    # Collect parameters and assign appropriate LR multiplier
    param_groups: list[dict] = []

    # Map each parameter to its module hierarchy to determine block family
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1D params (biases, norm scales) get weight decay 0
        use_decay = param.ndim > 1
        wd = weight_decay if use_decay else 0.0

        # Determine LR multiplier
        lr_mult = 1.0
        # Check stages for block type
        if "stages." in name:
            try:
                # e.g., stages.1.blocks.0.in_proj.weight
                parts = name.split(".")
                stage_idx = int(parts[1])
                if parts[2] == "blocks":
                    # Access the block type from the model structure if available
                    stage_module = model.stages[stage_idx]  # type: ignore[union-attr]
                    block_idx = int(parts[3])
                    block = stage_module.blocks[block_idx]
                    block_type = getattr(block, "block_type", None)
                    if block_type and block_type in block_mults:
                        lr_mult = block_mults[block_type]
            except Exception:
                lr_mult = 1.0

        param_groups.append({
            "params": [param],
            "weight_decay": wd,
            "lr": base_lr * lr_mult,
            "initial_lr": base_lr * lr_mult,
            "base_group_lr": base_lr * lr_mult,
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    # Warmup + Cosine Annealing Lambda
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            # Linear warmup from 0.1 to 1.0
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        else:
            # Cosine decay from 1.0 to min_lr / base_lr
            progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            min_ratio = min_lr / base_lr
            return float(min_ratio + (1.0 - min_ratio) * cosine_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def evaluate(
    model: nn.Module,
    val_loader: FastBatchLoader,
    device: str,
) -> tuple[float, float]:
    """Compute validation loss and top-1 accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            by = by.to(device=device, dtype=torch.long)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            preds = logits.argmax(dim=-1)

            total_loss += loss.item() * bx.size(0)
            total_correct += (preds == by).sum().item()
            total_samples += bx.size(0)

    val_loss = total_loss / max(1, total_samples)
    val_acc = float(total_correct) / max(1, total_samples)
    return val_loss, val_acc


def run_worker(args: argparse.Namespace) -> None:
    """Main worker execution function."""
    start_time = time.time()
    env_info = get_env_info()

    # Base payload structure
    metrics_payload: Dict[str, Any] = {
        "status": "crash",
        "val_acc": None,
        "params_m": None,
        "macs_g": None,
        "peak_vram_mb": 0.0,
        "train_seconds": 0.0,
        "epochs_done": 0,
        "epoch_log": [],
        "error_tail": None,
        "spec_hash": None,
        "config_hash": args.config_hash,
        "seed": args.seed,
        "env": env_info,
    }

    # Test hooks for deliberate verification of error modes
    test_action = args.test_action or os.getenv("NAS_WORKER_TEST_ACTION")
    if test_action == "hang":
        # Simulate infinite loop / hang
        time.sleep(3600)
    elif test_action == "oom":
        if torch.cuda.is_available():
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (forced for test)")
        else:
            raise MemoryError("Out of memory (forced for test)")
    elif test_action == "nan":
        raise FloatingPointError("Loss is NaN (forced for test)")
    elif test_action == "crash":
        raise RuntimeError("Simulated unhandled worker crash")

    # 1. Parse proposal specification
    if not os.path.exists(args.spec):
        raise FileNotFoundError(f"Spec file not found: {args.spec}")

    with open(args.spec, "r", encoding="utf-8") as f:
        spec_dict = json.load(f)

    is_micro = (spec_dict.get("kind") == "micro")
    teacher_model = None

    if is_micro:
        proposal = MicroProposal.model_validate(spec_dict)
        parent_spec_dict = spec_dict.get("parent_spec", {})
        parent_macro = MacroProposal.model_validate(parent_spec_dict)
        spec_hash = proposal.spec_hash(parent_macro.spec_hash())
        metrics_payload["spec_hash"] = spec_hash
    else:
        proposal = MacroProposal.model_validate(spec_dict)
        spec_hash = proposal.spec_hash()
        metrics_payload["spec_hash"] = spec_hash
        stats = count_macro_arch(proposal)
        metrics_payload["params_m"] = round(stats.params_m, 4)
        metrics_payload["macs_g"] = round(stats.macs_g, 4)

    # 2. Set random seeds deterministically
    seed = args.seed
    if seed is None:
        # Derive seed from spec_hash
        seed = int(spec_hash[:8], 16) % (2**31)
    metrics_payload["seed"] = seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 3. Determine device
    device_str = args.device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Reset VRAM tracker
    reset_peak_vram(device_str)

    # 4. Compile model
    if is_micro:
        parent_model = compile_macro(parent_macro)
        parent_ckpt = spec_dict.get("parent_ckpt")
        if parent_ckpt and os.path.exists(parent_ckpt):
            try:
                parent_model.load_state_dict(torch.load(parent_ckpt, map_location="cpu"))
            except Exception:
                pass

        if proposal.distill and proposal.distill.enabled:
            teacher_model = copy.deepcopy(parent_model).to(device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False

        model = compile_micro(proposal, parent_model).to(device)
        actual_params = sum(p.numel() for p in model.parameters())
        metrics_payload["params_m"] = round(actual_params / 1e6, 4)
        metrics_payload["macs_g"] = round(count_macro_arch(parent_macro).macs_g, 4)
    else:
        model = compile_macro(proposal).to(device)

    # 5. Load data
    train_split, val_split = get_cifar10_data(
        data_dir=args.data_dir,
        subset_size=args.subset_size,
        val_size=args.val_size,
        device=device_str,
        seed=42,  # Fixed stratified split seed across all candidates
        mock=args.mock_data,
    )

    train_loader = FastBatchLoader(
        split=train_split,
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
        seed=seed,
    )
    val_loader = FastBatchLoader(
        split=val_split,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        seed=42,
    )

    # 6. Build optimizer and scheduler
    block_mults = None
    if args.block_multipliers:
        try:
            block_mults = json.loads(args.block_multipliers)
        except Exception:
            pass

    total_epochs = 4 if is_micro else args.epochs
    warmup_epochs = 1 if is_micro else args.warmup_epochs
    base_lr = (args.lr * 0.3) if is_micro else args.lr

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        base_lr=base_lr,
        min_lr=args.min_lr,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        block_multipliers=block_mults,
    )

    # AMP setup
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epoch_log: List[Dict[str, Any]] = []

    # 7. Training loop
    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logits = model(bx)
                if teacher_model is not None and is_micro and proposal.distill and proposal.distill.enabled:
                    with torch.no_grad():
                        teacher_logits = teacher_model(bx)
                    loss = compute_distillation_loss(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        targets=by,
                        alpha=proposal.distill.alpha,
                        temperature=proposal.distill.temperature,
                    )
                else:
                    loss = F.cross_entropy(logits, by)

            # Check for NaN / Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                raise FloatingPointError(f"Encountered NaN/Inf loss in epoch {epoch}")

            # Backward pass with GradScaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient clipping (max norm 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * bx.size(0)
            running_correct += (logits.argmax(dim=-1) == by).sum().item()
            running_samples += bx.size(0)

        # Validation at epoch end
        train_loss = running_loss / max(1, running_samples)
        train_acc = float(running_correct) / max(1, running_samples)
        val_loss, val_acc = evaluate(model, val_loader, device=device_str)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(current_lr, 6),
        }
        epoch_log.append(epoch_record)

        # Incremental metrics update after each epoch
        metrics_payload["epochs_done"] = epoch
        metrics_payload["epoch_log"] = epoch_log
        metrics_payload["val_acc"] = round(val_acc, 4)
        metrics_payload["train_seconds"] = round(time.time() - start_time, 2)
        metrics_payload["peak_vram_mb"] = round(get_peak_vram_mb(device_str), 2)
        metrics_payload["status"] = "ok"

        atomic_write_json(args.out, metrics_payload)

    # 8. Checkpoint saving if requested
    if args.ckpt_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.ckpt_out)), exist_ok=True)
        torch.save(model.state_dict(), args.ckpt_out)

    # Final completion status
    metrics_payload["status"] = "ok"
    metrics_payload["train_seconds"] = round(time.time() - start_time, 2)
    atomic_write_json(args.out, metrics_payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="NAS-Agent training worker")
    parser.add_argument("--spec", type=str, required=True, help="Path to spec JSON")
    parser.add_argument("--out", type=str, required=True, help="Path to output metrics.json")
    parser.add_argument("--device", type=str, default=None, help="Target device (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Base learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for CIFAR-10 data")
    parser.add_argument("--subset-size", type=int, default=10000, help="Train subset size")
    parser.add_argument("--val-size", type=int, default=5000, help="Validation set size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--config-hash", type=str, default="frozen_phase3_default")
    parser.add_argument("--block-multipliers", type=str, default=None, help="JSON of per-block multipliers")
    parser.add_argument("--mock-data", action="store_true", help="Use synthetic mock data")
    parser.add_argument("--test-action", type=str, default=None, help="Deliberate error injection for testing")
    parser.add_argument("--ckpt-out", type=str, default=None, help="Path to save checkpoint")

    args = parser.parse_args()

    try:
        run_worker(args)
        sys.exit(0)
    except (torch.cuda.OutOfMemoryError, MemoryError) as e:
        # Blueprint 6.1: Catch OOM, write status "oom", exit 0
        err_msg = f"{type(e).__name__}: {str(e)}"
        payload = {
            "status": "oom",
            "val_acc": None,
            "error_tail": err_msg,
            "config_hash": args.config_hash,
            "seed": args.seed,
            "env": get_env_info(),
        }
        # If output file already exists with partial metrics, preserve them
        if os.path.exists(args.out):
            try:
                with open(args.out, "r") as f:
                    prev = json.load(f)
                prev.update(payload)
                payload = prev
            except Exception:
                pass
        atomic_write_json(args.out, payload)
        sys.exit(0)
    except (FloatingPointError, ValueError) as e:
        # Catch NaN/Inf
        if "NaN" in str(e) or "Inf" in str(e):
            err_msg = f"{type(e).__name__}: {str(e)}"
            payload = {
                "status": "nan",
                "val_acc": None,
                "error_tail": err_msg,
                "config_hash": args.config_hash,
                "seed": args.seed,
                "env": get_env_info(),
            }
            if os.path.exists(args.out):
                try:
                    with open(args.out, "r") as f:
                        prev = json.load(f)
                    prev.update(payload)
                    payload = prev
                except Exception:
                    pass
            atomic_write_json(args.out, payload)
            sys.exit(0)
        else:
            # Other ValueError falls under crash
            tb_lines = traceback.format_exc().splitlines()[-40:]
            payload = {
                "status": "crash",
                "val_acc": None,
                "error_tail": "\n".join(tb_lines),
                "config_hash": args.config_hash,
                "seed": args.seed,
                "env": get_env_info(),
            }
            atomic_write_json(args.out, payload)
            sys.exit(0)
    except Exception as e:
        # Blueprint 6.1: Catch unhandled crash, write status "crash", exit 0
        tb_lines = traceback.format_exc().splitlines()[-40:]
        payload = {
            "status": "crash",
            "val_acc": None,
            "error_tail": "\n".join(tb_lines),
            "config_hash": args.config_hash,
            "seed": args.seed,
            "env": get_env_info(),
        }
        if os.path.exists(args.out):
            try:
                with open(args.out, "r") as f:
                    prev = json.load(f)
                prev.update(payload)
                payload = prev
            except Exception:
                pass
        atomic_write_json(args.out, payload)
        sys.exit(0)


if __name__ == "__main__":
    main()
