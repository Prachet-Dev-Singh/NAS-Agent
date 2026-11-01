# NAS Agent: Architecture Critique, Hidden Pitfalls & Actionable Corrections

> **Reference Document:** [nas-agent-blueprint.md](file:///c:/Users/prach/Downloads/NAS-Agent/nas-agent-blueprint.md)  
> **Purpose:** Detailed catalog of theoretical cracks, ML/systems pitfalls, and concrete engineering corrections to incorporate into the project design before writing code.

---

## 1. Machine Learning & NAS Validity Issues

### 1.1 Proxy Metric Validity & Rank Collapse (3 Epochs on CIFAR-10)
* **The Flaw:**
  * In standard NAS literature, short proxy runs often exhibit **rank collapse** (Spearman correlation $\rho < 0.2$ between proxy performance and fully trained models).
  * **Heterogeneous Block Optimization Bias:** The search space combines standard Convolutions, Depthwise-Separable Convolutions, Residual Blocks, Lite Attention, and Selective-Scan SSMs (Mamba). These primitives possess vastly different optimization dynamics:
    * Convolutions converge quickly with standard SGD/AdamW and aggressive initial learning rates.
    * Self-Attention and SSM blocks are notoriously sensitive to learning rate warmup, gradient scaling, and weight decay schedules.
  * If a fixed training harness evaluates all models with an identical, fixed learning rate and optimizer schedule over 3 epochs, **Attention and SSM blocks will consistently underperform Convolutions**, not due to architectural capacity, but due to optimizer mismatch. The LLM will falsely learn that Attention/SSMs are inferior.
* **The Correction:**
  * **Early Correlation Sanity Check (Phase 1):** Before running full agent loops, sample 12–15 diverse architectures across the search space. Train them for 3 epochs (proxy) and 25 epochs (baseline target). Calculate the Spearman rank correlation $\rho$. If $\rho < 0.5$, adjust the proxy.
  * **Balanced Proxy Budget:** Increase proxy training to 8–10 epochs on a stratified 10,000-image subset of CIFAR-10 with a short 2-epoch linear warmup + cosine annealing schedule, or utilize an adaptive optimizer like AdamW with weight-norm normalized gradients.

---

### 1.2 Weight Inheritance vs. Discrete Topological Mutation
* **The Flaw:**
  * The blueprint proposes **weight inheritance** (child starts from parent weights) as an efficiency mechanism.
  * While weight inheritance is mathematically well-defined for **continuous compression** (structured channel pruning, SVD low-rank factorizations, fake-quantization), it **breaks completely during discrete topological mutations** (e.g., swapping a `conv` stage for an `ssm` or `attn_lite` stage, changing stage depths, or altering downsampling ratios).
* **The Correction:**
  * Explicitly separate the search operations into two distinct workflows in the state machine:
    1. **Macro Search (Topology Exploration):** Proposes new `ArchSpec` definitions (block types, stage depths, activation functions). Initialized from scratch, trained via proxy run.
    2. **Micro Search / Surgery (Compression & Adaptation):** Operates on an existing Pareto-optimal candidate checkpoint. Applies `Transform` (prune, low-rank, quantize, distill) and fine-tunes for fewer epochs.
  * Prevent the agent from attempting to graft inherited weights across incompatible block type mutations.

---

### 1.3 Latency Benchmarking on Virtualized Cloud GPUs (Modal T4)
* **The Flaw:**
  * Measuring sub-5ms batch-1 inference latency (`batch-1 latency <= 5 ms`) on virtualized cloud GPUs (Modal T4 instances) is prone to high variance:
    * Python interpreter dispatch overhead on tiny sub-2M parameter models frequently exceeds the actual CUDA kernel execution time.
    * Cold starts, GPU clock throttling, and shared hardware virtualization introduce jitter that can overshadow architectural differences.
* **The Correction:**
  * **Micro-benchmarking Protocol:**
    1. Run 50 warmup iterations before logging timings.
    2. Measure strictly using PyTorch CUDA events rather than Python `time.perf_counter()`:
       ```python
       start = torch.cuda.Event(enable_timing=True)
       end = torch.cuda.Event(enable_timing=True)
       start.record()
       with torch.no_grad():
           _ = model(dummy_input)
       end.record()
       torch.cuda.synchronize()
       latency_ms = start.elapsed_time(end)
       ```
    3. Run 200 repetitions and record the **median** and **p95** latency.
    4. Optional: Export the candidate to TorchScript (`torch.jit.trace`) before timing to strip out Python runtime overhead.

---

## 2. Systems, Engineering & LLM Architecture Issues

### 2.1 Coder LLM Redundancy in Tier 1
* **The Flaw:**
  * In Tier 1, the search space and transforms are strictly constrained by Pydantic models (`ArchSpec` and `Transform`).
  * The **Optimizer** already outputs a structured `Directive JSON` containing the strategy, target layer, and hyperparameters.
  * Having a **Coder LLM** simply translate the Directive into `ArchSpec` or `Transform` JSON introduces an unnecessary LLM hop:
    * Doubles token consumption and API request latency.
    * Doubles the probability of rate-limit exhaustion on free tiers.
    * Introduces potential hallucination/schema-parsing errors where a deterministic compiler suffices.
* **The Correction:**
  * **Tier 1 (Spec-Only):** Remove the Coder LLM node entirely. The Optimizer outputs the `Directive` or `ArchSpec` directly, which is compiled or mutated deterministically by Python.
  * **Tier 2 (Custom Code):** Reserve the Coder LLM strictly for generating new PyTorch block implementations when Tier 2 is enabled.

---

### 2.2 Semantic Vector Search (ChromaDB) Mismatch for NAS Experience
* **The Flaw:**
  * Storing past outcomes in a vector database (ChromaDB) and querying via text embeddings (`sentence-transformers`) on `note` + `module_stats` is fundamentally ill-suited for hardware-aware NAS:
    * Text embeddings lack relational, numerical, and Pareto awareness.
    * An embedding for *"pruned stage 2 channels by 25%"* cannot distinguish between a model that collapsed accuracy to 10% versus one that maintained 90% accuracy while cutting latency by 3ms.
    * Cosine distance between prose notes fails to retrieve structurally relevant Pareto tradeoffs.
* **The Correction:**
  * Replace ChromaDB with **SQLite or DuckDB** storing structured relational columns:
    * `arch_hash`, `technique`, `parent_id`, `params_m`, `vram_mb`, `latency_ms`, `val_acc`, `is_feasible`, `pareto_dominant`.
  * **Context Retrieval Algorithm for the Optimizer:**
    1. **Pareto Frontier Query:** Retrieve all currently non-dominated architectures.
    2. **Nearest Architectural Neighbors:** Structured filter by block type and stage width.
    3. **Failure Mode Summary:** Query past failures for the targeted transform/block to warn the Optimizer (e.g., *"SVD rank < 16 on Stage 3 caused NaN losses in 3 prior attempts"*).

---

### 2.3 Fragility of In-Place PyTorch Model Surgery (`surgery.py`)
* **The Flaw:**
  * Manually replacing layers or pruning channels by string name in models with residual skip connections (`x + residual(x)`) and normalization layers leads to shape mismatch crashes:
    * Pruning output channels of convolution $L_i$ requires adjusting input channels of convolution $L_{i+1}$.
    * If $L_{i+1}$ has a skip connection to an earlier layer, modifying its channel dimensions without simultaneously modifying the identity projection will crash the addition operator (`Tensor shape [B, 64, H, W] cannot be added to [B, 48, H, W]`).
* **The Correction:**
  * In `src/nas_agent/lib/blocks.py`, enforce an **explicit block interface**:
    * Each block defines its own internal prunable dimensions (e.g., intermediate expansion/bottleneck layers) while keeping input/output stage widths locked to stage boundary contracts.
    * Or define explicit `channel_groups` so that `surgery.py` mutates coupled layers synchronously without guessing.

---

### 2.4 CUDA OOM Memory Leakage & Process Pollution
* **The Flaw:**
  * The blueprint suggests handling OOM inside the harness and continuing.
  * In PyTorch, catching `torch.cuda.OutOfMemoryError` inside a persistent process often leaves CUDA cache fragmentation and orphaned tensor allocations in the runtime. Consecutive runs inside the same process can OOM on models that should have easily fit.
* **The Correction:**
  * Run the trainer harness as an **isolated child subprocess**:
    ```python
    # Run training worker in isolated process
    proc = subprocess.run(
        [sys.executable, "-m", "nas_agent.harness.worker", "--spec", spec_path, "--out", metrics_path],
        capture_output=True,
        timeout=180
    )
    ```
  * If the worker crashes or runs out of VRAM, the operating system reclaims all GPU memory instantaneously, guaranteeing a clean slate for the next candidate.

---

## 3. Revised Component Matrix

| Component | Blueprint Original | Identified Risk | Corrected Design |
|---|---|---|---|
| **Pipeline Flow (Tier 1)** | Optimizer $\to$ Coder $\to$ Gate | Extra API latency, schema repair loops, free-tier throttling | **Optimizer $\to$ Deterministic Compiler $\to$ Gate** |
| **Experience Store** | ChromaDB (Vector Search) | Cosine similarity ignores numerical metrics & Pareto dominance | **SQLite / DuckDB** with Pareto queries and structural filters |
| **Proxy Training** | 3 epochs on CIFAR-10 subset | Rank collapse & unfair penalty against Attention/SSMs | **8–10 epochs** with LR warmup + validate Spearman $\rho \ge 0.5$ in Phase 1 |
| **Model Surgery** | Unconstrained module replacement | Shape mismatches across residual skip connections | **Bottleneck-only pruning** or coupled channel group contracts |
| **Trainer Isolation** | In-process try/except for OOM | CUDA memory fragmentation across loop iterations | **Subprocess worker execution** with clean OS-level VRAM reclamation |
| **Latency Profiling** | Wall-clock time on batch size 1 | Python CPU dispatch jitter dominates $<2\text{M}$ models | **CUDA Event profiling** after 50-step warmup + TorchScript tracing |

---

## 4. Phase-by-Phase Integration Plan

When executing the blueprint roadmap, integrate these fixes at each phase:

* **Phase 1 (Harness & Profiler):**
  - Implement worker execution via `subprocess`.
  - Add CUDA Event timing with warmup to `profiler.py`.
  - Run the proxy validity check: train 10 diverse models for 3 epochs vs 25 epochs to verify Spearman $\rho$.
* **Phase 2 (Primitives & Surgery):**
  - Design `blocks.py` with explicit coupled channel contracts so surgery cannot break residual additions.
  - Implement contract unit tests verifying that pruning preserves skip-connection shape compatibility.
* **Phase 3 (Baselines):**
  - Implement Random Mutation and Regularized Evolution using the deterministic spec compiler.
* **Phase 4 & 5 (LLM Gateway, Optimizer & Memory):**
  - Bypass the Coder node for Tier 1; wire the Optimizer directly to the Gate.
  - Build `store.py` around SQLite with explicit Pareto queries instead of ChromaDB vector search.
* **Phase 6 (Experiments):**
  - Report Pareto hypervolume, rank correlation, and LLM call counts alongside final accuracy.
