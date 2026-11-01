# Self-Improving Neural Architecture Search Agent (NAS-Agent v1.1)

> An LLM-driven, constraint-aware architecture search system on LangGraph. A single LLM Optimizer proposes macro-topologies or surgical micro-compressions; a deterministic compiler, a fixed subprocess-isolated harness, and a structured SQLite experience memory handle execution and Pareto analysis.

---

## 1. Overview & Research Goal

Under an **equal GPU-seconds budget (7,200s per seed)**, does an LLM Optimizer equipped with a structured Pareto memory find higher-performing feasible architectures than:
1. **Arm A:** Uniform Random Search
2. **Arm B:** Regularized (Aging) Evolution (Real et al., 2019)
3. **Arm C:** LLM No-Memory Ablation
4. **Arm D:** Full NAS-Agent (Structured Pareto Memory + Near-Feasible Checkpoint Refinement)
5. **Arm E:** LLM Raw In-Context History Ablation (40-candidate unaggregated history)
6. **Arm G:** Contextual Thompson-Sampling Bandit without LLM (Beta-Bernoulli with $\gamma = 0.95$)

---

## 2. Key Architecture & Design Invariants

| Principle | Failure Mode Prevented |
|---|---|
| **Fixed Training Harness** | Reward hacking, data leakage, skipped validation, learning rate tampering. |
| **Uniform Block Contract** ($y = x + \text{out\_proj}(\text{mixer}(\text{in\_proj}(x)))$) | Incompatible residual shapes; enables universal channel pruning, widening, and low-rank factorisation. |
| **Subprocess Isolation** | PyTorch CUDA OOM memory fragmentation, hung worker processes, DataLoader thread leakage. |
| **Decoupled Pareto Rewards** | Replaces sparse accuracy deltas with hypervolume improvement per GPU-second ($r_{\text{hv}}$) and constraint violation reduction credit ($r_{\text{viol}}$). |
| **Frozen Configuration (`config_hash`)** | Prevents experimental drift between baseline runs and ablation arms. |
| **Equal Compute Stance** | Search budget is bounded by cumulative own GPU-seconds, not candidate count (micro edits cost less compute than macro training from scratch). |

---

## 3. Search Space & Primitives

All blocks satisfy the contract $y = x + \text{out\_proj}(\text{mixer}(\text{in\_proj}(x)))$ with identity zero-initialisation:
- `conv`: Standard 3x3 depthwise/pointwise convolutional block.
- `dwsep`: Inverted bottleneck depthwise separable block.
- `resnet`: Classic pre-activation residual block.
- `attn_lite`: Lightweight single-head spatial reduction attention.
- `ssm`: Mamba-style selective state-space sequence mixer.

### Surgical Micro-Operators (Function-Preserving Checkpoint Surgery)
- **SVD Low-Rank (`svd_lowrank`):** Truncated-SVD parameter reduction.
- **L1 Channel Pruning (`prune_internal`):** Coupled internal channel trimming preserving stage boundaries.
- **Zero-Init Widening (`widen_internal`):** Expands hidden dimension with exact mathematical function preservation ($\max |y - x| < 10^{-6}$).
- **Zero-Init Insertion (`insert_block`):** Appends identity block at initialisation.
- **Self-Distillation (`distill`):** Optional KD guidance from parent checkpoint ($L = \alpha T^2 \text{KL} + (1-\alpha)\text{CE}$).

---

## 4. Installation & Verification

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support (CPU mode supported for verification and mock runs)

### Setup
```bash
git clone https://github.com/your-username/NAS-Agent.git
cd NAS-Agent
pip install -e .
```

### Run Comprehensive Test Suite
The repository includes **281 automated unit and integration tests** covering all 8 phases:
```bash
python -m pytest tests/ -v
```

---

## 5. Running Experiments

### Quick Development Run (Mock Data / CPU Mode)
```bash
python -m nas_agent.runner --arm D --seed 42 --budget-seconds 60 --max-iterations 3 --mock-data --device cpu
```

### Full Production Experiment (CUDA GPU)
```bash
# Set your LLM provider API key (OpenRouter, OpenAI, or Gemini)
export OPENROUTER_API_KEY="your-api-key"

# Run Arm D (Full NAS-Agent) for 7,200 GPU-seconds
python -m nas_agent.runner --arm D --seed 42 --budget-seconds 7200 --provider openrouter
```

### Available Baseline Arms
```bash
python -m nas_agent.runner --arm A --seed 42  # Uniform Random Search
python -m nas_agent.runner --arm B --seed 42  # Regularized Aging Evolution
python -m nas_agent.runner --arm C --seed 42  # LLM No-Memory
python -m nas_agent.runner --arm D --seed 42  # Full NAS-Agent
python -m nas_agent.runner --arm E --seed 42  # LLM Raw Context History
python -m nas_agent.runner --arm G --seed 42  # Contextual Thompson Bandit
```

---

## 6. Post-Hoc Benchmarking & Analysis

### Post-Hoc Pareto Front Evaluation
Measures latency using CUDA events (50 warmup iterations, 200 repetitions at batch size 1) and untouched CIFAR-10 test set accuracy:
```bash
python -m nas_agent.eval.final_benchmark --db experiments/runs/<run_id>/nas_agent.db --run-id <run_id>
```

### Results Extraction & Headline Comparison
Generates anytime hypervolume trajectories, summary tables, and comparison plots:
```bash
python experiments/analyze.py
```
Output files produced in `experiments/results/`:
- `summary.md`: Publication-ready comparison table across arms.
- `runs_summary.csv`: Full metric aggregation per seed.
- `hypervolume_curves.png`: Feasible hypervolume vs cumulative GPU-seconds curves.

---

## 7. Repository Layout

```
nas-agent/
├── configs/
│   ├── v1.yaml                     # Frozen configuration with cryptographic hash
│   └── arms/                       # Arm-specific hyperparameters (A, B, C, D, E, G)
├── src/nas_agent/
│   ├── config.py                   # Pydantic schemas, validation, and tamper detection
│   ├── graph.py                    # LangGraph closed-loop search orchestration
│   ├── runner.py                   # CLI entrypoint for experiments
│   ├── state.py                    # Proposal schemas and LangGraph AgentState
│   ├── lib/                        # Primitive blocks, contract tests, analytical counters
│   │   ├── blocks.py               # 5 uniform blocks under residual contract
│   │   ├── contracts.py            # C1-C4 contract tests
│   │   ├── counters.py             # Analytical MACs/params counters
│   │   ├── distill.py              # Self-distillation loss
│   │   ├── insert.py               # Zero-init block insertion
│   │   ├── lowrank.py              # SVD low-rank factorisation
│   │   ├── prune.py                # Coupled L1 internal pruning
│   │   └── widen.py                # Zero-init internal channel widening
│   ├── harness/                    # Subprocess worker and hardware isolation
│   │   ├── data.py                 # GPU-resident CIFAR-10 data loaders
│   │   ├── executor.py             # Local and Modal process group execution
│   │   ├── metrics.py              # Atomic JSON IO and CUDA event latency timing
│   │   └── worker.py               # Standalone worker with top-level error catching
│   ├── llm/                        # Multi-provider gateway and prompt formatters
│   │   ├── gateway.py              # Disk cache, rate-limiting, repair retry loop
│   │   └── prompts.py              # SQL memory context formatters for Arms C, D, E
│   ├── memory/                     # Structured experience database
│   │   ├── store.py                # SQLite WAL candidate and LLM audit store
│   │   └── queries.py              # Pareto front extraction and decoupled rewards
│   ├── nodes/                      # LangGraph deterministic and LLM nodes
│   │   ├── compiler.py             # Macro and micro model compilation
│   │   ├── gate.py                 # Analytical static budget & CPU smoke test gate
│   │   ├── optimizer.py            # Multi-arm proposal generator
│   │   ├── profiler.py             # Hardware cap metric validation
│   │   ├── record.py               # Reward calculation and experience recording
│   │   ├── trainer.py              # Subprocess worker execution dispatch
│   │   └── validate.py             # Schema parsing and spec_hash deduplication
│   ├── search/                     # Baseline search strategies
│   │   ├── bandit.py               # Arm G Thompson-sampling bandit
│   │   ├── evolution.py            # Arm B regularized aging evolution
│   │   ├── operators.py            # 8 shared macro and micro mutation operators
│   │   └── random_search.py        # Arm A uniform random search
│   └── eval/                       # Hypervolume and post-hoc benchmarking
│       ├── hypervolume.py          # Exact 2D/3D hypervolume calculations
│       └── final_benchmark.py      # Untouched test set and latency evaluation
├── tests/                          # 281 unit, contract, and integration tests
└── experiments/
    └── analyze.py                  # Statistical analysis and anytime curve generation
```
