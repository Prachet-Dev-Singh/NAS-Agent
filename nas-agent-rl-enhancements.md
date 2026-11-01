# NAS Agent: RL Concepts Integration & Proposed Blueprint Changes

> **Reference Document:** [nas-agent-blueprint-v1.0.md](file:///c:/Users/prach/Downloads/NAS-Agent/nas-agent-blueprint-v1.0.md)  
> **Purpose:** Standalone specification of the proposed Reinforcement Learning (RL) concept integrations, their architectural impacts, and exact proposed modifications to the blueprint for external review.

---

## 1. Executive Summary: Why These RL Concepts (And Why Not Full RL)?

### What We Reject: Full Deep RL (PPO / REINFORCE Policy Networks)
* **Sample Starvation:** Policy-gradient NAS requires thousands of evaluated architectures to train a neural policy network from scratch. Our budget is fixed at 7,200 GPU-seconds (~80–100 candidates per run), where policy gradient methods degenerate to random exploration.
* **Reward Noise & Sparsity:** Most sampled architectures yield zero Pareto improvement, and proxy validation accuracy contains measurement noise ($\sigma_{\text{acc}}$).
* **Frozen Free LLM APIs:** On black-box free endpoints (Google AI Studio, Groq, Cerebras), backpropagation through model weights is impossible.

### What We Adopt: 3 High-Value, Zero/Low-Cost RL Concepts
1. **Reward Shaping:** Reward defined as Hypervolume Improvement per GPU-second ($\text{HVI} / \text{sec}$), plus a shaped credit term for moving infeasible parents closer to the boundary while maintaining validation accuracy above the floor.
2. **Contextual Multi-Armed Bandit (Thompson Sampling Baseline — Arm G):** A zero-LLM, lightweight RL baseline learning Beta-Bernoulli posteriors over operators conditioned on parent feasibility state.
3. **Bandit-Style Early Stopping (Conservative Median Stopping):** Terminating hopelessly lagging candidates at epoch 5 of 8 to conserve GPU-seconds for high-promise models, validated against slow-starting Attention/SSM bias.

---

## 2. Detailed Breakdown of Architectural Changes

### Change 1: Reward Shaping via Hypervolume Improvement per GPU-Second
* **Problem:** In standard NAS, raw validation accuracy ($\Delta \text{acc}$) ignores computational cost, parameter constraints, and multi-objective Pareto trade-offs. Most candidates fail to improve the frontier, leaving a sparse $0$ signal.
* **Solution:**
  * Define **Hypervolume Improvement (HVI)**:
    $$\Delta \text{HV}_t = \max(0, \text{Hypervolume}(F_t) - \text{Hypervolume}(F_{t-1}))$$
    where $F_t$ is the current set of non-dominated feasible candidates in normalized space.
  * Define **Cost-Normalized Base Reward**:
    $$R_{\text{base}} = \frac{\Delta \text{HV}_t}{\text{own\_gpu\_seconds}}$$
  * Add **Constraint-Shaped Credit ($R_{\text{shaped}}$):** For micro-moves operating on an infeasible parent (e.g., $VRAM > 2048\text{ MB}$):
    * If candidate validation accuracy stays above `min_val_acc`:
      $$R_{\text{shaped}} = \gamma \cdot \max\left(0, \text{violation\_ratio}_{\text{parent}} - \text{violation\_ratio}_{\text{child}}\right)$$
    * Otherwise, $R_{\text{shaped}} = 0$.
  * Total Reward: $R = R_{\text{base}} + R_{\text{shaped}}$.
* **Integration:**
  * Calculated deterministically by the `Record` node.
  * Stored in the `candidates` table as `hv_before`, `hv_after`, and `reward`.
  * Passed to the Optimizer prompt context in the "Technique Statistics" section (replacing raw $\Delta \text{acc}$ with success rate and mean $HVI / \text{sec}$).

---

### Change 2: Arm G — Contextual Bandit via Thompson Sampling (No LLM)
* **Problem:** Comparing the LLM agent only against Random Search (Arm A) and Regularized Evolution (Arm B) leaves open the critique: *"Could a simple 50-line statistical bandit achieve the same result without an LLM?"*
* **Solution:** Introduce **Arm G** as a rigorous, non-LLM learning baseline.
  * **Arms of the Bandit:** The discrete mutation operators (`conv`, `dwsep`, `resnet`, `attn_lite`, `ssm`, `svd_lowrank`, `prune_internal`, `widen_internal`, `insert_block`).
  * **Context State:** A discrete tuple:
    $$\text{Context} = (\text{parent\_feasible} \in \{0, 1\}, \text{primary\_violated\_constraint} \in \{\text{none}, \text{params}, \text{macs}, \text{vram}\})$$
  * **Likelihood & Prior:** Beta-Bernoulli conjugate distributions $\text{Beta}(\alpha_k, \beta_k)$ per operator $k$ within each context.
    * Success is defined as: $R > 0$ (either non-zero HVI or reduction in violation ratio above accuracy floor).
  * **Decision Rule (Thompson Sampling):**
    * Sample success probability $\theta_k \sim \text{Beta}(\alpha_k, \beta_k)$.
    * Select operator $k^* = \arg\max_k \frac{\theta_k}{\mathbb{E}[\text{cost}_k]}$, where $\mathbb{E}[\text{cost}_k]$ is empirical mean GPU-seconds for that operator.
* **Compute Impact:** Adds 1 arm $\times$ 3 seeds $\times$ 2.0 hours = **6.0 GPU-hours**. Total project compute shifts from ~40–45 to **~46–51 GPU-hours**.

---

### Change 3: Conservative Median Early Stopping in the Fixed Harness
* **Problem:** Infeasible or hopelessly underperforming candidates waste precious GPU-seconds completing all 8 epochs.
* **Risk:** Attention and SSM blocks take longer to warm up; aggressive early stopping (e.g. at epoch 2 or 3) would unfairly kill them, resurrecting proxy optimizer bias.
* **Solution:**
  * **Conservative Trigger:** Check validation accuracy **strictly after epoch 5 of 8**.
  * **Threshold:** Kill candidate only if its epoch-5 validation accuracy falls below the **25th percentile** of all epoch-5 accuracies recorded in the current run.
  * **Validation in Phase 3:** Before freezing `config_hash`, test the early-stopping rule on Phase 3 reference runs. Verify that $<5\%$ of models discarded at epoch 5 would have reached the top half of the population at epoch 8.
  * **State & Accounting:** Candidate is assigned status `'early_stopped'`. Consumed GPU-seconds up to epoch 5 are billed to the run budget.

---

## 3. Exact Blueprint Sections to Update

Below are the exact updates mapped to the sections of `nas-agent-blueprint-v1.0.md`:

### Section 2.2 — Experimental Arms Table
```markdown
| Arm | Description |
|---|---|
| **A** | Random search (macro + micro operators) |
| **B** | Regularized (aging) evolution, population 10, tournament size 3 |
| **C** | LLM Optimizer, no memory (sees only last result) |
| **D** | Full agent: LLM Optimizer + structured SQLite Pareto memory |
| **E** | LLM Optimizer + raw in-context history (last 40 rows) |
| **G** | Contextual Bandit: Thompson sampling (Beta-Bernoulli) over operators conditioned on feasibility context (no LLM) |
```

### Section 3.3 — Compute Plan Table
```markdown
| Item | Calculation | GPU-hours |
|---|---|---|
| LR-multiplier sweep | 5 families x 3 LRs x ~90s | 0.4 |
| Seed noise floor | 5 runs x ~90s | 0.1 |
| Proxy-validity check | 20 models x (~90s proxy + ~560s reference) | 3.6 |
| Main runs | 6 arms x 3 seeds x 2.0 h | 36.0 |
| Pilots / dev | 3 x 0.5 h | 1.5 |
| Overhead (cold starts, retries) | ~15% | ~6.0 |
| **Total** | | **~46-51** |
```

### Section 5.1 & 5.2 — Status Enum & Record Node
* Update status enum to include `early_stopped`:
  ```python
  Status = Literal[
      "feasible", "infeasible", "early_stopped", "rejected_static", 
      "gate_fail", "duplicate", "invalid", "oom", "timeout", "nan", "crash"
  ]
  ```
* Update `Record` node responsibilities:
  * Computes $\Delta \text{HV}$, shaped constraint penalty reduction, and final scalar reward $R$.
  * Updates SQLite table with `hv_before`, `hv_after`, and `reward`.

### Section 6.2 — Fixed Training Configuration
* Add early stopping specifications to the frozen table:
  ```markdown
  | Early stopping | Median-stopping: evaluated at epoch 5 of 8; terminate if val_acc < 25th percentile of historical epoch-5 accuracies. Validated in Phase 3. |
  ```

### Section 8.1 & 8.2 — SQLite Schema & Optimizer Prompt Context
* Update `candidates` schema:
  ```sql
  ALTER TABLE candidates ADD COLUMN hv_before REAL;
  ALTER TABLE candidates ADD COLUMN hv_after REAL;
  ALTER TABLE candidates ADD COLUMN reward REAL;
  ```
* Update Section 8.2 item 4 ("Technique statistics"):
  * Instead of raw $\Delta \text{acc}$, context provides:
    * Success rate: share of attempts yielding $R > 0$.
    * Mean reward rate: average $\text{HVI} / \text{own\_gpu\_seconds}$ for that technique over the last 15 attempts.

### Section 10 — Repository Layout
* Add the bandit module:
  ```
  src/nas_agent/search/
    operators.py
    random_search.py
    evolution.py
    bandit.py             # Contextual Thompson-Sampling Bandit (Arm G)
  ```

### Section 11 — Roadmap Milestones
* **Phase 3 Acceptance:** Add validation of the epoch-5 early stopping rule against proxy correlation curves.
* **Phase 5 Acceptance:** Implement and validate Arm G (Bandit) alongside Arms A and B before proceeding to LLM nodes.

---

## 4. Pros and Cons Matrix

| Dimension | Standard Blueprint v1.0 | With Proposed RL Enhancements |
|---|---|---|
| **Scientific Rigor** | Strong (competes against Random & Evolution). | **Stronger** (competes against a statistical learning controller; proves LLM adds reasoning beyond Bayesian bandits). |
| **Signal Density** | Sparse ($\Delta \text{acc}$ ignores constraint reductions and multi-objective tradeoffs). | **High** ($\text{HVI} / \text{sec}$ + shaped violation credit directly maps to Pareto optimization). |
| **Compute Efficiency** | Fixed 8 epochs per candidate. | **15–25% GPU savings** from conservative median stopping on doomed runs. |
| **Total GPU Footprint** | 40–45 GPU-hours. | 46–51 GPU-hours (+6 hours to include 3 seeds of Arm G). |
| **Implementation Complexity** | Low. | Very modest (+~60 lines in `bandit.py`, +~15 lines in `record.py` for HVI). |
