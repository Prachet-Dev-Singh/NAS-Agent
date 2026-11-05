## NAS-Agent Headline Comparison (Section 9.1)

> Accuracies are strictly computed over **feasible candidates** passing all parameter and MAC caps.
> Time to tau measures time to reach the calibrated quality threshold (tau = 72.96%).

| Arm | Description | Feasible Val Acc | Feasible Test Acc | Time to tau=72.96% (s) | Feasible / Total | Validity % | Pareto Count | LLM Calls |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | Random Search | **66.66%** | **66.68%** | N/A | 2 / 3 | 66.7% | 1 | 0 |
| **B** | Aging Evolution | **72.74%** | **72.71%** | N/A | 2 / 3 | 66.7% | 2 | 0 |
| **C** | LLM No-Memory | **0.00%** | **0.00%** | N/A | 0 / 6 | 0.0% | 0 | 6 |
| **D** | Full NAS-Agent | **74.72%** | **74.48%** | 234s | 2 / 6 | 33.3% | 2 | 6 |
| **E** | LLM Raw Context | **72.14%** | **72.11%** | N/A | 2 / 6 | 33.3% | 2 | 6 |
| **G** | Contextual Bandit | **72.32%** | **72.31%** | N/A | 6 / 6 | 100.0% | 1 | 0 |
