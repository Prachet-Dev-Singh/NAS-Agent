"""
test_analysis.py
----------------
Unit and acceptance tests for Analysis and Post-Hoc Evaluation (Phase 8):
- Any-time hypervolume extraction from SQLite runs.
- Comparative headline metric calculation (HV at 25/50/100%, Best Val Acc, time-to-tau).
- Summary table generation.
- Post-hoc Pareto benchmark execution with mock data.
"""

import json
import os
import tempfile
import pytest
import pandas as pd

from nas_agent.config import load_config
from nas_agent.eval.final_benchmark import benchmark_final_pareto
from nas_agent.memory.store import CandidateStore
from nas_agent.state import CandidateRecord
from experiments.analyze import (
    analyze_run,
    compute_anytime_hypervolume,
    generate_plots,
    print_markdown_summary,
)


@pytest.fixture
def populated_run_store():
    """Create a temporary SQLite DB populated with a realistic trajectory of candidates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "nas_agent.db")
        store = CandidateStore(db_path)

        dummy_spec = {
            "kind": "macro",
            "stem_width": 32,
            "norm": "bn",
            "activation": "silu",
            "stages": [
                {"block": "conv", "width": 32, "expand": 2, "depth": 1, "stride": 1},
                {"block": "conv", "width": 64, "expand": 2, "depth": 1, "stride": 2},
                {"block": "conv", "width": 128, "expand": 2, "depth": 1, "stride": 2},
            ],
            "rationale": "dummy",
        }

        # Candidate 1: Feasible acc=0.65 (below tau 0.72)
        store.insert_candidate(
            CandidateRecord(
                candidate_id="c1",
                run_id="run_d_test",
                arm="D",
                seed=42,
                iteration=1,
                config_hash="0a0f70ce47679bb0",
                spec_hash="h1",
                proposal_type="macro",
                technique="from_scratch",
                block_types="conv",
                params_m=2.0,
                macs_g=0.20,
                peak_vram_mb=800.0,
                val_acc=0.65,
                status="feasible",
                own_gpu_seconds=50.0,
                is_pareto=True,
                spec_json=json.dumps(dummy_spec),
            )
        )

        # Candidate 2: Infeasible (params > cap)
        store.insert_candidate(
            CandidateRecord(
                candidate_id="c2",
                run_id="run_d_test",
                arm="D",
                seed=42,
                iteration=2,
                config_hash="0a0f70ce47679bb0",
                spec_hash="h2",
                proposal_type="macro",
                technique="from_scratch",
                block_types="conv",
                params_m=12.0,
                macs_g=1.50,
                peak_vram_mb=900.0,
                val_acc=0.80,
                status="infeasible",
                own_gpu_seconds=60.0,
                is_pareto=False,
                spec_json=json.dumps(dummy_spec),
            )
        )

        # Candidate 3: Feasible acc=0.75 (above tau 0.7296)
        store.insert_candidate(
            CandidateRecord(
                candidate_id="c3",
                run_id="run_d_test",
                arm="D",
                seed=42,
                iteration=3,
                config_hash="0a0f70ce47679bb0",
                spec_hash="h3",
                proposal_type="macro",
                technique="from_scratch",
                block_types="conv",
                params_m=3.0,
                macs_g=0.30,
                peak_vram_mb=850.0,
                val_acc=0.75,
                status="feasible",
                own_gpu_seconds=50.0,
                is_pareto=True,
                spec_json=json.dumps(dummy_spec),
            )
        )

        yield db_path, "run_d_test"
        store.close()


def test_anytime_hypervolume_monotonicity(populated_run_store):
    """Verify that hypervolume is monotonically non-decreasing along the cumulative compute axis."""
    db_path, run_id = populated_run_store
    cfg = load_config()

    from experiments.analyze import load_candidates_df
    df = load_candidates_df(db_path)

    gpu_seq, hv_seq = compute_anytime_hypervolume(df, cfg.budget.params_cap_m, cfg.budget.macs_cap_g)
    assert len(gpu_seq) == 3
    assert gpu_seq == [50.0, 110.0, 160.0]

    # Hypervolume must be non-decreasing
    for i in range(1, len(hv_seq)):
        assert hv_seq[i] >= hv_seq[i - 1]


def test_analyze_run_metrics(populated_run_store):
    """Verify calculated run metrics match ground truth expectations."""
    db_path, run_id = populated_run_store
    cfg = load_config()

    result = analyze_run(
        run_name="test_run",
        db_path=db_path,
        caps_m=cfg.budget.params_cap_m,
        caps_g=cfg.budget.macs_cap_g,
        tau=cfg.calibration.tau,
    )

    assert result["arm"] == "D"
    assert result["seed"] == 42
    assert result["total_candidates"] == 3
    assert result["total_gpu_seconds"] == 160.0
    assert result["acc_100"] == 0.75
    # c3 crossed tau at cumulative time 160.0s
    assert result["time_to_tau"] == 160.0
    assert result["pareto_count"] == 2
    assert result["validity_rate"] == 1.0  # 3/3 were feasible or infeasible (0 errors)


def test_markdown_and_plots_generation(populated_run_store):
    """Verify markdown table and figure generation."""
    db_path, run_id = populated_run_store
    cfg = load_config()

    result = analyze_run(
        run_name="test_run",
        db_path=db_path,
        caps_m=cfg.budget.params_cap_m,
        caps_g=cfg.budget.macs_cap_g,
        tau=cfg.calibration.tau,
    )

    md_table = print_markdown_summary([result])
    assert "Full NAS-Agent" in md_table
    assert "75.00%" in md_table

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = os.path.join(tmpdir, "test_plot.png")
        generate_plots([result], out_path=plot_path)
        assert os.path.exists(plot_path)
        assert os.path.getsize(plot_path) > 1000


def test_final_pareto_benchmark(populated_run_store):
    """Verify post-hoc Pareto benchmark runs with mock data on CPU."""
    db_path, run_id = populated_run_store

    bench_results = benchmark_final_pareto(
        db_path=db_path,
        run_id=run_id,
        device="cpu",
        mock_data=True,
    )

    assert len(bench_results) == 2  # c1 and c3 are pareto
    for r in bench_results:
        assert r["test_acc"] > 0.0
        assert r["latency_ms_median"] > 0.0
        assert r["latency_mode"] in ("eager", "graph")
