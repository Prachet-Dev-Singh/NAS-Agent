"""
test_gateway.py
---------------
Unit tests for LLM Gateway:
- JSON extraction from markdown code fences.
- Disk caching mechanism.
- Mock provider proposals.
- Self-healing repair loop on validation errors.
- Fallback emission on repair exhaustion.
- Audit logging into SQLite llm_calls table.
"""

import json
import os
import tempfile
import pytest

from nas_agent.llm.gateway import LLMGateway, extract_json_from_text
from nas_agent.memory.store import CandidateStore
from nas_agent.state import MacroProposal, MicroProposal


def test_extract_json_from_text():
    """Verify extraction handles markdown blocks, leading/trailing text, and raw JSON."""
    raw_1 = '```json\n{"kind": "macro", "stages": []}\n```'
    assert extract_json_from_text(raw_1) == {"kind": "macro", "stages": []}

    raw_2 = 'Here is the requested architecture:\n```\n{"kind": "micro", "parent_id": "p1"}\n```\nHope this helps!'
    assert extract_json_from_text(raw_2) == {"kind": "micro", "parent_id": "p1"}

    raw_3 = '{"kind": "macro", "stem_width": 32}'
    assert extract_json_from_text(raw_3) == {"kind": "macro", "stem_width": 32}


def test_gateway_disk_caching():
    """Verify disk cache stores and retrieves identical responses on repeat queries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gw = LLMGateway(provider="mock", cache_dir=tmpdir)
        sys_p = "You are a NAS Optimizer."
        user_p = "Propose an architecture."

        # First query: should be cache miss
        prop1, audit1 = gw.query_proposal(sys_p, user_p, run_id="r1")
        assert audit1["cache_hit"] is False
        assert isinstance(prop1, MacroProposal)

        # Second query with identical prompt: should be cache hit
        prop2, audit2 = gw.query_proposal(sys_p, user_p, run_id="r1")
        assert audit2["cache_hit"] is True
        assert prop2.model_dump() == prop1.model_dump()


def test_gateway_audit_logging():
    """Verify every query is logged to the SQLite llm_calls table."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache_dir = os.path.join(tmpdir, "cache")
        store = CandidateStore(db_path)
        gw = LLMGateway(provider="mock", cache_dir=cache_dir)

        prop, audit = gw.query_proposal(
            system_prompt="sys",
            user_prompt="user",
            run_id="run_audit_test",
            store=store,
        )

        with store._connection() as conn:
            cursor = conn.execute("SELECT * FROM llm_calls WHERE call_id = ?", (audit["call_id"],))
            row = cursor.fetchone()
            assert row is not None
            assert row["run_id"] == "run_audit_test"
            assert row["provider"] == "mock"
            assert row["tokens_in"] == 100
            assert row["tokens_out"] == 150
            assert row["cache_hit"] == 0

        store.close()


def test_gateway_repair_and_fallback():
    """Verify gateway handles broken JSON with repair attempts and falls back safely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Custom gateway subclass that produces broken JSON then fixes it
        class RepairableGateway(LLMGateway):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.call_count = 0

            def _call_provider_raw(self, sys_p, user_p):
                self.call_count += 1
                if self.call_count == 1:
                    # Invalid JSON on first attempt
                    return "INVALID JSON {{{", 50, 10
                # Fixed valid JSON on repair attempt
                return '{"kind": "macro", "stem_width": 32, "norm": "bn", "activation": "silu", "stages": [{"block": "conv", "width": 32, "expand": 2, "depth": 2, "stride": 1}, {"block": "resnet", "width": 64, "expand": 2, "depth": 2, "stride": 2}, {"block": "conv", "width": 128, "expand": 2, "depth": 2, "stride": 2}], "rationale": "Repaired proposal"}', 60, 80

        gw = RepairableGateway(provider="mock", cache_dir=tmpdir, max_repair_attempts=2)
        prop, audit = gw.query_proposal("sys", "user")

        assert isinstance(prop, MacroProposal)
        assert audit["repair_count"] == 1
        assert audit["fallback"] is False


def test_gateway_exhausted_repair_fallback():
    """Verify gateway falls back to random macro proposal when all repairs fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        class AlwaysFailingGateway(LLMGateway):
            def _call_provider_raw(self, sys_p, user_p):
                return "ALWAYS BROKEN", 20, 20

        gw = AlwaysFailingGateway(provider="mock", cache_dir=tmpdir, max_repair_attempts=2)
        prop, audit = gw.query_proposal("sys", "user")

        assert isinstance(prop, MacroProposal)
        assert audit["fallback"] is True
        assert audit["repair_count"] == 2
        assert "Fallback proposal" in prop.rationale


def test_gateway_groq_configuration():
    """Verify Groq provider initializes with proper settings."""
    gw = LLMGateway(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk_test_123")
    assert gw.provider == "groq"
    assert gw.model == "llama-3.3-70b-versatile"
    assert gw.api_key == "gsk_test_123"

