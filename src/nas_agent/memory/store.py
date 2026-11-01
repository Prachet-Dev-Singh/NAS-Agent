"""
store.py
--------
SQLite structured experience memory store for NAS-Agent (Blueprint Section 8.2).

Conventions:
- SQLite with WAL (write-ahead logging) mode and busy timeout.
- Manages candidates and llm_calls tables.
- Stamped with config_hash on every row to prevent silent drift.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from typing import Any, Dict, Iterator, List, Optional

from nas_agent.state import CandidateRecord, Status


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    run_id TEXT,
    arm TEXT,
    seed INT,
    iteration INT,
    config_hash TEXT,
    spec_hash TEXT,
    parent_id TEXT,
    proposal_type TEXT,
    technique TEXT,
    distill INT DEFAULT 0,
    block_types TEXT,
    params_m REAL,
    macs_g REAL,
    peak_vram_mb REAL,
    val_acc REAL,
    latency_ms_median REAL,
    latency_ms_p95 REAL,
    latency_mode TEXT,
    status TEXT,
    own_gpu_seconds REAL DEFAULT 0.0,
    cumulative_gpu_seconds REAL DEFAULT 0.0,
    r_hv REAL DEFAULT 0.0,
    r_viol REAL DEFAULT 0.0,
    success INT DEFAULT 0,
    fallback INT DEFAULT 0,
    is_pareto INT DEFAULT 0,
    ckpt_ref TEXT,
    error_tail TEXT,
    spec_json TEXT,
    rationale TEXT
);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id TEXT PRIMARY KEY,
    run_id TEXT,
    provider TEXT,
    model_id TEXT,
    tokens_in INT,
    tokens_out INT,
    cache_hit INT,
    repair_count INT,
    ts TEXT
);

CREATE INDEX IF NOT EXISTS idx_spec ON candidates(spec_hash);
CREATE INDEX IF NOT EXISTS idx_run_arm ON candidates(run_id, arm);
CREATE INDEX IF NOT EXISTS idx_iter ON candidates(run_id, iteration);
"""


class CandidateStore:
    """Thread-safe SQLite store for candidate evaluations and LLM audit logs."""

    def __init__(self, db_path: str = "experiments/nas_agent.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    @contextlib.contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        """Checkpoint WAL and cleanly release locks."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.close()
        except Exception:
            pass

    def insert_candidate(self, record: CandidateRecord) -> None:
        """Insert a CandidateRecord row."""
        sql = """
        INSERT INTO candidates (
            candidate_id, run_id, arm, seed, iteration, config_hash,
            spec_hash, parent_id, proposal_type, technique, distill,
            block_types, params_m, macs_g, peak_vram_mb, val_acc,
            latency_ms_median, latency_ms_p95, latency_mode, status,
            own_gpu_seconds, cumulative_gpu_seconds, r_hv, r_viol,
            success, fallback, is_pareto, ckpt_ref, error_tail,
            spec_json, rationale
        ) VALUES (
            :candidate_id, :run_id, :arm, :seed, :iteration, :config_hash,
            :spec_hash, :parent_id, :proposal_type, :technique, :distill,
            :block_types, :params_m, :macs_g, :peak_vram_mb, :val_acc,
            :latency_ms_median, :latency_ms_p95, :latency_mode, :status,
            :own_gpu_seconds, :cumulative_gpu_seconds, :r_hv, :r_viol,
            :success, :fallback, :is_pareto, :ckpt_ref, :error_tail,
            :spec_json, :rationale
        )
        """
        payload = record.model_dump()
        payload["distill"] = 1 if payload["distill"] else 0
        payload["success"] = 1 if payload["success"] else 0
        payload["fallback"] = 1 if payload["fallback"] else 0
        payload["is_pareto"] = 1 if payload["is_pareto"] else 0

        with self._connection() as conn:
            conn.execute(sql, payload)

    def update_candidate(self, candidate_id: str, updates: Dict[str, Any]) -> None:
        """Update fields on an existing candidate row."""
        if not updates:
            return

        set_clauses = [f"{k} = :{k}" for k in updates.keys()]
        sql = f"UPDATE candidates SET {', '.join(set_clauses)} WHERE candidate_id = :_id"

        payload = dict(updates)
        payload["_id"] = candidate_id

        # Boolean conversion
        for bool_key in ("distill", "success", "fallback", "is_pareto"):
            if bool_key in payload:
                payload[bool_key] = 1 if payload[bool_key] else 0

        with self._connection() as conn:
            conn.execute(sql, payload)

    def get_candidate(self, candidate_id: str) -> Optional[CandidateRecord]:
        """Fetch candidate by candidate_id."""
        sql = "SELECT * FROM candidates WHERE candidate_id = ?"
        with self._connection() as conn:
            cursor = conn.execute(sql, (candidate_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def find_by_spec_hash(self, spec_hash: str, run_id: Optional[str] = None) -> Optional[CandidateRecord]:
        """Deduplication query: check if spec_hash has already been proposed."""
        if run_id:
            sql = "SELECT * FROM candidates WHERE spec_hash = ? AND run_id = ? LIMIT 1"
            params = (spec_hash, run_id)
        else:
            sql = "SELECT * FROM candidates WHERE spec_hash = ? LIMIT 1"
            params = (spec_hash,)

        with self._connection() as conn:
            cursor = conn.execute(sql, params)
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def get_run_candidates(self, run_id: str) -> List[CandidateRecord]:
        """Retrieve all candidates evaluated in a given run ordered by iteration."""
        sql = "SELECT * FROM candidates WHERE run_id = ? ORDER BY iteration ASC"
        with self._connection() as conn:
            cursor = conn.execute(sql, (run_id,))
            return [self._row_to_record(r) for r in cursor.fetchall()]

    def log_llm_call(
        self,
        call_id: str,
        run_id: str,
        provider: str,
        model_id: str,
        tokens_in: int,
        tokens_out: int,
        cache_hit: bool,
        repair_count: int,
        ts: str,
    ) -> None:
        """Record an LLM API call invocation."""
        sql = """
        INSERT INTO llm_calls (
            call_id, run_id, provider, model_id, tokens_in,
            tokens_out, cache_hit, repair_count, ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connection() as conn:
            conn.execute(
                sql,
                (
                    call_id,
                    run_id,
                    provider,
                    model_id,
                    tokens_in,
                    tokens_out,
                    1 if cache_hit else 0,
                    repair_count,
                    ts,
                ),
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> CandidateRecord:
        d = dict(row)
        d["distill"] = bool(d.get("distill", 0))
        d["success"] = bool(d.get("success", 0))
        d["fallback"] = bool(d.get("fallback", 0))
        d["is_pareto"] = bool(d.get("is_pareto", 0))
        return CandidateRecord.model_validate(d)
