"""
Results Store — AgentEval
==========================
SQLite database layer for storing all evaluation results.

Every eval run — RAG, single agent, multi-agent — writes
its results here. The dashboard reads from here to show
trends, regressions, and cost tracking over time.

Why SQLite:
- Zero infrastructure — single file, no server
- Built into Python — no installation needed
- Fast enough for eval workloads
- Portable — the entire history travels with the repo
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv()


DB_PATH = Path(__file__).parent.parent.parent / "agenteval.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory for dict-like access."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    """
    Create all tables if they don't exist.
    Called once at startup.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── Eval Runs ──────────────────────────────────────────────────────────
    # Each run is one complete evaluation pass (e.g. "RAG eval - 20 cases")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL UNIQUE,
            eval_mode TEXT NOT NULL, -- 'rag', 'agent', 'multi_agent'
            started_at TEXT NOT NULL,
            finished_at TEXT,
            total_cases INTEGER NOT NULL DEFAULT 0,
            passed INTEGER NOT NULL DEFAULT 0,
            failed INTEGER NOT NULL DEFAULT 0,
            avg_score REAL,
            avg_latency REAL,
            total_cost REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL DEFAULT 'running' -- 'running', 'complete', 'failed'
        )
    """)

    # ── Eval Cases ─────────────────────────────────────────────────────────
    # Each individual test case result within a run
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eval_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            eval_mode TEXT NOT NULL,
            query TEXT NOT NULL,
            expected TEXT,
            actual_response TEXT,
            score REAL,
            passed INTEGER NOT NULL DEFAULT 0, -- 0 or 1
            latency_seconds REAL,
            cost_usd REAL NOT NULL DEFAULT 0.0,
            failure_reason TEXT,
            judge_reasoning TEXT,
            evaluated_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
        )
    """)

    # ── Score History ───────────────────────────────────────────────────────
    # Aggregated daily scores for trend charts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS score_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            eval_mode TEXT NOT NULL,
            avg_score REAL NOT NULL,
            pass_rate REAL NOT NULL,
            total_runs INTEGER NOT NULL DEFAULT 1,
            UNIQUE(date, eval_mode)
        )
    """)

    # ── Regression Alerts ──────────────────────────────────────────────────
    # Fired when score drops below threshold vs previous run
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS regression_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            eval_mode TEXT NOT NULL,
            previous_score REAL NOT NULL,
            current_score REAL NOT NULL,
            drop_pct REAL NOT NULL,
            threshold_pct REAL NOT NULL,
            alerted_at TEXT NOT NULL,
            acknowledged INTEGER NOT NULL DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at: {DB_PATH}")


# ── Write Operations ──────────────────────────────────────────────────────────

def create_run(run_id: str, eval_mode: str, total_cases: int) -> str:
    """Create a new eval run record. Returns run_id."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO eval_runs (run_id, eval_mode, started_at, total_cases, status)
        VALUES (?, ?, ?, ?, 'running')
    """, (run_id, eval_mode, datetime.utcnow().isoformat(), total_cases))
    conn.commit()
    conn.close()
    return run_id


def save_case_result(
    run_id: str,
    case_id: str,
    eval_mode: str,
    query: str,
    actual_response: str,
    score: float,
    passed: bool,
    latency_seconds: float,
    cost_usd: float = 0.0,
    expected: str = "",
    failure_reason: str = "",
    judge_reasoning: str = "",
):
    """Save a single test case result."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO eval_cases (
            run_id, case_id, eval_mode, query, expected, actual_response,
            score, passed, latency_seconds, cost_usd,
            failure_reason, judge_reasoning, evaluated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id, case_id, eval_mode, query, expected, actual_response,
        score, 1 if passed else 0, latency_seconds, cost_usd,
        failure_reason, judge_reasoning, datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()


def complete_run(run_id: str):
    """
    Finalise a run — compute aggregate stats from its cases
    and update the run record.
    """
    conn = get_connection()

    cases = conn.execute("""
        SELECT score, passed, latency_seconds, cost_usd
        FROM eval_cases WHERE run_id = ?
    """, (run_id,)).fetchall()

    if not cases:
        conn.close()
        return

    scores = [c["score"] for c in cases if c["score"] is not None]
    latencies = [c["latency_seconds"] for c in cases if c["latency_seconds"] is not None]
    passed = sum(1 for c in cases if c["passed"] == 1)
    total = len(cases)
    total_cost = sum(c["cost_usd"] for c in cases)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    conn.execute("""
        UPDATE eval_runs SET
            finished_at = ?,
            passed = ?,
            failed = ?,
            avg_score = ?,
            avg_latency = ?,
            total_cost = ?,
            status = 'complete'
        WHERE run_id = ?
    """, (
        datetime.utcnow().isoformat(),
        passed, total - passed,
        avg_score, avg_latency, total_cost,
        run_id
    ))

    # Update score history for trend charts
    today = datetime.utcnow().strftime("%Y-%m-%d")
    eval_mode = conn.execute(
        "SELECT eval_mode FROM eval_runs WHERE run_id = ?", (run_id,)
    ).fetchone()["eval_mode"]

    pass_rate = passed / total if total > 0 else 0.0

    conn.execute("""
        INSERT INTO score_history (date, eval_mode, avg_score, pass_rate, total_runs)
        VALUES (?, ?, ?, ?, 1)
        ON CONFLICT(date, eval_mode) DO UPDATE SET
            avg_score = (avg_score * total_runs + excluded.avg_score) / (total_runs + 1),
            pass_rate = (pass_rate * total_runs + excluded.pass_rate) / (total_runs + 1),
            total_runs = total_runs + 1
    """, (today, eval_mode, avg_score, pass_rate))

    conn.commit()

    # Check for regression
    _check_regression(conn, run_id, eval_mode, avg_score)

    conn.close()


def _check_regression(
    conn: sqlite3.Connection,
    run_id: str,
    eval_mode: str,
    current_score: float,
    threshold_pct: float = 5.0
):
    """
    Compare current run score to the previous run for the same mode.
    Fire a regression alert if score dropped by more than threshold_pct.
    """
    prev = conn.execute("""
        SELECT avg_score FROM eval_runs
        WHERE eval_mode = ? AND status = 'complete' AND run_id != ?
        ORDER BY finished_at DESC LIMIT 1
    """, (eval_mode, run_id)).fetchone()

    if not prev or prev["avg_score"] is None:
        return

    previous_score = prev["avg_score"]
    if previous_score == 0:
        return

    drop_pct = ((previous_score - current_score) / previous_score) * 100

    if drop_pct >= threshold_pct:
        conn.execute("""
            INSERT INTO regression_alerts (
                run_id, eval_mode, previous_score, current_score,
                drop_pct, threshold_pct, alerted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, eval_mode, previous_score, current_score,
            drop_pct, threshold_pct, datetime.utcnow().isoformat()
        ))
        print(f"⚠ REGRESSION ALERT: {eval_mode} score dropped {drop_pct:.1f}% "
              f"({previous_score:.3f} → {current_score:.3f})")


# ── Read Operations ───────────────────────────────────────────────────────────

def get_recent_runs(limit: int = 20) -> list[dict]:
    """Get the most recent eval runs across all modes."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM eval_runs
        ORDER BY started_at DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_run_cases(run_id: str) -> list[dict]:
    """Get all test case results for a specific run."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM eval_cases WHERE run_id = ?
        ORDER BY evaluated_at ASC
    """, (run_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_score_history(eval_mode: str, days: int = 30) -> list[dict]:
    """Get score trend for the last N days for a given eval mode."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM score_history
        WHERE eval_mode = ?
        ORDER BY date DESC LIMIT ?
    """, (eval_mode, days)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dashboard_summary() -> dict:
    """
    Aggregate summary for the dashboard home page.
    Returns latest scores, pass rates, and regression alerts
    across all three eval modes.
    """
    conn = get_connection()

    summary = {}

    for mode in ["rag", "agent", "multi_agent"]:
        latest = conn.execute("""
            SELECT avg_score, passed, failed, avg_latency, total_cost, finished_at
            FROM eval_runs
            WHERE eval_mode = ? AND status = 'complete'
            ORDER BY finished_at DESC LIMIT 1
        """, (mode,)).fetchone()

        summary[mode] = dict(latest) if latest else {
            "avg_score": None, "passed": 0, "failed": 0,
            "avg_latency": None, "total_cost": 0.0, "finished_at": None
        }

    # Unacknowledged regression alerts
    alerts = conn.execute("""
        SELECT * FROM regression_alerts
        WHERE acknowledged = 0
        ORDER BY alerted_at DESC
    """).fetchall()

    summary["regression_alerts"] = [dict(a) for a in alerts]

    # Total runs and total cost
    totals = conn.execute("""
        SELECT COUNT(*) as total_runs,
               SUM(total_cost) as total_cost
        FROM eval_runs WHERE status = 'complete'
    """).fetchone()

    summary["totals"] = dict(totals) if totals else {"total_runs": 0, "total_cost": 0.0}

    conn.close()
    return summary


def get_failure_analysis(eval_mode: str, limit: int = 10) -> list[dict]:
    """
    Get recent failed cases with their failure reasons.
    Used for the failure analysis section of the dashboard.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT c.case_id, c.query, c.score, c.failure_reason,
               c.judge_reasoning, c.evaluated_at, r.run_id
        FROM eval_cases c
        JOIN eval_runs r ON c.run_id = r.run_id
        WHERE c.eval_mode = ? AND c.passed = 0
        ORDER BY c.evaluated_at DESC LIMIT ?
    """, (eval_mode, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

