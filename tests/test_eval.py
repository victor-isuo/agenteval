"""
Test Suite — AgentEval
========================
pytest tests covering:
- Database initialization and operations
- LLM judge scoring logic
- Evaluator dataset loading
- API endpoint responses
- Regression detection logic

Run with: pytest tests/ -v
"""

import pytest
import os
import sys
import uuid
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ── Database Tests ────────────────────────────────────────────────────────────

class TestResultsStore:
    """Tests for SQLite results store operations."""

    def setup_method(self):
        """Use a temporary database for each test."""
        import src.store.results_store as store
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        store.DB_PATH = Path(self.tmp.name)
        store.initialize_db()
        self.store = store

    def teardown_method(self):
        """Clean up temp database."""
        try:
            os.unlink(self.tmp.name)
        except Exception:
            pass

    def test_initialize_db_creates_tables(self):
        """Database should have all four required tables after init."""
        conn   = self.store.get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        conn.close()

        assert "eval_runs"          in table_names
        assert "eval_cases"         in table_names
        assert "score_history"      in table_names
        assert "regression_alerts"  in table_names

    def test_create_run(self):
        """create_run should insert a record and return the run_id."""
        run_id = f"test_{uuid.uuid4().hex[:8]}"
        result = self.store.create_run(run_id, "rag", 10)
        assert result == run_id

        conn = self.store.get_connection()
        row  = conn.execute(
            "SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["eval_mode"]   == "rag"
        assert row["total_cases"] == 10
        assert row["status"]      == "running"

    def test_save_case_result(self):
        """save_case_result should persist all fields correctly."""
        run_id = f"test_{uuid.uuid4().hex[:8]}"
        self.store.create_run(run_id, "rag", 1)
        self.store.save_case_result(
            run_id=run_id, case_id="case_001", eval_mode="rag",
            query="Test query", actual_response="Test response",
            score=0.85, passed=True, latency_seconds=1.5,
            expected="expected", failure_reason="",
            judge_reasoning="Good response"
        )

        conn  = self.store.get_connection()
        cases = conn.execute(
            "SELECT * FROM eval_cases WHERE run_id = ?", (run_id,)
        ).fetchall()
        conn.close()

        assert len(cases) == 1
        assert cases[0]["score"]   == pytest.approx(0.85)
        assert cases[0]["passed"]  == 1
        assert cases[0]["case_id"] == "case_001"

    def test_complete_run_calculates_aggregates(self):
        """complete_run should correctly compute avg_score and pass counts."""
        run_id = f"test_{uuid.uuid4().hex[:8]}"
        self.store.create_run(run_id, "rag", 3)

        for i, (score, passed) in enumerate([(0.9, True), (0.8, True), (0.5, False)]):
            self.store.save_case_result(
                run_id=run_id, case_id=f"case_{i}", eval_mode="rag",
                query="q", actual_response="a",
                score=score, passed=passed, latency_seconds=1.0
            )

        self.store.complete_run(run_id)

        conn = self.store.get_connection()
        run  = conn.execute(
            "SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        conn.close()

        assert run["status"]    == "complete"
        assert run["passed"]    == 2
        assert run["failed"]    == 1
        assert run["avg_score"] == pytest.approx((0.9 + 0.8 + 0.5) / 3)

    def test_regression_detection(self):
        """Should fire regression alert when score drops > 5%."""
        # First run — baseline score 0.90
        run1 = f"run1_{uuid.uuid4().hex[:8]}"
        self.store.create_run(run1, "rag", 1)
        self.store.save_case_result(
            run_id=run1, case_id="c1", eval_mode="rag",
            query="q", actual_response="a",
            score=0.9, passed=True, latency_seconds=1.0
        )
        self.store.complete_run(run1)

        # Second run — score drops to 0.70 (22% drop, above 5% threshold)
        run2 = f"run2_{uuid.uuid4().hex[:8]}"
        self.store.create_run(run2, "rag", 1)
        self.store.save_case_result(
            run_id=run2, case_id="c1", eval_mode="rag",
            query="q", actual_response="a",
            score=0.7, passed=False, latency_seconds=1.0
        )
        self.store.complete_run(run2)

        conn   = self.store.get_connection()
        alerts = conn.execute(
            "SELECT * FROM regression_alerts WHERE run_id = ?", (run2,)
        ).fetchall()
        conn.close()

        assert len(alerts) == 1
        assert alerts[0]["drop_pct"] > 5.0

    def test_get_dashboard_summary_empty(self):
        """Dashboard summary should return None scores when no runs exist."""
        summary = self.store.get_dashboard_summary()
        assert summary["rag"]["avg_score"]         is None
        assert summary["agent"]["avg_score"]       is None
        assert summary["multi_agent"]["avg_score"] is None


# ── LLM Judge Tests ───────────────────────────────────────────────────────────

class TestLLMJudge:
    """Tests for judge scoring logic — mocked to avoid API calls in CI."""

    def test_parse_scores_valid_json(self):
        """Score parser should extract valid JSON from judge response."""
        from src.judges.llm_judge import _parse_scores
        raw    = 'Some text {"faithfulness": 0.9, "citation_accuracy": 0.8, "relevance": 0.7} Reasoning: test'
        scores = _parse_scores(raw, ["faithfulness", "citation_accuracy", "relevance"])
        assert scores["faithfulness"]       == pytest.approx(0.9)
        assert scores["citation_accuracy"]  == pytest.approx(0.8)
        assert scores["relevance"]          == pytest.approx(0.7)

    def test_parse_scores_missing_key_defaults(self):
        """Missing axis should default to 0.5."""
        from src.judges.llm_judge import _parse_scores
        raw    = '{"faithfulness": 0.9}'
        scores = _parse_scores(raw, ["faithfulness", "citation_accuracy"])
        assert scores["citation_accuracy"] == pytest.approx(0.5)

    def test_parse_scores_invalid_json_defaults(self):
        """Invalid JSON should return all defaults of 0.5."""
        from src.judges.llm_judge import _parse_scores
        scores = _parse_scores("no json here", ["faithfulness", "relevance"])
        assert scores["faithfulness"] == pytest.approx(0.5)
        assert scores["relevance"]    == pytest.approx(0.5)

    @patch("src.judges.llm_judge._call_judge")
    def test_judge_rag_response_pass(self, mock_judge):
        """Judge should return passed=True for high scores."""
        mock_judge.return_value = (
            '{"faithfulness": 0.95, "citation_accuracy": 0.90, "relevance": 0.92}'
            "\nReasoning: Excellent response."
        )
        from src.judges.llm_judge import judge_rag_response
        result = judge_rag_response(
            query="Test query",
            response="Test response with citation. Source: Manual, Page 5.",
            context="Manual Page 5: Relevant content.",
            expected="relevant content"
        )
        assert result.passed is True
        assert result.score  >= 0.70

    @patch("src.judges.llm_judge._call_judge")
    def test_judge_rag_response_fail(self, mock_judge):
        """Judge should return passed=False for low scores."""
        mock_judge.return_value = (
            '{"faithfulness": 0.3, "citation_accuracy": 0.2, "relevance": 0.4}'
            "\nReasoning: Poor citation accuracy."
        )
        from src.judges.llm_judge import judge_rag_response
        result = judge_rag_response(
            query="Test query",
            response="Hallucinated response with no citations.",
        )
        assert result.passed        is False
        assert result.score         <  0.70
        assert result.failure_reason != ""

    @patch("src.judges.llm_judge._call_judge")
    def test_judge_agent_response(self, mock_judge):
        """Agent judge should score tool selection and task completion."""
        mock_judge.return_value = (
            '{"tool_selection": 0.9, "reasoning_validity": 0.85, "task_completion": 0.9}'
            "\nReasoning: Correct tool used."
        )
        from src.judges.llm_judge import judge_agent_response
        result = judge_agent_response(
            query="Convert 100 psi to bar",
            response="100 psi = 6.895 bar",
            tools_used=["unit_converter"],
            expected_tools=["unit_converter"],
        )
        assert result.passed is True

    @patch("src.judges.llm_judge._call_judge")
    def test_judge_multi_agent_response(self, mock_judge):
        """Multi-agent judge should score delegation and synthesis."""
        mock_judge.return_value = (
            '{"delegation_accuracy": 0.9, "inter_agent_consistency": 0.85, "synthesis_quality": 0.88}'
            "\nReasoning: Good delegation."
        )
        from src.judges.llm_judge import judge_multi_agent_response
        result = judge_multi_agent_response(
            query="Full plant health report",
            final_response="All equipment operating normally.",
            agents_used=["Telemetry Agent", "Safety Agent"],
            agent_results=[
                {"agent_name": "Telemetry Agent", "response": "All normal"},
                {"agent_name": "Safety Agent",    "response": "No violations"},
            ],
        )
        assert result.passed is True


# ── Evaluator Tests ───────────────────────────────────────────────────────────

class TestEvaluators:
    """Tests for evaluator dataset loading."""

    def test_rag_default_dataset_structure(self):
        """Default RAG dataset should have required fields."""
        from src.evaluators.rag_evaluator import _default_rag_cases
        cases = _default_rag_cases()
        assert len(cases) > 0
        for case in cases:
            assert "id"    in case
            assert "query" in case

    def test_agent_default_dataset_structure(self):
        """Default agent dataset should have expected_tools field."""
        from src.evaluators.agent_evaluator import _default_agent_cases
        cases = _default_agent_cases()
        assert len(cases) > 0
        for case in cases:
            assert "id"             in case
            assert "query"          in case
            assert "expected_tools" in case

    def test_multi_agent_default_dataset_structure(self):
        """Default multi-agent dataset should have expected_agents field."""
        from src.evaluators.multi_agent_evaluator import _default_multi_agent_cases
        cases = _default_multi_agent_cases()
        assert len(cases) > 0
        for case in cases:
            assert "id"              in case
            assert "query"           in case
            assert "expected_agents" in case


# ── API Tests ─────────────────────────────────────────────────────────────────

class TestAPI:
    """Tests for FastAPI endpoint responses."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import tempfile
        from pathlib import Path
        import src.store.results_store as store

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        store.DB_PATH = Path(tmp.name)
        store.initialize_db()

        from main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Health check should return 200 with status healthy."""
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"]   == "healthy"
        assert data["database"] == "connected"

    def test_results_endpoint_empty(self, client):
        """Results endpoint should return empty list when no runs exist."""
        res = client.get("/results")
        assert res.status_code == 200
        data = res.json()
        assert "runs"  in data
        assert "total" in data

    def test_dashboard_summary_endpoint(self, client):
        """Dashboard summary should return all three eval modes."""
        res = client.get("/dashboard/summary")
        assert res.status_code == 200
        data = res.json()
        assert "rag"         in data
        assert "agent"       in data
        assert "multi_agent" in data

    def test_failures_invalid_mode(self, client):
        """Invalid eval mode should return 400."""
        res = client.get("/failures/invalid_mode")
        assert res.status_code == 400

    def test_history_invalid_mode(self, client):
        """Invalid eval mode should return 400."""
        res = client.get("/history/bad_mode")
        assert res.status_code == 400

