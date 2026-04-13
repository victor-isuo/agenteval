"""
RAG Evaluator — AgentEval
==========================
Evaluates the Industrial AI Copilot's RAG pipeline
across three quality dimensions:

- Faithfulness: Does the answer stay grounded in retrieved docs?
- Citation Accuracy: Are sources cited correctly with page numbers?
- Relevance: Does the answer actually address the query?

Connects to the live Industrial AI Copilot at its HuggingFace URL
and evaluates real responses — not mocked ones.

This is what "evaluation-driven development" means in production:
every deployment gets checked against a fixed test suite and
regressions are caught before users see them.
"""

import os
import time
import uuid
import json
import logging
import httpx
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from src.judges.llm_judge import judge_rag_response
from src.store.results_store import (
    create_run, save_case_result, complete_run
)

logger = logging.getLogger(__name__)

COPILOT_URL = os.getenv(
    "INDUSTRIAL_COPILOT_URL",
    "https://victorisuo-industrial-ai-copilot.hf.space/"
)

DATASET_PATH = Path(__file__).parent.parent.parent / "datasets" / "rag_eval_cases.json"


def load_rag_dataset() -> list[dict]:
    """Load RAG evaluation test cases from JSON file."""
    if not DATASET_PATH.exists():
        logger.warning(f"RAG dataset not found at {DATASET_PATH}. Using defaults.")
        return _default_rag_cases()

    with open(DATASET_PATH, "r") as f:
        return json.load(f)


def _query_copilot_rag(question: str, timeout: int = 30) -> dict:
    """
    Send a query to the Industrial AI Copilot RAG endpoint.
    Returns the full response dict or error dict.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{COPILOT_URL}/query",
                json={"question": question},
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        return {"error": "timeout", "answer": "", "sources": []}
    except Exception as e:
        return {"error": str(e), "answer": "", "sources": []}


def run_rag_evaluation(
    dataset: list[dict] = None,
    run_label: str = "",
) -> dict:
    """
    Run the full RAG evaluation suite.

    Args:
        dataset: List of test cases. Loads from JSON if not provided.
        run_label: Optional label for this run (e.g. "pre-deploy check")

    Returns:
        Summary dict with run_id, pass_rate, avg_score, failed_cases.
    """
    if dataset is None:
        dataset = load_rag_dataset()

    run_id = f"rag_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"
    create_run(run_id, "rag", len(dataset))

    logger.info(f"Starting RAG eval run {run_id} — {len(dataset)} cases")
    print(f"\n{'='*60}")
    print(f"RAG EVALUATION — {len(dataset)} test cases")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}\n")

    passed_count = 0
    failed_cases = []

    for i, case in enumerate(dataset):
        case_id = case.get("id", f"rag_{i+1:03d}")
        query = case["query"]
        expected = case.get("expected_answer", "")

        print(f"[{i+1}/{len(dataset)}] {case_id}: {query[:60]}...")

        start = time.time()
        result = _query_copilot_rag(query)
        latency = round(time.time() - start, 2)

        if "error" in result:
            save_case_result(
                run_id=run_id, case_id=case_id, eval_mode="rag",
                query=query, actual_response=f"ERROR: {result['error']}",
                score=0.0, passed=False, latency_seconds=latency,
                expected=expected, failure_reason=f"API error: {result['error']}",
            )
            print(f" ✗ API ERROR: {result['error']}")
            failed_cases.append({"case_id": case_id, "reason": result["error"]})
            continue

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        context = "\n".join(sources) if sources else ""

        # LLM judge scores the response
        judge = judge_rag_response(
            query=query, response=answer,
            context=context, expected=expected,
        )

        save_case_result(
            run_id=run_id, case_id=case_id, eval_mode="rag",
            query=query, actual_response=answer,
            score=judge.score, passed=judge.passed,
            latency_seconds=latency,
            expected=expected,
            failure_reason=judge.failure_reason,
            judge_reasoning=judge.reasoning,
        )

        status = "✓" if judge.passed else "✗"
        print(f" {status} Score: {judge.score:.3f} | Latency: {latency}s")

        if judge.passed:
            passed_count += 1
        else:
            failed_cases.append({
                "case_id": case_id,
                "score": judge.score,
                "failure_reason": judge.failure_reason,
            })

    complete_run(run_id)

    total = len(dataset)
    pass_rate = passed_count / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"RAG EVAL COMPLETE")
    print(f"Passed: {passed_count}/{total} ({pass_rate:.1%})")
    print(f"Failed: {total - passed_count}")
    if failed_cases:
        print(f"Failures:")
        for f in failed_cases:
            print(f" - {f['case_id']}: {f.get('failure_reason', '')[:80]}")
    print(f"{'='*60}\n")

    return {
        "run_id": run_id,
        "eval_mode": "rag",
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": round(pass_rate, 3),
        "failed_cases": failed_cases,
    }


def _default_rag_cases() -> list[dict]:
    """
    Default RAG test cases used if no dataset file exists.
    Covers: factual retrieval, safety procedures,
    equipment specs, edge cases.
    """
    return [
        {
            "id": "rag_001",
            "query": "What are the lockout tagout procedures for electrical equipment?",
            "expected_answer": "lockout tagout energy isolation procedure",
            "category": "safety"
        },
        {
            "id": "rag_002",
            "query": "What is the maximum allowable working pressure for a gear pump?",
            "expected_answer": "pressure specification psi",
            "category": "specs"
        },
        {
            "id": "rag_003",
            "query": "What PPE is required when handling petroleum products?",
            "expected_answer": "personal protective equipment gloves goggles",
            "category": "safety"
        },
        {
            "id": "rag_004",
            "query": "How often should bearing lubrication be performed?",
            "expected_answer": "lubrication maintenance interval",
            "category": "maintenance"
        },
        {
            "id": "rag_005",
            "query": "What are the signs of pump cavitation?",
            "expected_answer": "cavitation vibration noise suction",
            "category": "diagnosis"
        },
        {
            "id": "rag_006",
            "query": "What is the ISO standard for vibration severity on rotating equipment?",
            "expected_answer": "ISO vibration mm/s",
            "category": "standards"
        },
        {
            "id": "rag_007",
            "query": "What should I do if a centrifugal pump loses prime?",
            "expected_answer": "prime suction inspection procedure",
            "category": "diagnosis"
        },
        {
            "id": "rag_008",
            "query": "What are the operating temperature limits for electric motor windings?",
            "expected_answer": "temperature winding insulation limit",
            "category": "specs"
        },
        {
            "id": "rag_009",
            "query": "How do I perform a pressure test on a pipeline?",
            "expected_answer": "pressure test procedure hydrostatic",
            "category": "procedures"
        },
        {
            "id": "rag_010",
            "query": "What are the emergency shutdown procedures for the compressor?",
            "expected_answer": "emergency shutdown ESD compressor procedure",
            "category": "safety"
        },
    ]
