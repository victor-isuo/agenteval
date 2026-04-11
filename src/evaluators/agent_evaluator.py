"""
Agent Evaluator — AgentEval
=============================
Evaluates the Industrial AI Copilot's single LangGraph agent
across three quality dimensions:

- Tool Selection:      Did the agent call the right tools?
- Reasoning Validity:  Is the reasoning chain logical and grounded?
- Task Completion:     Did the agent fully answer the query?

Connects to the live /agent endpoint and evaluates real responses.
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

from src.judges.llm_judge import judge_agent_response
from src.store.results_store import (
    create_run, save_case_result, complete_run
)

logger      = logging.getLogger(__name__)
COPILOT_URL = os.getenv(
    "INDUSTRIAL_COPILOT_URL",
    "https://victorisuo-industrial-ai-copilot.hf.space"
)
DATASET_PATH = (
    Path(__file__).parent.parent.parent / "datasets" / "agent_eval_cases.json"
)


def load_agent_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        logger.warning("Agent dataset not found. Using defaults.")
        return _default_agent_cases()
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


def _query_copilot_agent(question: str, timeout: int = 60) -> dict:
    """Query the live /agent endpoint."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{COPILOT_URL}/agent",
                json={"question": question},
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        return {"error": "timeout", "answer": "", "tools_used": []}
    except Exception as e:
        return {"error": str(e), "answer": "", "tools_used": []}


def run_agent_evaluation(
    dataset: list[dict] = None,
    run_label: str = "",
) -> dict:
    """
    Run the full single-agent evaluation suite.

    Returns summary dict with run_id, pass_rate, avg_score, failed_cases.
    """
    if dataset is None:
        dataset = load_agent_dataset()

    run_id = (
        f"agent_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        f"_{str(uuid.uuid4())[:6]}"
    )
    create_run(run_id, "agent", len(dataset))

    logger.info(f"Starting Agent eval run {run_id} — {len(dataset)} cases")
    print(f"\n{'='*60}")
    print(f"AGENT EVALUATION — {len(dataset)} test cases")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}\n")

    passed_count = 0
    failed_cases = []

    for i, case in enumerate(dataset):
        case_id         = case.get("id", f"agent_{i+1:03d}")
        query           = case["query"]
        expected        = case.get("expected_answer", "")
        expected_tools  = case.get("expected_tools", [])

        print(f"[{i+1}/{len(dataset)}] {case_id}: {query[:60]}...")

        start   = time.time()
        result  = _query_copilot_agent(query)
        latency = round(time.time() - start, 2)

        if "error" in result:
            save_case_result(
                run_id=run_id, case_id=case_id, eval_mode="agent",
                query=query,
                actual_response=f"ERROR: {result['error']}",
                score=0.0, passed=False,
                latency_seconds=latency,
                expected=expected,
                failure_reason=f"API error: {result['error']}",
            )
            print(f"  ✗ API ERROR: {result['error']}")
            failed_cases.append({"case_id": case_id, "reason": result["error"]})
            continue

        answer     = result.get("answer", "")
        tools_used = result.get("tools_used", [])

        judge = judge_agent_response(
            query=query,
            response=answer,
            tools_used=tools_used,
            expected_tools=expected_tools,
            expected=expected,
        )

        save_case_result(
            run_id=run_id, case_id=case_id, eval_mode="agent",
            query=query, actual_response=answer,
            score=judge.score, passed=judge.passed,
            latency_seconds=latency,
            expected=expected,
            failure_reason=judge.failure_reason,
            judge_reasoning=judge.reasoning,
        )

        status = "✓" if judge.passed else "✗"
        print(
            f"  {status} Score: {judge.score:.3f} | "
            f"Tools: {tools_used} | Latency: {latency}s"
        )

        if judge.passed:
            passed_count += 1
        else:
            failed_cases.append({
                "case_id":        case_id,
                "score":          judge.score,
                "failure_reason": judge.failure_reason,
            })

    complete_run(run_id)

    total     = len(dataset)
    pass_rate = passed_count / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"AGENT EVAL COMPLETE")
    print(f"Passed:    {passed_count}/{total} ({pass_rate:.1%})")
    print(f"Failed:    {total - passed_count}")
    if failed_cases:
        print("Failures:")
        for f in failed_cases:
            print(f"  - {f['case_id']}: {f.get('failure_reason','')[:80]}")
    print(f"{'='*60}\n")

    return {
        "run_id":       run_id,
        "eval_mode":    "agent",
        "total":        total,
        "passed":       passed_count,
        "failed":       total - passed_count,
        "pass_rate":    round(pass_rate, 3),
        "failed_cases": failed_cases,
    }


def _default_agent_cases() -> list[dict]:
    return [
        {
            "id": "agent_001",
            "query": "Pump discharge pressure is 450 psi. Rated spec is 380 psi. Is this dangerous?",
            "expected_tools": ["spec_checker"],
            "expected_answer": "WARNING spec_checker deviation percentage",
            "category": "spec_check"
        },
        {
            "id": "agent_002",
            "query": "Convert 150 psi to bar",
            "expected_tools": ["unit_converter"],
            "expected_answer": "10.34 bar",
            "category": "unit_conversion"
        },
        {
            "id": "agent_003",
            "query": "What are the lockout tagout procedures for electrical equipment?",
            "expected_tools": ["search_industrial_documentation"],
            "expected_answer": "lockout tagout energy isolation",
            "category": "retrieval"
        },
        {
            "id": "agent_004",
            "query": "Check live readings for pump-001",
            "expected_tools": ["get_equipment_telemetry"],
            "expected_answer": "pump-001 pressure flow temperature",
            "category": "telemetry"
        },
        {
            "id": "agent_005",
            "query": "What is the health status of all equipment?",
            "expected_tools": ["list_all_equipment"],
            "expected_answer": "pump-001 pump-002 motor-001 compressor-001",
            "category": "telemetry"
        },
        {
            "id": "agent_006",
            "query": "Vibration reading is 2.8 mm/s. ISO limit is 2.3 mm/s. What severity?",
            "expected_tools": ["spec_checker"],
            "expected_answer": "WARNING CAUTION vibration severity",
            "category": "spec_check"
        },
        {
            "id": "agent_007",
            "query": "Diagnose motor-001 using live telemetry",
            "expected_tools": ["get_equipment_telemetry", "search_industrial_documentation"],
            "expected_answer": "motor-001 temperature vibration diagnosis",
            "category": "diagnosis"
        },
        {
            "id": "agent_008",
            "query": "Convert 75 degrees celsius to fahrenheit",
            "expected_tools": ["unit_converter"],
            "expected_answer": "167 fahrenheit",
            "category": "unit_conversion"
        },
        {
            "id": "agent_009",
            "query": "What PPE is required for working near rotating equipment?",
            "expected_tools": ["search_industrial_documentation"],
            "expected_answer": "PPE safety equipment rotating",
            "category": "retrieval"
        },
        {
            "id": "agent_010",
            "query": "Motor current draw is 48 amps. Rated current is 42 amps. Safe to operate?",
            "expected_tools": ["spec_checker"],
            "expected_answer": "current WARNING CAUTION ampere",
            "category": "spec_check"
        },
    ]

