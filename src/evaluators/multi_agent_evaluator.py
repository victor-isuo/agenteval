"""
Multi-Agent Evaluator — AgentEval
====================================
Evaluates the Industrial AI Copilot's Phase 4 multi-agent
orchestration system across three quality dimensions:

- Delegation Accuracy: Did the supervisor select the right specialists?
- Inter-Agent Consistency: Are specialist findings consistent with each other?
- Synthesis Quality: Does the Report Agent integrate findings well?

This is the most complex evaluation mode — it assesses not just
the final response but the entire orchestration chain.
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

from src.judges.llm_judge import judge_multi_agent_response
from src.store.results_store import (
    create_run, save_case_result, complete_run
)

logger = logging.getLogger(__name__)
COPILOT_URL = os.getenv(
    "INDUSTRIAL_COPILOT_URL",
    "https://victorisuo-industrial-ai-copilot.hf.space"
)
DATASET_PATH = (
    Path(__file__).parent.parent.parent / "datasets" / "multi_agent_eval_cases.json"
)


def load_multi_agent_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        logger.warning("Multi-agent dataset not found. Using defaults.")
        return _default_multi_agent_cases()
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


def _query_copilot_multiagent(question: str, timeout: int = 90) -> dict:
    """Query the live /multiagent endpoint."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{COPILOT_URL}/multiagent",
                json={"question": question},
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        return {
            "error": "timeout",
            "final_answer": "",
            "agent_results": [],
            "agents_used": [],
        }
    except Exception as e:
        return {
            "error": str(e),
            "final_answer": "",
            "agent_results": [],
            "agents_used": [],
        }


def run_multi_agent_evaluation(
    dataset: list[dict] = None,
    run_label: str = "",
) -> dict:
    """
    Run the full multi-agent evaluation suite.

    Returns summary dict with run_id, pass_rate, avg_score, failed_cases.
    """
    if dataset is None:
        dataset = load_multi_agent_dataset()

    run_id = (
        f"multi_agent_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        f"_{str(uuid.uuid4())[:6]}"
    )
    create_run(run_id, "multi_agent", len(dataset))

    logger.info(f"Starting Multi-Agent eval run {run_id} — {len(dataset)} cases")
    print(f"\n{'='*60}")
    print(f"MULTI-AGENT EVALUATION — {len(dataset)} test cases")
    print(f"Run ID: {run_id}")
    print(f"Note: Multi-agent queries take 25-35s each")
    print(f"{'='*60}\n")

    passed_count = 0
    failed_cases = []

    for i, case in enumerate(dataset):
        case_id = case.get("id", f"multi_{i+1:03d}")
        query = case["query"]
        expected = case.get("expected_answer", "")
        expected_agents = case.get("expected_agents", [])

        print(f"[{i+1}/{len(dataset)}] {case_id}: {query[:60]}...")

        start = time.time()
        result = _query_copilot_multiagent(query)
        latency = round(time.time() - start, 2)

        if "error" in result:
            save_case_result(
                run_id=run_id, case_id=case_id, eval_mode="multi_agent",
                query=query,
                actual_response=f"ERROR: {result['error']}",
                score=0.0, passed=False,
                latency_seconds=latency,
                expected=expected,
                failure_reason=f"API error: {result['error']}",
            )
            print(f" ✗ API ERROR: {result['error']}")
            failed_cases.append({"case_id": case_id, "reason": result["error"]})
            continue

        final_answer = result.get("final_answer", "")
        agents_used = result.get("agents_used", [])
        agent_results = result.get("agent_results", [])

        judge = judge_multi_agent_response(
            query=query,
            final_response=final_answer,
            agents_used=agents_used,
            agent_results=agent_results,
            expected_agents=expected_agents,
        )

        save_case_result(
            run_id=run_id, case_id=case_id, eval_mode="multi_agent",
            query=query, actual_response=final_answer,
            score=judge.score, passed=judge.passed,
            latency_seconds=latency,
            expected=expected,
            failure_reason=judge.failure_reason,
            judge_reasoning=judge.reasoning,
        )

        status = "✓" if judge.passed else "✗"
        print(
            f" {status} Score: {judge.score:.3f} | "
            f"Agents: {agents_used} | Latency: {latency}s"
        )

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
    print(f"MULTI-AGENT EVAL COMPLETE")
    print(f"Passed: {passed_count}/{total} ({pass_rate:.1%})")
    print(f"Failed: {total - passed_count}")
    if failed_cases:
        print("Failures:")
        for f in failed_cases:
            print(f" - {f['case_id']}: {f.get('failure_reason','')[:80]}")
    print(f"{'='*60}\n")

    return {
        "run_id": run_id,
        "eval_mode": "multi_agent",
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": round(pass_rate, 3),
        "failed_cases": failed_cases,
    }


def _default_multi_agent_cases() -> list[dict]:
    return [
        {
            "id": "multi_001",
            "query": "Generate a full plant health report for all equipment",
            "expected_agents": [
                "Telemetry Agent", "Analysis Agent",
                "Safety Agent", "Retrieval Agent"
            ],
            "expected_answer": "pump motor compressor health status alerts",
            "category": "full_audit"
        },
        {
            "id": "multi_002",
            "query": "Is it safe to continue operating the plant right now?",
            "expected_agents": [
                "Telemetry Agent", "Safety Agent", "Analysis Agent"
            ],
            "expected_answer": "safe operate risk assessment equipment status",
            "category": "safety_assessment"
        },
        {
            "id": "multi_003",
            "query": "Diagnose pump-001 fault and recommend maintenance action",
            "expected_agents": [
                "Telemetry Agent", "Analysis Agent", "Retrieval Agent"
            ],
            "expected_answer": "pump-001 fault diagnosis maintenance recommendation",
            "category": "fault_diagnosis"
        },
        {
            "id": "multi_004",
            "query": "Audit all equipment against ISO safety standards",
            "expected_agents": [
                "Telemetry Agent", "Safety Agent", "Retrieval Agent"
            ],
            "expected_answer": "ISO standard compliance audit equipment",
            "category": "compliance"
        },
        {
            "id": "multi_005",
            "query": "What maintenance is overdue across all plant assets?",
            "expected_agents": [
                "Telemetry Agent", "Retrieval Agent", "Analysis Agent"
            ],
            "expected_answer": "maintenance overdue schedule inspection",
            "category": "maintenance_planning"
        },
    ]