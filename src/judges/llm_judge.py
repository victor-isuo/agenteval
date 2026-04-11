"""
LLM Judge — AgentEval
======================
The core scoring engine. Uses Groq Llama 4 to automatically
evaluate the quality of responses from the Industrial AI Copilot.

Why LLM-as-judge:
Traditional eval metrics (exact match, BLEU, ROUGE) fail on
open-ended AI responses. LLM-as-judge is the industry standard
for evaluating generative AI outputs — used by Anthropic, OpenAI,
and most serious AI labs.

The judge scores on three axes per eval mode:
- RAG: faithfulness, citation accuracy, relevance
- Agent: tool selection, reasoning validity, task completion
- Multi-Agent: delegation accuracy, consistency, synthesis quality

Each axis is scored 0.0-1.0. Final score is weighted average.
Pass threshold: 0.70 (same as Industrial AI Copilot eval suite)
"""

import os
import json
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.70


@dataclass
class JudgeResult:
    score: float
    passed: bool
    reasoning: str
    failure_reason: str
    axis_scores: dict


def _call_judge(prompt: str) -> str:
    """Call Groq LLM for judge scoring. Returns raw response text."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800,
    )
    return response.choices[0].message.content


def _parse_scores(raw: str, axes: list[str]) -> dict:
    """
    Parse axis scores from judge response.
    Expects JSON like: {"faithfulness": 0.9, "citation_accuracy": 0.8}
    Falls back to 0.5 for any missing axis.
    """
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in judge response")
        scores = json.loads(raw[start:end])
        return {axis: float(scores.get(axis, 0.5)) for axis in axes}
    except Exception as e:
        logger.warning(f"Score parse failed: {e}. Raw: {raw[:200]}")
        return {axis: 0.5 for axis in axes}


# ── RAG Judge ─────────────────────────────────────────────────────────────────

RAG_AXES = ["faithfulness", "citation_accuracy", "relevance"]
RAG_WEIGHTS = {"faithfulness": 0.40, "citation_accuracy": 0.35, "relevance": 0.25}

RAG_JUDGE_PROMPT = """You are an expert evaluator assessing a RAG system response.

QUERY: {query}

RETRIEVED CONTEXT (what the system had access to):
{context}

SYSTEM RESPONSE:
{response}

EXPECTED ANSWER (if available):
{expected}

Score the response on these three axes. Return ONLY a JSON object with scores from 0.0 to 1.0:

faithfulness (0.0-1.0):
  Does the response stay grounded in the retrieved context?
  Does it avoid adding information not present in the context?
  1.0 = fully grounded, 0.0 = completely fabricated

citation_accuracy (0.0-1.0):
  Does the response cite specific documents and page numbers?
  Are the citations actually present in the retrieved context?
  1.0 = accurate citations on every claim, 0.0 = no citations or fabricated citations

relevance (0.0-1.0):
  Does the response directly answer the query?
  Is the information provided useful for the specific question asked?
  1.0 = perfectly relevant complete answer, 0.0 = completely off-topic

After the JSON, write one sentence explaining the lowest score.

Return format:
{{"faithfulness": 0.0, "citation_accuracy": 0.0, "relevance": 0.0}}
Reasoning: [one sentence]"""


def judge_rag_response(
    query: str,
    response: str,
    context: str = "",
    expected: str = "",
) -> JudgeResult:
    """
    Score a RAG system response.

    Args:
        query: The user's question
        response: What the RAG system returned
        context: The retrieved chunks it had access to
        expected: Expected answer if available (optional)
    """
    prompt = RAG_JUDGE_PROMPT.format(
        query=query, context=context or "Not provided",
        response=response, expected=expected or "Not provided"
    )

    raw = _call_judge(prompt)
    axis_scores = _parse_scores(raw, RAG_AXES)
    final_score = sum(axis_scores[ax] * RAG_WEIGHTS[ax] for ax in RAG_AXES)
    passed = final_score >= PASS_THRESHOLD

    reasoning = ""
    if "Reasoning:" in raw:
        reasoning = raw.split("Reasoning:")[-1].strip()

    failure_reason = ""
    if not passed:
        worst_axis = min(axis_scores, key=axis_scores.get)
        failure_reason = (
            f"Failed on {worst_axis} "
            f"(score: {axis_scores[worst_axis]:.2f}). {reasoning}"
        )

    return JudgeResult(
        score=round(final_score, 3),
        passed=passed,
        reasoning=reasoning,
        failure_reason=failure_reason,
        axis_scores=axis_scores,
    )


# ── Agent Judge ───────────────────────────────────────────────────────────────

AGENT_AXES = ["tool_selection", "reasoning_validity", "task_completion"]
AGENT_WEIGHTS = {"tool_selection": 0.40, "reasoning_validity": 0.30, "task_completion": 0.30}

AGENT_JUDGE_PROMPT = """You are an expert evaluator assessing an AI agent response.

QUERY: {query}

TOOLS AVAILABLE: {available_tools}

TOOLS ACTUALLY USED: {tools_used}

AGENT RESPONSE:
{response}

EXPECTED TOOLS (if known): {expected_tools}
EXPECTED OUTCOME: {expected}

Score the agent on these three axes. Return ONLY a JSON object:

tool_selection (0.0-1.0):
  Did the agent call the correct tools for this query?
  Did it avoid unnecessary tool calls?
  Did it call tools in the right sequence?
  1.0 = perfect tool selection, 0.0 = wrong tools or no tools when needed

reasoning_validity (0.0-1.0):
  Is the reasoning chain logical and coherent?
  Does the agent correctly interpret tool outputs?
  Does it avoid hallucinating information not in tool results?
  1.0 = fully valid reasoning, 0.0 = illogical or fabricated reasoning

task_completion (0.0-1.0):
  Did the agent fully answer the query?
  Is the response actionable and specific?
  Does it include severity classifications where appropriate?
  1.0 = complete answer with all required elements, 0.0 = failed to answer

Return format:
{{"tool_selection": 0.0, "reasoning_validity": 0.0, "task_completion": 0.0}}
Reasoning: [one sentence on the lowest score]"""


def judge_agent_response(
    query: str,
    response: str,
    tools_used: list[str],
    available_tools: list[str] = None,
    expected_tools: list[str] = None,
    expected: str = "",
) -> JudgeResult:
    """Score a single LangGraph agent response."""
    prompt = AGENT_JUDGE_PROMPT.format(
        query=query,
        available_tools=", ".join(available_tools or [
            "search_industrial_documentation", "spec_checker",
            "unit_converter", "engineering_calculator",
            "get_equipment_telemetry", "list_all_equipment",
            "analyze_equipment_image", "analyze_gauge_reading",
            "query_mcp_industrial_server"
        ]),
        tools_used=", ".join(tools_used) if tools_used else "None",
        expected_tools=", ".join(expected_tools) if expected_tools else "Not specified",
        response=response,
        expected=expected or "Not specified",
    )

    raw = _call_judge(prompt)
    axis_scores = _parse_scores(raw, AGENT_AXES)
    final_score = sum(axis_scores[ax] * AGENT_WEIGHTS[ax] for ax in AGENT_AXES)
    passed = final_score >= PASS_THRESHOLD

    reasoning = ""
    if "Reasoning:" in raw:
        reasoning = raw.split("Reasoning:")[-1].strip()

    failure_reason = ""
    if not passed:
        worst_axis = min(axis_scores, key=axis_scores.get)
        failure_reason = (
            f"Failed on {worst_axis} "
            f"(score: {axis_scores[worst_axis]:.2f}). {reasoning}"
        )

    return JudgeResult(
        score=round(final_score, 3),
        passed=passed,
        reasoning=reasoning,
        failure_reason=failure_reason,
        axis_scores=axis_scores,
    )


# ── Multi-Agent Judge ─────────────────────────────────────────────────────────

MULTI_AXES = ["delegation_accuracy", "inter_agent_consistency", "synthesis_quality"]
MULTI_WEIGHTS = {
    "delegation_accuracy": 0.35,
    "inter_agent_consistency": 0.30,
    "synthesis_quality": 0.35,
}

MULTI_JUDGE_PROMPT = """You are an expert evaluator assessing a multi-agent AI system.

QUERY: {query}

SUPERVISOR SELECTED AGENTS: {agents_used}

SPECIALIST FINDINGS:
{specialist_findings}

FINAL SYNTHESISED RESPONSE:
{final_response}

EXPECTED AGENTS TO BE USED: {expected_agents}

Score on three axes. Return ONLY a JSON object:

delegation_accuracy (0.0-1.0):
  Did the supervisor select the right specialists for this query?
  Were unnecessary agents avoided?
  1.0 = perfect delegation, 0.0 = wrong agents selected

inter_agent_consistency (0.0-1.0):
  Are the specialist findings consistent with each other?
  Does the final response contradict any specialist finding?
  Does the report agent faithfully represent all specialist outputs?
  1.0 = fully consistent, 0.0 = contradictions or missing findings

synthesis_quality (0.0-1.0):
  Does the final response coherently integrate all findings?
  Is the response actionable and well-structured?
  Does it include severity levels and cited recommendations?
  1.0 = excellent synthesis, 0.0 = poor or incomplete synthesis

Return format:
{{"delegation_accuracy": 0.0, "inter_agent_consistency": 0.0, "synthesis_quality": 0.0}}
Reasoning: [one sentence on the lowest score]"""


def judge_multi_agent_response(
    query: str,
    final_response: str,
    agents_used: list[str],
    agent_results: list[dict],
    expected_agents: list[str] = None,
) -> JudgeResult:
    """Score a multi-agent orchestration response."""
    findings = ""
    for r in agent_results:
        findings += f"\n{r.get('agent_name', 'Agent').upper()}:\n{r.get('response', '')[:300]}\n"

    prompt = MULTI_JUDGE_PROMPT.format(
        query=query,
        agents_used=", ".join(agents_used),
        specialist_findings=findings,
        final_response=final_response,
        expected_agents=", ".join(expected_agents) if expected_agents else "Not specified",
    )

    raw = _call_judge(prompt)
    axis_scores = _parse_scores(raw, MULTI_AXES)
    final_score = sum(axis_scores[ax] * MULTI_WEIGHTS[ax] for ax in MULTI_AXES)
    passed = final_score >= PASS_THRESHOLD

    reasoning = ""
    if "Reasoning:" in raw:
        reasoning = raw.split("Reasoning:")[-1].strip()

    failure_reason = ""
    if not passed:
        worst_axis = min(axis_scores, key=axis_scores.get)
        failure_reason = (
            f"Failed on {worst_axis} "
            f"(score: {axis_scores[worst_axis]:.2f}). {reasoning}"
        )

    return JudgeResult(
        score=round(final_score, 3),
        passed=passed,
        reasoning=reasoning,
        failure_reason=failure_reason,
        axis_scores=axis_scores,
    )
