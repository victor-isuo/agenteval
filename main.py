"""
AgentEval — LLM Evaluation & Observability Platform
=====================================================
FastAPI application exposing eval endpoints and the dashboard.

Endpoints:
  GET / Dashboard UI
  POST /eval/rag Run RAG evaluation suite
  POST /eval/agent Run single-agent evaluation suite
  POST /eval/multi-agent Run multi-agent evaluation suite
  POST /eval/all Run all three suites sequentially
  GET /results Recent eval runs
  GET /results/{run_id} Specific run with all case results
  GET /dashboard/summary Aggregated dashboard data
  GET /history/{mode} Score trend history for a mode
  GET /failures/{mode} Recent failures for a mode
  GET /health System health check
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.store.results_store import (
    initialize_db, get_recent_runs, get_run_cases,
    get_dashboard_summary, get_score_history, get_failure_analysis
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising AgentEval database...")
    initialize_db()
    logger.info("AgentEval ready.")
    yield
    logger.info("Shutting down AgentEval.")


app = FastAPI(
    title="AgentEval",
    description=(
        "LLM Evaluation & Observability Platform for the "
        "Industrial AI Copilot — evaluates RAG, single-agent, "
        "and multi-agent systems with automated LLM-as-judge scoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request Models ────────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    run_label: str = ""


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/")
async def dashboard():
    return FileResponse("static/dashboard.html")


# ── Eval Endpoints ────────────────────────────────────────────────────────────

@app.post("/eval/rag")
async def run_rag_eval(request: EvalRequest, background_tasks: BackgroundTasks):
    """
    Trigger a RAG evaluation run.
    Runs in the background — returns run_id immediately.
    Poll /results/{run_id} to check status.
    """
    from src.evaluators.rag_evaluator import run_rag_evaluation
    import uuid

    run_id = f"rag_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(
        run_rag_evaluation, run_label=request.run_label
    )
    return {
        "message": "RAG evaluation started",
        "run_label": request.run_label,
        "note": "Poll /results to check progress",
    }


@app.post("/eval/agent")
async def run_agent_eval(request: EvalRequest, background_tasks: BackgroundTasks):
    """Trigger a single-agent evaluation run."""
    from src.evaluators.agent_evaluator import run_agent_evaluation

    background_tasks.add_task(
        run_agent_evaluation, run_label=request.run_label
    )
    return {
        "message": "Agent evaluation started",
        "run_label": request.run_label,
        "note": "Poll /results to check progress",
    }


@app.post("/eval/multi-agent")
async def run_multi_agent_eval(request: EvalRequest, background_tasks: BackgroundTasks):
    """Trigger a multi-agent evaluation run. Note: takes 3-5 minutes."""
    from src.evaluators.multi_agent_evaluator import run_multi_agent_evaluation

    background_tasks.add_task(
        run_multi_agent_evaluation, run_label=request.run_label
    )
    return {
        "message": "Multi-agent evaluation started",
        "run_label": request.run_label,
        "note": "Multi-agent eval takes 3-5 minutes. Poll /results to check progress.",
    }


@app.post("/eval/all")
async def run_all_evals(request: EvalRequest, background_tasks: BackgroundTasks):
    """
    Run all three evaluation suites sequentially.
    RAG → Agent → Multi-Agent.
    Full run takes approximately 10-15 minutes.
    """
    from src.evaluators.rag_evaluator import run_rag_evaluation
    from src.evaluators.agent_evaluator import run_agent_evaluation
    from src.evaluators.multi_agent_evaluator import run_multi_agent_evaluation

    async def run_all():
        logger.info("Running full evaluation suite...")
        run_rag_evaluation(run_label=request.run_label)
        run_agent_evaluation(run_label=request.run_label)
        run_multi_agent_evaluation(run_label=request.run_label)
        logger.info("Full evaluation suite complete.")

    background_tasks.add_task(run_all)
    return {
        "message": "Full evaluation suite started (RAG → Agent → Multi-Agent)",
        "note": "Takes 10-15 minutes. Poll /results to check progress.",
    }


# ── Results Endpoints ─────────────────────────────────────────────────────────

@app.get("/results")
async def get_results(limit: int = 20):
    """Get the most recent evaluation runs."""
    runs = get_recent_runs(limit=limit)
    return {"runs": runs, "total": len(runs)}


@app.get("/results/{run_id}")
async def get_run_detail(run_id: str):
    """Get full details for a specific eval run including all case results."""
    cases = get_run_cases(run_id)
    if not cases:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found or has no cases yet."
        )
    runs = get_recent_runs(limit=100)
    run = next((r for r in runs if r["run_id"] == run_id), None)
    return {
        "run": run,
        "cases": cases,
        "total": len(cases),
    }


@app.get("/dashboard/summary")
async def dashboard_summary():
    """
    Aggregated summary for the dashboard.
    Returns latest scores and regression alerts for all three modes.
    """
    return get_dashboard_summary()


@app.get("/history/{eval_mode}")
async def score_history(eval_mode: str, days: int = 30):
    """
    Score trend history for a specific eval mode.
    eval_mode: 'rag', 'agent', or 'multi_agent'
    """
    if eval_mode not in ("rag", "agent", "multi_agent"):
        raise HTTPException(
            status_code=400,
            detail="eval_mode must be 'rag', 'agent', or 'multi_agent'"
        )
    history = get_score_history(eval_mode=eval_mode, days=days)
    return {"eval_mode": eval_mode, "history": history}


@app.get("/failures/{eval_mode}")
async def failure_analysis(eval_mode: str, limit: int = 10):
    """Recent failures for a given eval mode with root cause reasoning."""
    if eval_mode not in ("rag", "agent", "multi_agent"):
        raise HTTPException(
            status_code=400,
            detail="eval_mode must be 'rag', 'agent', or 'multi_agent'"
        )
    failures = get_failure_analysis(eval_mode=eval_mode, limit=limit)
    return {
        "eval_mode": eval_mode,
        "failures": failures,
        "total": len(failures),
    }


@app.get("/health")
async def health():
    """System health check."""
    copilot_url = os.getenv(
        "INDUSTRIAL_COPILOT_URL",
        "https://victorisuo-industrial-ai-copilot.hf.space"
    )
    return {
        "status": "healthy",
        "version": "1.0.0",
        "database": "connected",
        "copilot_url": copilot_url,
        "eval_modes": ["rag", "agent", "multi_agent"],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

