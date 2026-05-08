---
title: AgentEval
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# ◈ AgentEval
### LLM Evaluation & Observability Platform

Production-grade evaluation infrastructure for the [Industrial AI Copilot](https://github.com/victor-isuo/industrial-ai-copilot) — automatically scores RAG pipelines, single-agent tool use, and multi-agent orchestration using LLM-as-judge, with a live observability dashboard and CI/CD regression detection.

## 🔴 Live Demo

| Interface | URL |
|-----------|-----|
| Dashboard | https://victorisuo-agenteval.hf.space |
| System Under Evaluation | https://victorisuo-industrial-ai-copilot.hf.space |

---

## Why This Exists

The Industrial AI Copilot reports 90% evaluation accuracy — but that number was measured against a static 30-case test suite built before optimization. In production, accuracy needs continuous monitoring across all three system layers — retrieval quality, single-agent reasoning, and multi-agent orchestration — not just a one-time benchmark.

AgentEval closes that gap. Every deployment is automatically evaluated. Score regressions block the pipeline before they reach users.

---

## Architecture

```
Industrial AI Copilot (System Under Evaluation)
              ↓
┌─────────────────────────────────────────────┐
│              Evaluation Engine              │
│                                             │
│  ┌──────────┐ ┌─────────┐ ┌─────────────┐  │
│  │   RAG    │ │  Agent  │ │ Multi-Agent │  │
│  │  Eval   │ │  Eval   │ │    Eval     │  │
│  └──────────┘ └─────────┘ └─────────────┘  │
│              ↓       ↓         ↓            │
│         LLM-as-Judge Scorer                 │
│         (Groq Llama 4 Scout)                │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│           SQLite Results Store              │
│  Runs, cases, scores, costs, regressions    │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│       Observability Dashboard               │
│  Score trends, failure analysis,            │
│  regression alerts, cost tracking           │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│         GitHub Actions CI/CD                │
│  Push → tests → regression check → deploy  │
└─────────────────────────────────────────────┘
```

---

## Three Evaluation Modes

### RAG Mode
Evaluates the hybrid retrieval pipeline on three axes:

| Axis | Weight | Measures |
|------|--------|---------|
| Faithfulness | 40% | Does the answer stay grounded in retrieved context? |
| Citation Accuracy | 35% | Are sources cited correctly with page numbers? |
| Relevance | 25% | Does the answer address the query? |

### Single Agent Mode
Evaluates the LangGraph 9-tool agent:

| Axis | Weight | Measures |
|------|--------|---------|
| Tool Selection | 40% | Did the agent call the correct tools? |
| Reasoning Validity | 30% | Is the reasoning chain logical and grounded? |
| Task Completion | 30% | Did the agent fully answer the query? |

### Multi-Agent Mode
Evaluates the Phase 4 supervisor orchestration system:

| Axis | Weight | Measures |
|------|--------|---------|
| Delegation Accuracy | 35% | Did the supervisor select the right specialists? |
| Inter-Agent Consistency | 30% | Are specialist findings consistent? |
| Synthesis Quality | 35% | Does the Report Agent integrate findings well? |

**Pass threshold: 0.70 across all modes.**

---

## LLM-as-Judge

Every response is automatically scored by Groq Llama 4 Scout acting as an expert evaluator. The judge receives the query, response, and context, then returns axis scores and a one-sentence reasoning for the lowest score.

Why LLM-as-judge over traditional metrics:

Traditional metrics (BLEU, ROUGE, exact match) fail on open-ended AI responses. They can't assess whether a citation is accurate, whether a tool was the right choice, or whether a multi-agent synthesis is coherent. LLM-as-judge is the industry standard for evaluating generative AI systems and is used by Anthropic, OpenAI, and Google in their own evaluation pipelines.

---

## Regression Detection

Every eval run is compared against the previous run for the same mode. If the score drops more than 5%, a regression alert fires and the CI/CD pipeline blocks deployment.

```
Previous RAG score: 0.91
Current  RAG score: 0.76
Drop: 16.5% > 5% threshold
→ REGRESSION DETECTED — deployment blocked
```

This means broken prompts, degraded retrieval, or model changes never reach production silently.

---

## CI/CD Pipeline

```yaml
Push to main
    ↓
pytest test suite (unit + API tests)
    ↓
Regression check (5-case smoke eval)
    ↓ (only if tests + regression check pass)
Deploy to HuggingFace Spaces
```

The pipeline runs automatically on every push. No manual deployment steps.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Observability dashboard |
| `/eval/rag` | POST | Trigger RAG evaluation |
| `/eval/agent` | POST | Trigger agent evaluation |
| `/eval/multi-agent` | POST | Trigger multi-agent evaluation |
| `/eval/all` | POST | Run all three suites |
| `/results` | GET | Recent eval runs |
| `/results/{run_id}` | GET | Run detail with all cases |
| `/dashboard/summary` | GET | Aggregated dashboard data |
| `/history/{mode}` | GET | Score trend history |
| `/failures/{mode}` | GET | Recent failures with root cause |
| `/health` | GET | System health check |

---

## Local Setup

```bash
git clone https://github.com/victor-isuo/agenteval.git
cd agenteval
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
INDUSTRIAL_COPILOT_URL=https://victorisuo-industrial-ai-copilot.hf.space
```

Run:
```bash
uvicorn main:app --reload --port 8001
```

Run tests:
```bash
pytest tests/ -v
```

Run regression check:
```bash
python scripts/regression_check.py
```

---

## GitHub Actions Secrets Required

Add these in your GitHub repo Settings → Secrets → Actions:

| Secret | Value |
|--------|-------|
| `GROQ_API_KEY` | Your Groq API key |
| `LANGCHAIN_API_KEY` | Your LangSmith API key |
| `INDUSTRIAL_COPILOT_URL` | HuggingFace Space URL |
| `HF_TOKEN` | HuggingFace write token |

---

## Project Structure

```
agenteval/
├── src/
│   ├── evaluators/
│   │   ├── rag_evaluator.py          # RAG eval — 10 cases
│   │   ├── agent_evaluator.py        # Agent eval — 10 cases
│   │   └── multi_agent_evaluator.py  # Multi-agent eval — 5 cases
│   ├── judges/
│   │   └── llm_judge.py              # LLM-as-judge scoring engine
│   ├── store/
│       └── results_store.py          # SQLite persistence layer
│ 
│                   
├── tests/
│   └── test_agenteval.py             # Full pytest suite
├── scripts/
│   └── regression_check.py           # CI regression check script
├── static/
│   └── dashboard.html                # Observability dashboard
├── datasets/
│   ├── rag_eval_cases.json           # Customisable test cases
│   ├── agent_eval_cases.json
│   └── multi_agent_eval_cases.json
├── .github/
│   └── workflows/
│       └── eval_ci.yml               # GitHub Actions pipeline
├── main.py
├── Dockerfile
└── requirements.txt
```

---

## Relationship to Industrial AI Copilot

AgentEval is not a standalone product — it is the evaluation infrastructure for the Industrial AI Copilot. The two projects together demonstrate a complete production AI engineering story:

**Industrial AI Copilot** — builds the system.
**AgentEval** — proves the system works and keeps it working.

This is the gap between demo-quality and production-quality AI engineering.

---

## Author

**Victor Isuo** — Applied LLM Engineer

[GitHub](https://github.com/victor-isuo) · [LinkedIn](https://linkedin.com/in/victor-isuo-a02b65171) · [Industrial AI Copilot](https://victorisuo-industrial-ai-copilot.hf.space/ui)




