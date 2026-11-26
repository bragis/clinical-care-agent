# clinical-care-agent

A small **clinical-AI agent** for chronic-disease
care. Uses **RAG**, **LLM tool-calling**, **multi-step orchestration
with a human-in-the-loop**, **structured outputs**, and an **offline evaluation
harness wired into CI**.

It runs **out of the box with no API key and no network** (a deterministic mock
"agent" drives the same loop a real LLM would), so you can read the trace, run
the eval gate, and *then* swap in a real model.

> ⚠️ **Not for clinical use.** All patient data is synthetic. All "protocols"
> are simplified, illustrative, and not medical guidance. The agent produces a
> **draft for a human clinician**, never an autonomous decision. This is an
> engineering/learning artifact about agent *plumbing*, not about medicine.

---

## What maps to what

|---|---|
| RAG: embeddings/retrieval, grounding | `src/retrieval.py` (TF-IDF default + optional embeddings) |
| LLM APIs: tool use / function calling | `src/llm.py`, `src/tools.py` |
| Structured outputs | `src/schemas.py` (pydantic → tool JSON schema) |
| Orchestration: planning, tool use, error recovery | `src/agent.py` |
| Human-in-the-loop escalation | `escalate_to_clinician` tool + `escalate` flag |
| Eval: offline benchmarks, safety tests, regression, LLM-as-judge | `eval/` |
| Eval wired into CI/CD | `.github/workflows/ci.yml` |

---

## Quickstart (offline, ~2 minutes)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Watch the agent reason over one patient (mock client, no key needed)
python run_agent.py --patient P004      # hypertensive-crisis case → escalates
python run_agent.py --patient P003      # stable diabetic → no escalation

# 2) Run the unit tests
pytest -q

# 3) Run the evaluation gate (this is what CI runs)
python eval/run_eval.py --client mock
```

Patients: `P001` (HTN+T2DM, above target), `P002` (HFrEF, decompensating),
`P003` (stable T2DM), `P004` (HTN, crisis-range BP), `P005` (Chronic Kidney Disease).

---

## How it works

### The agent loop (`src/agent.py`)

```
  user: "review patient P004"
        │
        ▼
   ┌─────────────┐   tool_use     ┌──────────────────┐
   │  LLM client │ ─────────────▶ │  ToolRegistry     │
   │ (mock/real) │ ◀───────────── │  executes tool    │
   └─────────────┘  tool_result   └──────────────────┘
        │  (loop: plan → call tool → observe → repeat)
        ▼
   submit_recommendation  ──▶  validated CareRecommendation  (terminal)
```

Tools available to the model: `get_patient`, `get_vitals`, `search_protocols`
(RAG), `escalate_to_clinician` (human-in-the-loop), `submit_recommendation`
(terminal, structured). The loop stops when the model submits, or when it hits
the step budget. Tool errors are returned to the model as `tool_result` error
payloads so it can recover instead of crashing.

### The two swappable seams

Everything is built so you change *one line* to go from toy to real:

- **LLM client** — `MockLLMClient` (scripted, offline) ↔ `AnthropicClient` (real
  Messages API tool-use). Same `create(system, messages, tools) -> AssistantTurn`
  interface; the agent loop never knows which it's talking to.
- **Retriever** — `TfidfRetriever` (pure numpy, no download) ↔ `EmbeddingRetriever`
  (sentence-transformers). Same `search(query, k) -> [Chunk]` interface.

### Evaluation (`eval/`)

`cases.yaml` declares cases tagged `safety` / `regression` / `behavioural`. Each
case runs the agent and applies **deterministic checks** on the *structured*
output (did it escalate when it should? are there citations? did it avoid a
forbidden action like dosing insulin itself?). `run_eval.py` exits non-zero if a
**safety or regression** case fails — that's the CI gate. An optional
**LLM-as-judge** (`eval/judge.py`) scores free-text quality against a rubric and
only runs when you pass a real client.

---

## Layout

```
clinical-care-agent/
├── data/
│   ├── patients.json            # synthetic patients
│   └── protocols/*.md           # RAG knowledge base
├── src/
│   ├── llm.py                   # client abstraction: Anthropic + Mock
│   ├── retrieval.py             # TF-IDF + optional embedding retriever
│   ├── tools.py                 # tool schemas + implementations
│   ├── schemas.py               # structured-output models
│   └── agent.py                 # the orchestration loop
├── eval/
│   ├── cases.yaml               # benchmark / safety / regression cases
│   ├── judge.py                 # deterministic checks + LLM-as-judge
│   └── run_eval.py              # harness with CI exit code
├── tests/test_smoke.py
├── run_agent.py                 # CLI
└── .github/workflows/ci.yml
```
