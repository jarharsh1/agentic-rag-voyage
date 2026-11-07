# Agentic RAG Engine

An **Ollama-first**, fully local, **domain-agnostic** *agentic* Retrieval-Augmented
Generation engine. Instead of a single retrieve-then-generate pass, every
question runs a short reasoning loop over *any* text corpus — and every answer
ships with its citations and the full reasoning trace.

> Ships with a tiny neutral sample knowledge base so it runs out of the box.
> Replace `data/sample_knowledge_base.md` with your own corpus; nothing else
> changes.

## The loop (the core idea)

A single pass misses multi-part questions and can drift from the evidence. The
agent instead:

```
question ─► 1. PLAN         decompose into focused sub-queries
          ─► 2. RETRIEVE    gather + pool evidence for each sub-query
          ─► 3. SYNTHESIZE  draft a grounded answer from the pooled evidence
          ─► 4. GROUND-CHECK score support; if weak, retrieve again with the
                            draft's own terms and re-synthesize
          ─► answer + citations + trace
```

Planning and the grounding check are **deterministic** functions, so the whole
agent is testable offline; the model is used only for the final synthesis
wording. The `/ask` response exposes the `trace` (sub-queries, per-step
retrievals, grounding score, whether it refined).

## Use cases

The engine is the same; only the corpus changes. It shines on **multi-part**
questions, which the planner splits and answers from pooled evidence.

| Domain | Corpus you ingest | Example multi-part question |
|--------|-------------------|-----------------------------|
| **Healthcare** | clinical guidelines, care protocols | "What is first-line for type 2 diabetes and when do you escalate?" |
| **Legal** | contracts, policies, case notes | "What is the notice period and what are the penalties for breach?" |
| **Customer support** | help center, FAQs, SLAs | "How long do I have to request a refund and how is it issued?" |
| **Research** | papers, technical docs | "What method did the authors use and what were the limitations?" |
| **Compliance** | regulations, filings, runbooks | "What are the SLA and data-retention rules?" |

## Stack

- **Backend:** FastAPI (an async Flask-equivalent; auto docs at `/docs`).
- **Frontend:** a browser chat UI (`web/`) served at `/`, showing each answer's
  reasoning trace (sub-query count, grounding score) and sources. **This is
  where the user interacts.**
- **Model:** Ollama (local), with a deterministic **offline fallback** so the
  pipeline and tests run with no model server — CI-safe.
- **Deploy:** single Docker image or `docker compose` (API + Ollama). See
  [DEPLOYMENT.md](DEPLOYMENT.md).

## Components

- **Agent** (`core/agent.py`) — the plan → retrieve → synthesize → ground-check
  loop, with deterministic `plan_subqueries` and `grounding_score`.
- **Pipeline** (`core/pipeline.py`) — ingest (chunk + embed) then answer.
- **Vector store** (`core/vector_store.py`) — in-memory embeddings + search.
- **Backends** (`core/backends.py`) — Ollama or the deterministic offline pair.
- **API** (`api/main.py`) — `/`, `/health`, `/ingest`, `/ask`.

## Quickstart

```bash
pip install -r requirements.txt
ollama pull llama3.2 && ollama pull nomic-embed-text   # optional
python demo.py                                         # or RAG_OFFLINE=1 python demo.py
uvicorn voyageiq.api.main:app --reload                 # UI at /, API at /docs
```

### Example

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "What are the SLA and data-retention rules?"}'
```

```json
{
  "answer": "Standard support tickets receive a first response within 24 hours... Personal data is retained for 24 months...",
  "sources": [
    { "source": "sample_knowledge_base.md", "excerpt": "..." }
  ],
  "trace": {
    "subqueries": ["What are the SLA and data-retention rules?", "the SLA", "data-retention rules"],
    "grounding": 0.82,
    "refined": false
  }
}
```

## Configuration (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `RAG_LLM_MODEL` / `RAG_EMBED_MODEL` | `llama3.2` / `nomic-embed-text` | models |
| `RAG_OFFLINE` | `0` | `1` forces the deterministic offline backend |
| `RAG_TOP_K` | `3` | evidence pooled per sub-query |
| `RAG_MAX_SUBQUERIES` / `RAG_MAX_STEPS` | `3` / `4` | agentic loop limits |

## Tests

```bash
pytest -q          # offline; no Ollama needed
```

CI lints, runs tests + the demo offline, and builds the Docker image.
