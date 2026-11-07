# Voyage & Port-Call Intelligence — Agentic RAG

An **Ollama-first**, fully local **agentic** RAG service over voyage reports,
port circulars and incident logs. A **Bahri-style fleet-operations use case**:
ops teams ask multi-part operational questions ("why was the call delayed and
what was the cost?") that a single retrieve-then-answer pass handles poorly.

> Sample records are illustrative and fictional — not real operational data.

## Agentic loop

Instead of one retrieval pass, the agent reasons in steps and shows its work:

```
question
  │  1. PLAN         decompose into focused sub-queries
  ├─ 2. RETRIEVE     gather + pool unique evidence per sub-query
  ├─ 3. SYNTHESIZE   draft a grounded answer from the pooled evidence
  └─ 4. GROUND-CHECK score support; if weak, retrieve once more with the
                     draft's own terms and re-synthesize
                           │
                           ▼
        answer + citations + full reasoning trace
```

- **Planner** (`core/agent.py::plan_subqueries`) — deterministic decomposition
  on conjunctions/clauses; the full question stays as sub-query #1.
- **Grounding check** (`grounding_score`) — fraction of the answer's content
  words supported by the retrieved evidence; a low score triggers one
  refinement retrieval.
- The `/ask` response includes a `trace` (sub-queries, per-step retrievals,
  grounding score, whether it refined) so the reasoning is auditable.

Planning and grounding are **deterministic** (fully testable offline); the LLM
is used only for the final answer wording. Ollama-first, with a deterministic
offline fallback so the pipeline and tests run with no model server.

## Quickstart

```bash
pip install -r requirements.txt

ollama pull llama3.2 && ollama pull nomic-embed-text   # optional
python demo.py                                         # or RAG_OFFLINE=1 python demo.py

uvicorn voyageiq.api.main:app --reload                 # API at /docs
```

### Example

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Why was the Jeddah call delayed and what was the demurrage impact?"}'
```

```json
{
  "answer": "Berthing at Jeddah was delayed by six hours due to port congestion ... demurrage of approximately USD 150,000 ...",
  "sources": [{ "source": "voyage_records.md", "excerpt": "..." }],
  "trace": {
    "subqueries": ["Why was the Jeddah call delayed ...", "what was the demurrage impact"],
    "grounding": 0.71,
    "refined": false
  }
}
```

## Configuration (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `RAG_LLM_MODEL` / `RAG_EMBED_MODEL` | `llama3.2` / `nomic-embed-text` | models |
| `RAG_OFFLINE` | `0` | `1` forces the offline backend |
| `RAG_TOP_K` | `3` | evidence chunks per retrieval |
| `RAG_MAX_SUBQUERIES` | `3` | sub-queries the planner may emit |
| `RAG_MAX_STEPS` | `4` | upper bound on agent steps |

## Tests

```bash
pytest -q          # offline; no Ollama needed
```

CI lints, runs tests + the demo offline, and builds the Docker image.
