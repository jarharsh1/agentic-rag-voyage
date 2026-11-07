# Deployment guide

An **Ollama-first agentic RAG** engine (plan → retrieve → synthesize →
ground-check) over any text corpus, with a bundled browser UI. Runs as local
Python, a single Docker container, or the full `docker compose` stack (API +
Ollama). Falls back to a deterministic **offline backend** when no model server
is present.

## Architecture

```
  browser ──► FastAPI (voyageiq.api.main)
  (web/ UI)    GET  /         static chat UI
               GET  /health   backend + index size
               POST /ingest   add any text document
               POST /ask      plan → retrieve → synthesize → ground-check
                    │                (returns answer + sources + trace)
                    ▼
             VoyagePipeline → VoyageAgent + VectorStore + LLM
                    │ (Ollama, or deterministic offline)
                    ▼
              Ollama model server
```

The bundled neutral sample (`data/sample_knowledge_base.md`) is pre-ingested at
startup; add more via `POST /ingest`.

## 1. Local (Python)

```bash
pip install -r requirements.txt
RAG_OFFLINE=1 uvicorn voyageiq.api.main:app --reload       # instant, no Ollama
# or with Ollama:
ollama pull llama3.2 && ollama pull nomic-embed-text
uvicorn voyageiq.api.main:app --reload
```

Open <http://localhost:8000> (UI) or <http://localhost:8000/docs> (API).

## 2. Single Docker container

```bash
docker build -t agentic-rag-voyage .
docker run -p 8000:8000 -e RAG_OFFLINE=1 agentic-rag-voyage
docker run -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 agentic-rag-voyage
```

## 3. Full stack (docker compose)

```bash
cp .env.example .env
docker compose up --build
```

## 4. Cloud notes

- Stateless container; the index is per-instance (re-ingested at startup from
  the bundled sample). Front with a shared store for multi-instance persistence.
- Set `OLLAMA_HOST` to a managed/sidecar Ollama in production.
- `GET /health` is a ready-made liveness/readiness probe.

## Configuration

| Var | Default | Meaning |
|-----|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `RAG_LLM_MODEL` / `RAG_EMBED_MODEL` | `llama3.2` / `nomic-embed-text` | models |
| `RAG_OFFLINE` | `0` | `1` forces the offline backend |
| `RAG_TOP_K` | `3` | evidence pooled per sub-query |
| `RAG_MAX_SUBQUERIES` / `RAG_MAX_STEPS` | `3` / `4` | agentic loop limits |
