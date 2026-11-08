"""FastAPI serving layer for the agentic RAG engine.

Endpoints:
  GET  /         - the browser chat UI (static)
  GET  /health   - liveness + active backend (ollama/offline)
  POST /ingest   - add any text document to the corpus
  POST /ask      - agentic answer (plan -> retrieve -> synthesize -> ground-check),
                   returns the answer, citations, and the reasoning trace.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from voyageiq.core.backends import OllamaLLM
from voyageiq.core.pipeline import VoyagePipeline

_WEB_DIR = Path(__file__).resolve().parents[3] / "web"
_SAMPLE = Path("data/sample_knowledge_base.md")

app = FastAPI(
    title="Agentic RAG Engine",
    version="0.1.0",
    description="Ollama-first, domain-agnostic agentic retrieval-augmented QA over "
    "any text corpus. Each question runs a multi-step loop — plan sub-queries, "
    "retrieve evidence, synthesize a grounded answer, and ground-check it — "
    "returning the answer, its citations, and the full reasoning trace. Swap in "
    "your own documents.",
)

_pipeline = VoyagePipeline()

# Pre-ingest the bundled sample knowledge base so the UI works immediately.
if _SAMPLE.exists():
    try:
        _pipeline.ingest(_SAMPLE.read_text(encoding="utf-8"), source=_SAMPLE.name)
    except Exception as exc:  # pragma: no cover - startup best-effort
        print(f"[startup] could not ingest {_SAMPLE}: {exc}", file=sys.stderr)


class IngestRequest(BaseModel):
    text: str = Field(..., examples=["Refunds are issued to the original payment method within five business days of approval."])
    source: str = Field("inline", examples=["handbook.txt"])


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int


class AskRequest(BaseModel):
    question: str = Field(..., examples=["How long do I have to request a refund and how is it issued?"])


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    trace: dict


@app.get("/health")
def health() -> dict:
    backend = "ollama" if isinstance(_pipeline.llm, OllamaLLM) else "offline"
    return {"status": "ok", "backend": backend, "indexed_chunks": len(_pipeline.store)}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    added = _pipeline.ingest(req.text, req.source)
    return IngestResponse(chunks_added=added, total_chunks=len(_pipeline.store))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return AskResponse(**_pipeline.ask(req.question))


@app.post("/reset")
def reset() -> dict:
    """Clear the shared in-process corpus (the index is append-only otherwise)."""
    _pipeline.reset()
    return {"status": "ok", "indexed_chunks": len(_pipeline.store)}


# --- Browser UI ------------------------------------------------------------ #
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(_WEB_DIR / "index.html"))
