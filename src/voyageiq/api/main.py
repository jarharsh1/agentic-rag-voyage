"""FastAPI serving layer for agentic voyage & port-call intelligence.

Endpoints:
  GET  /health   - liveness + active backend
  POST /ingest   - add a voyage report / port circular / incident log
  POST /ask      - agentic answer (plan -> retrieve -> synthesize -> ground-check),
                   returns the answer, citations, and the reasoning trace.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from voyageiq.core.backends import OllamaLLM
from voyageiq.core.pipeline import VoyagePipeline

app = FastAPI(
    title="Voyage & Port-Call Intelligence (Agentic RAG)",
    version="0.1.0",
    description="Ollama-first agentic RAG over voyage reports, port circulars and "
    "incident logs — a Bahri-style fleet-operations use case.",
)

_pipeline = VoyagePipeline()


class IngestRequest(BaseModel):
    text: str = Field(..., examples=["Vessel arrived Jeddah 02:00; berth delayed 6h due to congestion."])
    source: str = Field("inline", examples=["voyage-report-0423.txt"])


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int


class AskRequest(BaseModel):
    question: str = Field(..., examples=["Why was the Jeddah call delayed and what was the demurrage impact?"])


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
