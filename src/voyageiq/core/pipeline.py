"""Thin pipeline: ingest voyage/port records, then answer via the agentic loop."""

from __future__ import annotations

from voyageiq.config import settings
from voyageiq.core.agent import VoyageAgent
from voyageiq.core.backends import get_backends
from voyageiq.core.chunking import chunk_text
from voyageiq.core.vector_store import VectorStore


class VoyagePipeline:
    def __init__(self, store: VectorStore | None = None):
        self.embedder, self.llm = get_backends()
        self.store = store or VectorStore()
        self.agent = VoyageAgent(self.embedder, self.llm, self.store)

    def ingest(self, text: str, source: str) -> int:
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            return 0
        self.store.add(chunks, self.embedder.embed(chunks), source)
        return len(chunks)

    def ask(self, question: str) -> dict:
        return self.agent.answer(question)

    def reset(self) -> None:
        """Drop all ingested records. The shared in-process store is otherwise
        append-only, so this lets callers reclaim it between corpora/sessions."""
        self.store.clear()
