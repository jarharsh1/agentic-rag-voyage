"""In-memory cosine vector store with source metadata for citations."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Record:
    id: int
    text: str
    embedding: list[float]
    source: str


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


class VectorStore:
    def __init__(self):
        self._records: list[Record] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, texts: list[str], embeddings: list[list[float]], source: str) -> None:
        for text, emb in zip(texts, embeddings):
            self._records.append(Record(len(self._records), text, emb, source))

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[Record, float]]:
        scored = [(r, cosine(query_embedding, r.embedding)) for r in self._records]
        scored.sort(key=lambda t: t[1], reverse=True)
        # Drop clean-zero scores (no shared dimensions at all). This does NOT
        # remove the offline hashed embedder's collision noise floor; callers
        # that need true relevance apply a lexical-overlap check on top.
        return [(r, s) for r, s in scored[:top_k] if s > 0]

    def clear(self) -> None:
        self._records.clear()
