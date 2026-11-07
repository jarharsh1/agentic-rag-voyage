"""Pluggable embedding + LLM backends (Ollama-first, deterministic offline).

Same maritime-RAG pattern as the sibling repos, with two extra offline
capabilities the agentic loop needs: planning sub-queries and a grounding
check. Both are deterministic so the agent is fully testable without a model.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

from voyageiq.config import settings

_EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "what", "which", "when", "where", "who", "how", "why", "is", "are", "the",
    "a", "an", "of", "to", "and", "or", "for", "on", "in", "at", "did", "was",
    "were", "do", "does", "with", "that", "this", "it", "any",
}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def keywords(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOP and len(t) > 2]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


class HashingEmbedder:
    dim = _EMBED_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return vec if norm == 0 else [v / norm for v in vec]


class ExtractiveLLM:
    """Offline LLM: extractive synthesis over a CONTEXT/QUESTION prompt."""

    def generate(self, prompt: str) -> str:
        context, question = _split(prompt, "CONTEXT:", "QUESTION:")
        sentences = _sentences(context)
        if not sentences:
            return "I don't have enough context to answer that."
        q = set(keywords(question))
        scored = sorted(sentences, key=lambda s: len(q & set(keywords(s))), reverse=True)
        top = [s for s in scored[:3] if q & set(keywords(s))]
        return " ".join(top) if top else "The records do not appear to answer that."


class OllamaEmbedder:
    def __init__(self, host: str, model: str):
        self.host, self.model = host.rstrip("/"), model

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        with httpx.Client(timeout=60) as c:
            for t in texts:
                r = c.post(f"{self.host}/api/embeddings", json={"model": self.model, "prompt": t})
                r.raise_for_status()
                out.append(r.json()["embedding"])
        return out


class OllamaLLM:
    def __init__(self, host: str, model: str):
        self.host, self.model = host.rstrip("/"), model

    def generate(self, prompt: str) -> str:
        with httpx.Client(timeout=120) as c:
            r = c.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            r.raise_for_status()
            return r.json()["response"].strip()


def _split(prompt: str, a: str, b: str) -> tuple[str, str]:
    left, right = "", prompt
    if a in prompt and b in prompt:
        left = prompt.split(a, 1)[1].split(b, 1)[0].strip()
        right = prompt.split(b, 1)[1].strip()
    return left, right


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def _ollama_up(host: str) -> bool:
    try:
        with httpx.Client(timeout=2) as c:
            return c.get(f"{host.rstrip('/')}/api/tags").status_code == 200
    except Exception:
        return False


def get_backends() -> tuple[Embedder, LLM]:
    if not settings.offline and _ollama_up(settings.ollama_host):
        return (
            OllamaEmbedder(settings.ollama_host, settings.embed_model),
            OllamaLLM(settings.ollama_host, settings.llm_model),
        )
    return HashingEmbedder(), ExtractiveLLM()
