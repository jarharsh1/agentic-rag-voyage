"""Runtime configuration (env with local defaults). Ollama-first, offline fallback."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("RAG_LLM_MODEL", "llama3.2")
    embed_model: str = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
    offline: bool = os.getenv("RAG_OFFLINE", "0") == "1"
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("RAG_TOP_K", "3"))
    # Agentic loop controls.
    max_subqueries: int = int(os.getenv("RAG_MAX_SUBQUERIES", "3"))
    max_steps: int = int(os.getenv("RAG_MAX_STEPS", "4"))


settings = Settings()
