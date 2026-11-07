"""Force the offline backend for the whole test session (no Ollama in CI)."""

import os

os.environ.setdefault("RAG_OFFLINE", "1")
