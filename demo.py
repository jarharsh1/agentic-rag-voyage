"""End-to-end CLI demo: ingest voyage records and run the agentic loop.

    python demo.py            # Ollama if up, else offline
    RAG_OFFLINE=1 python demo.py
"""

from pathlib import Path

from voyageiq.core.pipeline import VoyagePipeline

QUESTIONS = [
    "How long do I have to request a refund and how is it issued?",
    "What are the SLA and data-retention rules?",
    "What are the security requirements?",
]


def main() -> None:
    doc = Path("data/sample_knowledge_base.md").read_text(encoding="utf-8")
    pipe = VoyagePipeline()
    n = pipe.ingest(doc, source="sample_knowledge_base.md")
    print(f"Ingested {n} chunks (backend: {type(pipe.llm).__name__})\n")

    for q in QUESTIONS:
        result = pipe.ask(q)
        trace = result["trace"]
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"   [sub-queries: {len(trace['subqueries'])}, "
              f"grounding: {trace['grounding']}, refined: {trace['refined']}]")
        print()


if __name__ == "__main__":
    main()
