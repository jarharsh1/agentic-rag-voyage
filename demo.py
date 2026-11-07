"""End-to-end CLI demo: ingest voyage records and run the agentic loop.

    python demo.py            # Ollama if up, else offline
    RAG_OFFLINE=1 python demo.py
"""

from pathlib import Path

from voyageiq.core.pipeline import VoyagePipeline

QUESTIONS = [
    "Why was the Jeddah call delayed and what was the demurrage impact?",
    "What new rule did Fujairah announce and what must dangerous-goods vessels file?",
    "Was there any engine problem during the voyage and did it cause off-hire?",
]


def main() -> None:
    doc = Path("data/voyage_records.md").read_text(encoding="utf-8")
    pipe = VoyagePipeline()
    n = pipe.ingest(doc, source="voyage_records.md")
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
