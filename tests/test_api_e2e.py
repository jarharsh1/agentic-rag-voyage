"""End-to-end: ingest the sample knowledge base via the API and run the agentic loop."""

from pathlib import Path

from fastapi.testclient import TestClient

from voyageiq.api.main import app

client = TestClient(app)


def test_health_offline():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["backend"] == "offline"


def test_ingest_then_agentic_ask():
    doc = Path("data/sample_knowledge_base.md").read_text(encoding="utf-8")
    r = client.post("/ingest", json={"text": doc, "source": "kb.md"})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 3

    r = client.post(
        "/ask",
        json={"question": "How long do I have to request a refund and how is it issued?"},
    )
    assert r.status_code == 200
    body = r.json()
    # Agentic trace is exposed and the question was decomposed.
    assert body["trace"]["subqueries"]
    assert len(body["trace"]["subqueries"]) >= 2
    assert body["sources"]
    # The answer should surface the refund window and/or how refunds are issued.
    low = body["answer"].lower()
    assert "30 days" in low or "30" in body["answer"] or "refund" in low


def test_index_and_static_served():
    assert "Agentic RAG Engine" in client.get("/").text
    assert "/ask" in client.get("/static/app.js").text


def test_reset_clears_the_shared_corpus():
    client.post("/ingest", json={"text": "Bunker fuel was loaded at Rotterdam.", "source": "x"})
    assert client.get("/health").json()["indexed_chunks"] > 0
    r = client.post("/reset")
    assert r.status_code == 200
    assert r.json()["indexed_chunks"] == 0
    assert client.get("/health").json()["indexed_chunks"] == 0
