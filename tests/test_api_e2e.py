"""End-to-end: ingest voyage records via the API and run the agentic loop."""

from pathlib import Path

from fastapi.testclient import TestClient

from voyageiq.api.main import app

client = TestClient(app)


def test_health_offline():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["backend"] == "offline"


def test_ingest_then_agentic_ask():
    doc = Path("data/voyage_records.md").read_text(encoding="utf-8")
    r = client.post("/ingest", json={"text": doc, "source": "voyage_records.md"})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 3

    r = client.post(
        "/ask",
        json={"question": "Why was the Jeddah call delayed and what was the demurrage impact?"},
    )
    assert r.status_code == 200
    body = r.json()
    # Agentic trace is exposed and the question was decomposed.
    assert body["trace"]["subqueries"]
    assert len(body["trace"]["subqueries"]) >= 2
    assert body["sources"]
    # The answer should surface the congestion delay and/or the demurrage figure.
    low = body["answer"].lower()
    assert "congestion" in low or "150,000" in body["answer"] or "demurrage" in low
