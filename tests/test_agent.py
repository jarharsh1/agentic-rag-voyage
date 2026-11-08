"""Unit tests for the agent's deterministic planning and grounding."""

from voyageiq.core.agent import VoyageAgent, grounding_score, plan_subqueries
from voyageiq.core.backends import ExtractiveLLM, HashingEmbedder
from voyageiq.core.vector_store import VectorStore


def test_plan_splits_multipart_question():
    subs = plan_subqueries(
        "How long do I have to request a refund and how is it issued?",
        max_subqueries=3,
    )
    # Full question first, then at least one focused sub-query.
    assert subs[0].startswith("How long do I have to request a refund")
    assert len(subs) >= 2
    assert any("issued" in s.lower() for s in subs)


def test_plan_single_question_has_one_subquery():
    subs = plan_subqueries("How long is personal data retained?", max_subqueries=3)
    assert len(subs) == 1


def test_plan_respects_cap():
    # Multi-character parts so keywords() keeps them and the cap is actually
    # exercised: five content-bearing clauses must be capped to three.
    subs = plan_subqueries(
        "alpha and beta and gamma and delta and epsilon", max_subqueries=3
    )
    assert len(subs) == 3
    assert subs[0] == "alpha and beta and gamma and delta and epsilon"


def test_grounding_score_bounds():
    assert grounding_score("", ["anything"]) == 0.0
    ev = ["refunds are issued to the original payment method within five business days"]
    assert grounding_score("refunds are issued to the original payment method", ev) > 0.5
    assert grounding_score("the vessel sank off iceland", ev) < 0.5


def _agent_with_corpus() -> VoyageAgent:
    emb = HashingEmbedder()
    store = VectorStore()
    # A corpus rich enough that the hashed embedder's collision noise floor makes
    # even an unrelated query retrieve chunks (cosine > 0) — exactly the case the
    # lexical-overlap guard must reject.
    chunks = [
        "A customer may request a refund within 30 days of purchase and refunds are issued to the original payment method within five business days.",
        "Priority incidents affecting production are acknowledged within one hour and are updated every four hours until the incident is fully resolved.",
        "Personal data is retained for twenty four months after the last account activity after which it is anonymized and may be deleted on request.",
        "New users must verify their email address before accessing the dashboard and administrators can invite additional team members at any time.",
        "Passwords must be at least twelve characters long and are stored only as salted hashes and two factor authentication is required for admins.",
        "Invoices are issued on the first of each month and are payable within fourteen days and overdue accounts are suspended until the balance clears.",
    ]
    store.add(chunks, emb.embed(chunks), source="kb.md")
    return VoyageAgent(emb, ExtractiveLLM(), store)


def test_gibberish_question_cites_nothing_and_does_not_refine():
    # The question shares no vocabulary with any chunk. The hashed embedder still
    # returns chunks (collision noise floor), but the agent must NOT cite them or
    # mark the run as refined: it should answer honestly with empty sources.
    agent = _agent_with_corpus()
    result = agent.answer("xyzzy quux zzz unmatchedword")
    assert result["sources"] == []
    assert result["trace"]["refined"] is False
    assert result["trace"]["grounding"] == 0.0


def test_real_question_still_cites_and_can_refine():
    # A genuinely answerable question keeps the normal retrieve/cite path.
    agent = _agent_with_corpus()
    result = agent.answer("How long do I have to request a refund?")
    assert result["sources"], "expected at least one citation for an answerable question"
    assert "30" in result["answer"]
