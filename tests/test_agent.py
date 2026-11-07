"""Unit tests for the agent's deterministic planning and grounding."""

from voyageiq.core.agent import grounding_score, plan_subqueries


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
    subs = plan_subqueries("a and b and c and d and e", max_subqueries=3)
    assert len(subs) <= 3


def test_grounding_score_bounds():
    assert grounding_score("", ["anything"]) == 0.0
    ev = ["refunds are issued to the original payment method within five business days"]
    assert grounding_score("refunds are issued to the original payment method", ev) > 0.5
    assert grounding_score("the vessel sank off iceland", ev) < 0.5
