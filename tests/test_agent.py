"""Unit tests for the agent's deterministic planning and grounding."""

from voyageiq.core.agent import grounding_score, plan_subqueries


def test_plan_splits_multipart_question():
    subs = plan_subqueries(
        "Why was the Jeddah call delayed and what was the demurrage impact?",
        max_subqueries=3,
    )
    # Full question first, then at least one focused sub-query.
    assert subs[0].startswith("Why was the Jeddah call delayed")
    assert len(subs) >= 2
    assert any("demurrage" in s.lower() for s in subs)


def test_plan_single_question_has_one_subquery():
    subs = plan_subqueries("What speed did the vessel average?", max_subqueries=3)
    assert len(subs) == 1


def test_plan_respects_cap():
    subs = plan_subqueries("a and b and c and d and e", max_subqueries=3)
    assert len(subs) <= 3


def test_grounding_score_bounds():
    assert grounding_score("", ["anything"]) == 0.0
    ev = ["demurrage of approximately USD 150,000 at the charter rate"]
    assert grounding_score("demurrage was USD 150,000", ev) > 0.5
    assert grounding_score("the vessel sank off iceland", ev) < 0.5
