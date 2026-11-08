"""The agentic RAG loop.

Rather than a single retrieve-then-generate pass, the agent:

  1. PLAN      - decompose a multi-part question into focused sub-queries.
  2. RETRIEVE  - gather evidence for each sub-query (deduped across the question).
  3. SYNTHESIZE- draft a grounded answer from the pooled evidence.
  4. GROUND-CHECK - verify the draft is supported by the evidence; if weak,
                 retrieve once more with the draft's own terms and re-synthesize.

Planning and grounding are deterministic functions so the whole agent is
testable offline; the LLM is only used for the final synthesis wording.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from voyageiq.config import settings
from voyageiq.core.backends import LLM, Embedder, keywords
from voyageiq.core.vector_store import Record, VectorStore

_SYNTH_PROMPT = """You are a voyage-operations analyst. Answer the question using
ONLY the evidence below. Be specific (ports, dates, figures). If the evidence is
insufficient, say so.

CONTEXT:
{context}

QUESTION:
{question}
"""


def plan_subqueries(question: str, max_subqueries: int) -> list[str]:
    """Split a multi-part question into focused sub-queries (deterministic).

    Breaks on ' and ' / '?' / ';' and keeps parts that carry content words.
    Always includes the full question as the first sub-query.
    """
    raw = question.replace("?", "?|").replace(";", ";|")
    for conj in (" and ", " & "):
        raw = raw.replace(conj, "|")
    parts = [p.strip(" ?;|") for p in raw.split("|")]
    subs = [question.strip()]
    for p in parts:
        if p and p != question.strip() and keywords(p):
            subs.append(p)
    # Deduplicate on a normalised key (case- and trailing-punctuation-insensitive),
    # preserve order, and cap. This keeps a bare restatement of the full question
    # from counting as a separate sub-query.
    seen, out = set(), []
    for s in subs:
        key = s.strip().rstrip("?.;! ").lower()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
        if len(out) >= max_subqueries:
            break
    return out


def grounding_score(answer: str, evidence: list[str]) -> float:
    """Fraction of the answer's content words that appear in the evidence.

    A cheap, deterministic proxy for 'is this answer supported by context'.
    """
    ans_kw = set(keywords(answer))
    if not ans_kw:
        return 0.0
    ev_kw = set()
    for e in evidence:
        ev_kw |= set(keywords(e))
    return len(ans_kw & ev_kw) / len(ans_kw)


def _has_lexical_overlap(question: str, evidence: list[Record]) -> bool:
    """True if any question content word appears in any retrieved chunk.

    The offline hashed embedder collides tokens into a fixed-dimension space, so
    cosine similarity never scores a clean zero even for unrelated text; the
    vector store therefore always returns chunks. This guard distinguishes
    genuinely relevant hits from that noise floor before we cite or refine.
    """
    q_kw = set(keywords(question))
    if not q_kw:
        return False
    for rec in evidence:
        if q_kw & set(keywords(rec.text)):
            return True
    return False


@dataclass
class AgentTrace:
    subqueries: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    grounding: float = 0.0
    refined: bool = False


class VoyageAgent:
    def __init__(self, embedder: Embedder, llm: LLM, store: VectorStore):
        self.embedder = embedder
        self.llm = llm
        self.store = store

    def _retrieve(self, query: str, k: int) -> list[tuple[Record, float]]:
        q_emb = self.embedder.embed([query])[0]
        return self.store.search(q_emb, k)

    def answer(self, question: str) -> dict:
        trace = AgentTrace()
        if len(self.store) == 0:
            return {"answer": "No voyage records have been ingested yet.",
                    "sources": [], "trace": trace.__dict__}

        # 1 + 2: plan and retrieve per sub-query, pooling unique evidence.
        # Each retrieval round counts as one agent step, so the loop as a whole
        # is bounded by settings.max_steps (initial rounds + optional refine).
        subs = plan_subqueries(question, settings.max_subqueries)
        trace.subqueries = subs
        pooled: dict[int, tuple[Record, float]] = {}
        steps_used = 0
        for sq in subs:
            hits = self._retrieve(sq, settings.top_k)
            steps_used += 1
            trace.steps.append({"subquery": sq, "retrieved": [r.id for r, _ in hits]})
            for rec, score in hits:
                if rec.id not in pooled or score > pooled[rec.id][1]:
                    pooled[rec.id] = (rec, score)

        evidence = [rec for rec, _ in sorted(pooled.values(), key=lambda t: t[1], reverse=True)]

        # If the question shares no vocabulary with any retrieved chunk, the hits
        # are pure embedding noise (the offline hashed embedder never scores a
        # clean zero). Don't synthesize an answer over, refine on, or cite them:
        # answer honestly with empty sources and an un-refined trace.
        if not _has_lexical_overlap(question, evidence):
            trace.grounding = 0.0
            return {
                "answer": "The records do not appear to answer that.",
                "sources": [],
                "trace": trace.__dict__,
            }

        draft = self._synthesize(question, evidence)

        # 4: grounding check; one refinement retrieval if weak and we still have
        # step budget. steps_used bounds the loop by settings.max_steps for real
        # (len(subs) alone is capped at max_subqueries and never bound it).
        trace.grounding = round(grounding_score(draft, [r.text for r in evidence]), 3)
        if trace.grounding < 0.5 and steps_used < settings.max_steps:
            trace.refined = True
            steps_used += 1
            extra = self._retrieve(draft, settings.top_k)
            for rec, score in extra:
                if rec.id not in pooled:
                    pooled[rec.id] = (rec, score)
            evidence = [r for r, _ in sorted(pooled.values(), key=lambda t: t[1], reverse=True)]
            draft = self._synthesize(question, evidence)
            trace.grounding = round(grounding_score(draft, [r.text for r in evidence]), 3)

        sources = [
            {"source": rec.source, "excerpt": rec.text[:200]}
            for rec in evidence[: settings.top_k]
        ]
        return {"answer": draft, "sources": sources, "trace": trace.__dict__}

    def _synthesize(self, question: str, evidence: list[Record]) -> str:
        context = "\n\n".join(f"[{i + 1}] {r.text}" for i, r in enumerate(evidence))
        return self.llm.generate(_SYNTH_PROMPT.format(context=context, question=question))
