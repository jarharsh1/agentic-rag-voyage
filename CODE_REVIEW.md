# Code Review — `agentic-rag-voyage`

A reviewer-facing deep-dive on the correctness and robustness findings fixed in
the second review round for the agentic loop (plan → retrieve → synthesize →
ground-check → optional refine), with root causes, the actual before/after from
each fix commit, and the regression test that locks each one in.

Scope: the offline (CI/test) path — `HashingEmbedder` + `ExtractiveLLM`, with
deterministic planning and grounding. Fixes keep the Ollama-backed path correct.

Commits reviewed: `e863127`, `eab6fbd`.

---

## 1. Citations didn't match the synthesized context — `src/voyageiq/core/agent.py` (`answer`)

**Where.** `VoyageAgent.answer`: the evidence that `_synthesize` consumed vs. the
evidence reported in `sources`.

**Root cause.** The answer was synthesized over the **entire pooled evidence**
set (every unique chunk retrieved across all sub-queries), but `sources` was then
a **truncation to `top_k`** of that same pool:

```python
evidence = [rec for rec, _ in sorted(pooled.values(), key=..., reverse=True)]
draft = self._synthesize(question, evidence)          # over ALL pooled evidence
...
sources = [
    {"source": rec.source, "excerpt": rec.text[:200]}
    for rec in evidence[: settings.top_k]             # but cite only top_k
]
```

So the answer could draw a fact from the 5th-ranked chunk while only the first 3
were cited — the citation list silently misrepresented what the answer was built
from. For a tool whose whole value proposition is a *grounded, traceable* answer,
a sources≠context mismatch is a real correctness bug, not a cosmetic one.

**Fix (`eab6fbd`).** Establish a **sources == context** invariant: synthesize
over, ground-check against, and cite the *same* `top_k` set.

```python
# Synthesize over, ground-check against, and cite the SAME evidence set
# (the top_k most relevant chunks), so the citations always match the
# context the answer was actually built from.
evidence = ordered[: settings.top_k]
draft = self._synthesize(question, evidence)
...
sources = [
    {"source": rec.source, "excerpt": rec.text[:200]}
    for rec in evidence                                # exactly what was synthesized
]
```

**Design trade-off (deliberate).** The alternative was "cite *all* pooled
evidence" (keep synthesizing over everything, widen the citation list to match).
I chose "bound synthesis to `top_k` and cite that set" instead because: (a) it
keeps the cited list short and honest; (b) with the offline extractive LLM,
`top_k=3` chunks already supply ample sentences, and measured grounding on the
sample KB stays `1.0`; (c) it bounds the synthesis context deterministically. The
cost: the offline model sees slightly fewer chunks than before. If a future need
arises to reason over the full pool, flip to "cite-all" — it's a one-line change
to the `sources` comprehension — but the invariant (sources == context) must hold
either way.

**Regression test** (`tests/test_agent.py`):

```python
def test_sources_match_synthesized_evidence_and_rank_relevant_first():
    agent = _agent_with_corpus()
    result = agent.answer("How long do I have to request a refund and how is it issued?")
    sources = result["sources"]
    assert 0 < len(sources) <= settings.top_k
    assert "refund" in sources[0]["excerpt"].lower()   # relevant chunk cited first
    assert result["trace"]["grounding"] > 0.5
```

---

## 2. Offline evidence ranked by noisy cosine — `src/voyageiq/core/agent.py` (`_order_evidence`)

**Where.** `VoyageAgent.answer`, the ordering of `pooled.values()` before
synthesis and citation.

**Root cause.** Evidence was ordered purely by cosine score:

```python
evidence = [rec for rec, _ in sorted(pooled.values(), key=lambda t: t[1], reverse=True)]
```

With `HashingEmbedder`, cosine is dominated by hash-collision noise (unrelated
chunks score ~0.1 rather than 0). A genuinely relevant chunk can therefore be
ranked *below* noise, so the `top_k` slice fed to synthesis and citation may omit
the chunk that actually answers the question.

**Fix (`eab6fbd`).** A backend-aware ordering: offline, rank by how many question
content words each chunk actually contains (ties broken by cosine); online, keep
cosine order.

```python
def _order_evidence(self, question, pooled):
    if isinstance(self.embedder, HashingEmbedder):
        q_kw = set(keywords(question))
        return [rec for rec, _ in sorted(
            pooled.values(),
            key=lambda t: (len(q_kw & self._rec_keywords(t[0])), t[1]),
            reverse=True,
        )]
    return [rec for rec, _ in sorted(pooled.values(), key=lambda t: t[1], reverse=True)]
```

**Design trade-off (deliberate).** Lexical-overlap ranking is only appropriate
*because* the offline embedder is lexical in nature. A real embedder's whole
point is to match paraphrases that share no surface words, so overriding its
cosine order with word counts would defeat it — hence the
`isinstance(..., HashingEmbedder)` gate. Verified on the sample KB: the refund
chunk (overlap 3) now ranks first instead of being buried under noise-floor
cosine. The same gate reasoning backs the lexical-overlap *guard* in `answer`
(it suppresses noise-floor citations for unanswerable questions, offline only).

---

## 3. Refine retrieval ignored a stronger re-hit — `src/voyageiq/core/agent.py` (`answer`, refine branch)

**Where.** The refinement loop that runs when grounding is weak.

**Root cause.** The planning loop pools with a **max-score** update
(`or score > pooled[rec.id][1]`), but the refine loop only added **brand-new**
ids:

```python
for rec, score in extra:
    if rec.id not in pooled:        # never updates an existing chunk's score
        pooled[rec.id] = (rec, score)
```

If refinement re-retrieved an already-pooled chunk with a *higher* score (which
is the signal that the refinement query found it more relevant), that stronger
score was discarded and the stale one kept — so a chunk the refinement
specifically surfaced could fail to rank up.

**Fix (`eab6fbd`).** Mirror the planning loop's max-score update:

```python
for rec, score in extra:
    # Mirror the planning loop: keep the max score so a strongly
    # re-retrieved chunk ranks correctly instead of keeping a stale one.
    if rec.id not in pooled or score > pooled[rec.id][1]:
        pooled[rec.id] = (rec, score)
```

This is covered by the existing refine/grounding tests; it changes ranking only
when a re-hit beats the stored score, which is exactly the intended behavior.

---

## 4. `keywords()` re-tokenized each chunk 2–3× per answer — `src/voyageiq/core/agent.py`

**Where.** `grounding_score`, the overlap guard, and (now) `_order_evidence` each
called `keywords(rec.text)` on the same chunks within a single `answer()`.

**Root cause.** No memoization: the grounding check, the lexical-overlap guard,
and evidence ordering independently re-ran the regex tokenizer + stopword filter
over the same record texts.

**Fix (`eab6fbd`).** A per-answer keyword cache keyed by record id, reset at the
top of each `answer()`:

```python
def _rec_keywords(self, rec: Record) -> set[str]:
    kw = self._kw_cache.get(rec.id)
    if kw is None:
        kw = set(keywords(rec.text))
        self._kw_cache[rec.id] = kw
    return kw
```

```python
def answer(self, question: str) -> dict:
    self._kw_cache = {}            # fresh per answer
    ...
```

**Correctness note.** The cache lifetime is a single `answer()` call (reset on
entry), so it can't leak stale keywords across questions or across store
mutations. The now-redundant module-level `_has_lexical_overlap` was removed and
replaced by the cache-backed method `_has_overlap`.

---

## 5. Startup pre-ingest silently did nothing off-root — `src/voyageiq/api/main.py:25`

**Where.** Module load time, `_SAMPLE` definition.

**Root cause.** Same latent bug as its siblings: `_WEB_DIR` anchored to
`__file__` but `_SAMPLE` was CWD-relative, so the pre-ingest guard was false
whenever the server launched off the repo root.

**Fix (`e863127`).**

```python
_ROOT = Path(__file__).resolve().parents[3]
_WEB_DIR = _ROOT / "web"
_SAMPLE = _ROOT / "data" / "sample_knowledge_base.md"
```

**Regression test** (`tests/test_api_e2e.py`): asserts `_SAMPLE.is_absolute()`
and `.exists()`.

---

## Verified correct, not changed

- **The refine step budget** (`steps_used < settings.max_steps`, from the prior
  round) genuinely bounds the loop now; the earlier dead guard
  (`len(subs) < max_steps`, always true since subs ≤ `max_subqueries` < `max_steps`)
  was already replaced by a real per-round counter.
- **The lexical-overlap guard** still returns an honest no-answer with empty
  sources and `refined=False` for a gibberish question — verified by
  `test_gibberish_question_cites_nothing_and_does_not_refine`.
- **`plan_subqueries`** dedups on a normalized key, always lists the full question
  first, caps at `max_subqueries`, and emits no empty sub-queries.
- **`grounding_score`** is division-safe (empty answer → `0.0`) and the trace it
  reports matches the evidence actually used.
