# Evaluation & Observability

You cannot improve what you cannot measure. This chapter covers what to measure and how to look at it.

## The four metrics that matter

RAGAS (recipe 37) gives you the canonical four:

1. **Faithfulness** — does the answer only claim things in the retrieved passages?
2. **Answer relevance** — does the answer address the question?
3. **Context precision** — are the retrieved passages on-topic?
4. **Context recall** — was anything important *missed*?

A good baseline holds the first two above 0.75. The latter two have wider variance; treat 5 % movement as noise.

## Why LLM-as-judge is OK

The four metrics use an LLM to judge. That sounds circular. It is acceptable because:

- The judge is held constant across runs; differences reveal pipeline differences.
- Judges are evaluated against human raters in the RAGAS / DeepEval test suites; they correlate well with humans on the metrics above.
- They are *much* cheaper than human raters and let you iterate dozens of times a day.

What they cannot do: judge factual correctness on topics they do not know. Always combine with one or two ground-truth questions (recipes 37 and 38 include some).

## Tracing

Tracing answers "*why* did this answer happen?" Phoenix (recipe 39) gives every LLM call, embedding call, and vector search a span in an OpenTelemetry trace. When a metric drops, you walk a representative failing trace and find the cause.

## Guardrails

Patronus Lynx (recipe 40) sits as a last gate before the user sees the answer. It scores whether the answer is grounded in the passages. Score < 0.5 → refuse. Production RAG systems run guardrails on every answer.

## The eval loop

1. Define a small eval set (this cookbook ships 80 hand-curated pairs).
2. Run it on your current pipeline. Record metrics.
3. Make one change. Re-run. Compare.
4. Repeat.

That is the loop. Everything else — DSPy compilation, agent harnesses, multi-modal indexes — is a faster way of doing more of step 3.
