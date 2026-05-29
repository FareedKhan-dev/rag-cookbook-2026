# Production Patterns

Notebooks are for understanding; production is for serving. This page collects the patterns that turn a clean recipe into a service that runs at 3 a.m. without paging anyone.

## Latency budget

A user-perceived RAG response has a hard ceiling around 1–2 seconds before it feels slow. Inside that budget:

| Step | Typical | Hard cap |
|---|---|---|
| Query rewrite (optional) | 100–300 ms | 500 ms |
| Embedding | 20–80 ms | 200 ms |
| Vector search (hybrid) | 5–50 ms | 200 ms |
| Reranking (cross-encoder) | 50–150 ms | 300 ms |
| Answer generation | 400–1500 ms | streaming |

Most teams overspend on rewriting and underspend on reranking. Measure before you optimize.

## Caching

Three layers, in order of payoff:

1. **Answer cache** — exact-query hits. Implementation: hash the query, key a Redis lookup. ~5 % hit rate at first; grows with traffic.
2. **Retrieval cache** — query embedding + ranked candidates. ~30 % hit rate on FAQs.
3. **Prompt cache** — Anthropic / OpenAI / Nebius prompt caching for the shared system prompt + context blob. Big cost saver, no quality impact.

## Cost control

- **Embedding** is the largest line item at index time, but it amortizes. Re-embed only when you change models.
- **LLM calls** dominate query-time cost. The biggest lever is *fewer hops* (recipes 24, 26, 27).
- **Reranking cost** scales with shortlist size. Cap at 30 candidates; do not feed 500.
- **Storage** is rarely the bottleneck unless you go multi-vector ColBERT at billions of chunks.

## Freshness

Three patterns:

1. **Batch reindex** — nightly. Simplest. Lag up to 24 h.
2. **Append-only upserts** — index new documents as they arrive. Lag minutes. Most production systems land here.
3. **Streaming differential RAG** — Pathway, Vespa Streaming. Sub-second freshness. Worth the complexity only when freshness IS the product.

## Security

- **Prompt injection** — every retrieved passage is untrusted input. Never let the model execute tools without a confirmation step.
- **Tenant isolation** — index per tenant or filter by tenant ID at query time. Auto-retrieval (recipe 20) extends to this.
- **PII** — strip at indexing if your corpus contains it. Recipes 7 and 11 give you the LLM-rewrite hook.

## Rollouts

Use feature flags on every retrieval change. Even a "free" change (RRF k tuning, MMR lambda) can quietly hurt latency or quality on some query slice. Roll out behind a flag, watch traces, only then ramp.

## What to monitor

- p50 / p95 / p99 query latency
- Cost per query
- RAGAS metrics on a held-out eval set, refreshed weekly
- Guardrail block rate (drift up = the model is hallucinating more, or the corpus is decaying)
- Cache hit rates

When any of those move 10 %, look at traces. Always look at traces first; only then change code.
