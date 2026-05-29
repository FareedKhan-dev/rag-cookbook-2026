# Reranking & Fusion

Retrieval finds candidates. Reranking re-orders them. Fusion combines multiple rankings into one.

## Why rerank

Bi-encoders (dense retrieval) score queries and chunks independently. Cross-encoders score them jointly and can catch fine-grained signal a bi-encoder misses. The trade-off is speed — cross-encoders cannot index, they must be called per pair. The solution is to shortlist with a bi-encoder, rerank with a cross-encoder.

## Cross-encoder options (recipe 22)

| Model | Where it shines |
|---|---|
| `bge-reranker-v2-m3` | Open weights, strong multilingual baseline. |
| Cohere `rerank-4` | Hosted, top-of-the-line latency. |
| `jina-reranker-v3` | Listwise — sees the whole candidate set at once. |
| `voyage-rerank-2.5` | Hosted, very strong for English. |

## Listwise LLM reranking (recipe 23)

Sometimes you want the ranker to *reason* about candidates ("these two contradict each other; this one is from 2018 and outdated"). Ask a small fast LLM to reorder. Pattern coined as RankGPT in 2023; still going strong in 2026 with smaller, cheaper models.

## Fusion

Reciprocal Rank Fusion sums `1 / (k + rank)` across rankings. Parameter-free (`k=60` is standard), robust to outliers, works on any two rankings — dense+sparse, multi-query, step-back. The `cookbook.retrievers.reciprocal_rank_fusion` helper is used everywhere.

## How to think about the stack

1. **Bi-encoder** (~5 ms, top-100). Coarse filter.
2. **Cross-encoder** (~50 ms, top-20). Fine filter.
3. **Optional listwise LLM** (~500 ms, top-5). Reasoned filter, only when the budget allows.

Stop at whichever level your eval metrics flatten. Most production RAG systems stop at level 2.
