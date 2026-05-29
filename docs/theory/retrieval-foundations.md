# Retrieval Foundations

If you only read one theory page, read this one.

## Why RAG at all

Large language models forget two things: anything that happened after training, and anything that was never in training. Retrieval-augmented generation puts a search engine in front of the model. The search engine finds passages relevant to the question; the model uses them to answer. The model gets to be small and current instead of huge and stale.

That is the whole pitch. Everything below is implementation detail.

## The unavoidable trade-off

Every retrieval system trades **precision** against **recall**. Precision is "did I retrieve junk?"; recall is "did I miss the right answer?". A pipeline that takes only the single closest chunk has high precision and low recall; a pipeline that takes the top thousand has the opposite. The whole rest of this cookbook is variations on tipping that balance.

## The five jobs

Every RAG system does five things, even when it looks like it does fewer:

1. **Chunk** — cut documents into pieces small enough to retrieve.
2. **Embed** — represent pieces as vectors.
3. **Store** — index those vectors so cosine search is fast.
4. **Retrieve** — given a query, find candidate pieces.
5. **Generate** — use the candidates as context for an LLM call.

The interesting variation is between (3) and (5). Recipes 5–12 obsess over (1); 13–17 rewrite the query before (4); 18–21 change (4) itself; 22–23 re-order what (4) returned; 24–30 wrap the whole loop in an agent that can repeat (3)–(5).

## Vocabulary

- **Dense retrieval** — cosine similarity between query and chunk embeddings.
- **Sparse retrieval** — BM25 over token frequencies. Catches exact-match terms dense retrieval drops.
- **Hybrid retrieval** — fuse dense and sparse rankings, usually with Reciprocal Rank Fusion.
- **Late interaction** — store per-token vectors, compute MaxSim at query time. ColBERT.
- **Cross-encoder** — a model that takes (query, chunk) jointly and outputs a relevance score. Slow but accurate.
- **Reranker** — anything that re-orders an initial candidate set. Usually a cross-encoder or a small LLM.

## When the answer is "you don't need RAG"

- The question is answered in the model's training data and doesn't need to be recent.
- The whole corpus fits in the model's context window and the per-token cost is acceptable.
- The question is conversational ("what did I just say?") — that's memory, recipe 34, not retrieval.

When RAG is the right answer, the rest of this cookbook helps you make it work in production.
