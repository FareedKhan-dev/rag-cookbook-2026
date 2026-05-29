# Query Transformation

Users do not ask the questions your corpus answers. Query transformation closes the gap.

## Why queries miss

A user query is short and uses the user's vocabulary; a document passage is long and uses the author's. Their embeddings live in different parts of the space. Even when the answer is in the corpus, cosine similarity can fail to bridge the linguistic distance.

## Five strategies

1. **HyDE** (recipe 13) — Hallucinate a fake answer first; embed *that*. Fake answers live near real ones.
2. **Multi-query / RAG-Fusion** (recipe 14) — Generate N paraphrases; retrieve for each; fuse with Reciprocal Rank Fusion. Coverage jumps.
3. **Step-back** (recipe 15) — Ask a more abstract version of the question; retrieve principles; combine with the specific query. Wins on reasoning questions.
4. **Sub-question decomposition** (recipe 16) — Break complex queries into sub-questions; retrieve and answer each; compose. Wins when the question has multiple parts.
5. **Semantic routing** (recipe 17) — Classify the query and dispatch to the right index. Free at scale.

## When to use which

| If the user's query is… | Try |
|---|---|
| Long and over-specific | HyDE |
| Ambiguous about scope | Multi-query |
| A reasoning question on a textbook | Step-back |
| Multi-part ("X and how does it relate to Y") | Sub-question |
| Likely about one of many indexed sources | Semantic routing |

In production, real systems stack them: route first, then fan out paraphrases, then fuse. Costs add up fast — measure before stacking three.

## Failure modes

- **Garbage in, garbage out.** A paraphraser that misunderstands the question generates queries that retrieve confidently wrong passages.
- **Stylistic drift.** HyDE answers that read like Wikipedia retrieve Wikipedia; if your corpus is technical SEC filings, HyDE drags you to the wrong register.
- **Cost.** Five paraphrases means five embedding calls *and* five retrieval calls per query.
