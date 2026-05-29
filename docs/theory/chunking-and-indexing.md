# Chunking & Indexing

Cutting documents into chunks is the most consequential decision in the pipeline and the easiest one to tune. This page maps the choices to the recipes.

## Why chunking matters

A vector represents one chunk. If the chunk is too big, the vector averages over too many ideas and matches nothing well. Too small, and the vector loses the surrounding context that disambiguates it. The right size is corpus-dependent — recipe 4 shows the sweep.

## Four families

**Fixed-window** — slide a window across the document. The baseline. Recipe 2.

**Semantic** — cut where adjacent sentence embeddings disagree the most. Chunks align with topic boundaries. Recipe 5.

**Late chunking** — embed the whole document first, then pool token embeddings into chunks. Each chunk's vector carries information about the whole document. Recipe 6.

**Hierarchical** — index small chunks for search and return big ones for context (parent/child, recipe 9), or build a tree of summaries over chunks (RAPTOR, recipe 10).

## Context engineering

Three patterns let chunks carry more than their own text:

1. **Contextual headers** — Anthropic 2024, recipe 7. An LLM writes a one-sentence description of where each chunk fits, embedded with the chunk. Highest-ROI single change.
2. **Proposition decomposition** — recipe 8. Rewrite the document as atomic claims; index those. Wins on factual recall.
3. **Document summary routing** — recipe 11. Index per-document summaries; route the query to the right document first.

## Indexing the index

Once you have chunks, decide:

- **Vector index** — Qdrant, LanceDB, Chroma, Weaviate, Milvus, pgvector. Recipe defaults to Qdrant.
- **BM25 index** — `rank-bm25` for laptop scale, Tantivy/Elastic for production. Used in recipe 18.
- **Multi-vector index** — ColBERT-style late-interaction. Recipe 19.

The cookbook's `cookbook.stores` module wraps the first three with one API.

## The cost reality

Chunking is cheap. Embedding is medium (one model call per chunk). Storage is cheap unless you go to multi-vector at billions of chunks. Re-indexing is *expensive* — get the chunking right before you commit to a vector store deployment.
