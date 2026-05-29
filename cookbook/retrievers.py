"""Retriever building blocks shared across recipes 18–21.

These are plain functions, not classes — easier to read in a notebook cell
and easier to recombine. Backend-agnostic; expects anything that exposes
`.search(vector, top_k)`.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

import numpy as np

from .stores import Hit, VectorBackend


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[Hit]:
    """RRF score = sum(1 / (k + rank)) across ranked lists."""
    scored: dict[str, float] = defaultdict(float)
    keepers: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            scored[hit.doc_id] += 1.0 / (k + rank)
            keepers.setdefault(hit.doc_id, hit)
    fused = [
        Hit(doc_id=h.doc_id, text=h.text, score=scored[h.doc_id], metadata=h.metadata)
        for h in keepers.values()
    ]
    fused.sort(key=lambda h: h.score, reverse=True)
    return fused[:top_k] if top_k else fused


# ---------------------------------------------------------------------------
# Maximal Marginal Relevance
# ---------------------------------------------------------------------------
def mmr(
    query_vector: Sequence[float],
    candidate_vectors: Sequence[Sequence[float]],
    candidates: Sequence[Hit],
    *,
    lambda_mult: float = 0.5,
    top_k: int = 5,
) -> list[Hit]:
    """Greedy MMR diversification over an initial candidate pool."""
    q = np.asarray(query_vector, dtype=np.float32)
    q /= np.linalg.norm(q).clip(min=1e-9)
    vectors = np.asarray(candidate_vectors, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-9)

    sims_to_query = vectors @ q
    chosen: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(chosen) < top_k:
        if not chosen:
            pick = int(max(remaining, key=lambda i: sims_to_query[i]))
        else:
            chosen_vecs = vectors[chosen]

            def score(i: int) -> float:
                diversity = float(np.max(vectors[i] @ chosen_vecs.T))
                return lambda_mult * sims_to_query[i] - (1 - lambda_mult) * diversity

            pick = int(max(remaining, key=score))
        chosen.append(pick)
        remaining.remove(pick)
    return [candidates[i] for i in chosen]


# ---------------------------------------------------------------------------
# Hybrid (dense + BM25)
# ---------------------------------------------------------------------------
def hybrid_search(
    query: str,
    query_vector: Sequence[float],
    backend: VectorBackend,
    bm25_corpus: Sequence[str],
    bm25_ids: Sequence[str],
    bm25_meta: Sequence[dict],
    *,
    dense_k: int = 20,
    sparse_k: int = 20,
    fused_k: int = 10,
) -> list[Hit]:
    """Combine dense retrieval from `backend` with BM25 over `bm25_corpus`."""
    from rank_bm25 import BM25Okapi

    dense = backend.search(query_vector, top_k=dense_k)

    tokenized = [doc.lower().split() for doc in bm25_corpus]
    bm25 = BM25Okapi(tokenized)
    sparse_scores = bm25.get_scores(query.lower().split())
    sparse_idx = np.argsort(sparse_scores)[::-1][:sparse_k]
    sparse = [
        Hit(
            doc_id=bm25_ids[i],
            text=bm25_corpus[i],
            score=float(sparse_scores[i]),
            metadata=bm25_meta[i],
        )
        for i in sparse_idx
    ]
    return reciprocal_rank_fusion([dense, sparse], top_k=fused_k)


# ---------------------------------------------------------------------------
# Step-back retrieval (recipe 15)
# ---------------------------------------------------------------------------
def step_back_retrieve(
    query: str,
    backend: VectorBackend,
    chat: Callable[[str], str],
    embed: Callable[[Sequence[str]], list[list[float]]],
    *,
    top_k: int = 8,
) -> list[Hit]:
    """Retrieve with both the original query and an abstracted parent query."""
    abstract_prompt = (
        "Rewrite the user's question into a more general, higher-level question "
        "that asks about the underlying principle or category. Output only the "
        "rewritten question.\n\nQuestion: " + query
    )
    parent = chat(abstract_prompt).strip()
    vectors = embed([query, parent])
    rankings = [
        backend.search(vectors[0], top_k=top_k),
        backend.search(vectors[1], top_k=top_k),
    ]
    return reciprocal_rank_fusion(rankings, top_k=top_k)


__all__ = [
    "reciprocal_rank_fusion",
    "mmr",
    "hybrid_search",
    "step_back_retrieve",
]
