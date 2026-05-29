"""Reference baseline pipelines that every other recipe compares against.

The vanilla pipeline here is *the* baseline cited in every later recipe's
"Comparison" section. Recipes import it as a single line:

    from cookbook.baselines import vanilla_pipeline

This avoids notebook-to-notebook imports (which break in nbconvert) and gives
us one canonical place to tune the baseline. Tune the baseline → every recipe's
comparison gets the new number automatically on the next execution pass.

The pipeline is deliberately minimal: fixed-window chunks, dense embeddings,
Qdrant in-memory, top-k cosine, stuffed-context generation. No reranker, no
query rewrite, no graph. The whole point is that it's the dumbest thing that
could work, so the technique under test has somewhere to win.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Sequence

from .chunkers import fixed_window
from .corpora import Document
from .providers import LLMClient
from .stores import QdrantBackend


PROMPT = (
    "Answer the question using only the passages below. "
    "If the passages do not contain the answer, say so plainly.\n\n"
    "Passages:\n{context}\n\nQuestion: {question}\nAnswer:"
)


@dataclass
class VanillaResult:
    """One vanilla-pipeline answer plus its retrieved context."""

    answer: str
    contexts: list[str]
    scores: list[float]


@lru_cache(maxsize=8)
def _build_index(
    corpus_key: str,
    target_tokens: int,
    overlap_tokens: int,
    embed_model: str | None,
    provider: str,
) -> tuple[QdrantBackend, LLMClient]:
    """Cache the index per (corpus, chunk params, model). One-time build cost."""
    from .corpora import (
        load_arxiv_mamba,
        load_rust_book,
        load_sec_10k,
        load_wikipedia_superconductors,
    )

    loaders: dict[str, Callable[[], Iterable[Document]]] = {
        "arxiv-mamba": load_arxiv_mamba,
        "wikipedia-superconductors": load_wikipedia_superconductors,
        "sec-10k-pltr": load_sec_10k,
        "rust-book": load_rust_book,
    }
    if corpus_key not in loaders:
        raise KeyError(
            f"Unknown corpus '{corpus_key}'. Choose one of {sorted(loaders)}."
        )

    docs = list(loaders[corpus_key]())
    chunks = fixed_window(
        docs, target_tokens=target_tokens, overlap_tokens=overlap_tokens
    )
    client = LLMClient(provider=provider) if embed_model is None else LLMClient(
        provider=provider, embed_model=embed_model
    )
    vectors = client.embed([c.text for c in chunks])
    store = QdrantBackend(
        f"baseline-{corpus_key}-{target_tokens}-{overlap_tokens}",
        dim=len(vectors[0]),
    )
    store.add(
        texts=[c.text for c in chunks],
        vectors=vectors,
        metadatas=[c.metadata for c in chunks],
        ids=[c.chunk_id for c in chunks],
    )
    return store, client


def vanilla_pipeline(
    question: str,
    *,
    corpus: str = "rust-book",
    top_k: int = 5,
    target_tokens: int = 384,
    overlap_tokens: int = 64,
    embed_model: str | None = None,
    provider: str = "nebius",
) -> VanillaResult:
    """The cookbook's canonical baseline. Build-or-reuse index, retrieve, answer.

    Parameters
    ----------
    question:
        Natural-language question.
    corpus:
        One of `arxiv-mamba`, `wikipedia-superconductors`, `sec-10k-pltr`, `rust-book`.
    top_k:
        Number of chunks to stuff into the prompt.
    target_tokens, overlap_tokens:
        Fixed-window parameters.

    Returns
    -------
    `VanillaResult(answer, contexts, scores)`.
    """
    store, client = _build_index(
        corpus, target_tokens, overlap_tokens, embed_model, provider
    )
    q_vec = client.embed([question])[0]
    hits = store.search(q_vec, top_k=top_k)
    contexts = [h.text for h in hits]
    scores = [float(h.score) for h in hits]
    answer = client.chat(
        PROMPT.format(context="\n\n".join(contexts), question=question)
    )
    return VanillaResult(answer=answer, contexts=contexts, scores=scores)


def compare_to_baseline(
    question: str,
    technique_fn: Callable[[str], tuple[str, Sequence[str]]],
    *,
    corpus: str = "rust-book",
    top_k: int = 5,
) -> dict[str, dict]:
    """Run both vanilla and `technique_fn`; return a dict for tabular display.

    `technique_fn(question)` must return `(answer: str, contexts: Sequence[str])`.
    Recipes wire their own `answer_question` in as `technique_fn`.
    """
    base = vanilla_pipeline(question, corpus=corpus, top_k=top_k)
    t_answer, t_contexts = technique_fn(question)
    return {
        "baseline": {
            "answer": base.answer,
            "n_contexts": len(base.contexts),
            "top_score": max(base.scores) if base.scores else 0.0,
        },
        "technique": {
            "answer": t_answer,
            "n_contexts": len(list(t_contexts)),
            "top_score": None,
        },
    }


__all__ = ["VanillaResult", "vanilla_pipeline", "compare_to_baseline"]
