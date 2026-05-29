"""Chunking primitives.

Recipes 4–12 each show *one* chunking strategy in detail; this module collects
the small reusable helpers so the recipes don't need to re-implement token
counting, overlap math, and breakpoint heuristics in every notebook.

Available strategies:
    - `fixed_window`: classic char or token sliding window
    - `sentence_window`: sentence-tokenize then group with overlap
    - `semantic_split`: split at cosine-distance breakpoints (recipe 5)
    - `proposition_split`: LLM rewrites text into atomic propositions (recipe 8)
    - `parent_child`: small chunks for retrieval, big chunks for context
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .corpora import Document


@dataclass(frozen=True)
class Chunk:
    """One indexable unit of text."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict


def _count_tokens(text: str) -> int:
    """Cheap word-count proxy. Recipes that need real tokens use tiktoken."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Fixed window
# ---------------------------------------------------------------------------
def fixed_window(
    documents: Sequence[Document],
    *,
    target_tokens: int = 384,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Sliding window over whitespace-tokenized text."""
    chunks: list[Chunk] = []
    for doc in documents:
        words = doc.text.split()
        step = max(1, target_tokens - overlap_tokens)
        for i, start in enumerate(range(0, len(words), step)):
            piece = " ".join(words[start : start + target_tokens])
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#fw{i:04d}",
                    doc_id=doc.doc_id,
                    text=piece,
                    metadata={**doc.metadata, "strategy": "fixed_window", "window": i},
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Sentence-window
# ---------------------------------------------------------------------------
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'`])")


def sentence_window(
    documents: Sequence[Document],
    *,
    sentences_per_chunk: int = 5,
    overlap: int = 1,
) -> list[Chunk]:
    """Group sentences with a small overlap."""
    chunks: list[Chunk] = []
    for doc in documents:
        sentences = _SENT_BOUNDARY.split(doc.text)
        step = max(1, sentences_per_chunk - overlap)
        for i, start in enumerate(range(0, len(sentences), step)):
            piece = " ".join(sentences[start : start + sentences_per_chunk]).strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#sw{i:04d}",
                    doc_id=doc.doc_id,
                    text=piece,
                    metadata={**doc.metadata, "strategy": "sentence_window"},
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Semantic split (recipe 5)
# ---------------------------------------------------------------------------
def semantic_split(
    documents: Sequence[Document],
    embed: Callable[[Sequence[str]], list[list[float]]],
    *,
    breakpoint_percentile: float = 90.0,
    min_sentences: int = 2,
) -> list[Chunk]:
    """Cut at sentence-pair cosine-distance breakpoints above a percentile."""
    import numpy as np

    chunks: list[Chunk] = []
    for doc in documents:
        sentences = [s.strip() for s in _SENT_BOUNDARY.split(doc.text) if s.strip()]
        if len(sentences) <= min_sentences:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#ss0000",
                    doc_id=doc.doc_id,
                    text=" ".join(sentences),
                    metadata={**doc.metadata, "strategy": "semantic_split"},
                )
            )
            continue
        vecs = np.asarray(embed(sentences), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        dists = 1.0 - (vecs[:-1] * vecs[1:]).sum(axis=1)
        threshold = float(np.percentile(dists, breakpoint_percentile))
        cuts = [0] + [i + 1 for i, d in enumerate(dists) if d >= threshold] + [len(sentences)]
        for i in range(len(cuts) - 1):
            piece = " ".join(sentences[cuts[i] : cuts[i + 1]]).strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#ss{i:04d}",
                    doc_id=doc.doc_id,
                    text=piece,
                    metadata={**doc.metadata, "strategy": "semantic_split"},
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Proposition decomposition (recipe 8)
# ---------------------------------------------------------------------------
PROPOSITION_PROMPT = (
    "Rewrite the passage below into a numbered list of atomic, self-contained "
    "factual statements. Each statement must stand alone without referring to "
    "the original passage. Resolve all pronouns. Do not add facts that are not "
    "in the passage.\n\nPassage:\n{passage}\n\nNumbered list:"
)


def proposition_split(
    documents: Sequence[Document],
    chat: Callable[[str], str],
    *,
    window_tokens: int = 800,
) -> list[Chunk]:
    """Convert text into atomic propositions via an LLM."""
    chunks: list[Chunk] = []
    for doc in documents:
        words = doc.text.split()
        windows = [
            " ".join(words[i : i + window_tokens])
            for i in range(0, len(words), window_tokens)
        ]
        for w_idx, window in enumerate(windows):
            raw = chat(PROPOSITION_PROMPT.format(passage=window))
            for p_idx, line in enumerate(raw.splitlines()):
                cleaned = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
                if not cleaned:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#prop{w_idx:03d}.{p_idx:03d}",
                        doc_id=doc.doc_id,
                        text=cleaned,
                        metadata={**doc.metadata, "strategy": "proposition"},
                    )
                )
    return chunks


# ---------------------------------------------------------------------------
# Parent / child (recipe 9)
# ---------------------------------------------------------------------------
def parent_child(
    documents: Sequence[Document],
    *,
    parent_tokens: int = 1024,
    child_tokens: int = 192,
    overlap: int = 32,
) -> tuple[list[Chunk], list[Chunk], dict[str, str]]:
    """Build aligned small-chunks-for-search and big-chunks-for-context.

    Returns (children, parents, child_to_parent).
    """
    parents = fixed_window(documents, target_tokens=parent_tokens, overlap_tokens=0)
    children: list[Chunk] = []
    mapping: dict[str, str] = {}
    for p in parents:
        words = p.text.split()
        step = max(1, child_tokens - overlap)
        for i, start in enumerate(range(0, len(words), step)):
            piece = " ".join(words[start : start + child_tokens])
            if not piece.strip():
                continue
            cid = f"{p.chunk_id}#c{i:03d}"
            children.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=p.doc_id,
                    text=piece,
                    metadata={**p.metadata, "parent_id": p.chunk_id},
                )
            )
            mapping[cid] = p.chunk_id
    return children, parents, mapping


__all__ = [
    "Chunk",
    "fixed_window",
    "sentence_window",
    "semantic_split",
    "proposition_split",
    "parent_child",
]
