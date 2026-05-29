"""Reranker primitives used by recipes 22–23.

Two interfaces:
    - `cross_encoder_rerank`: pointwise (query, chunk) scoring. The cookbook
      default uses an LLM scorer so the recipe runs with no model download.
      For production, swap in a real cross-encoder (`sentence-transformers`'s
      `CrossEncoder` with `BAAI/bge-reranker-v2-m3`, Cohere Rerank, etc.).
    - `listwise_llm_rerank`: ask the LLM to reorder a candidate list in one call.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Sequence

from .stores import Hit


POINTWISE_PROMPT = (
    "Score how well the passage answers the query, from 0.0 (irrelevant) to "
    "1.0 (directly answers). Reply with only the number.\n\n"
    "Query: {query}\n\nPassage: {text}\n\nScore:"
)


def cross_encoder_rerank(
    query: str,
    hits: Sequence[Hit],
    *,
    chat: Callable[[str], str] | None = None,
    top_k: int | None = None,
) -> list[Hit]:
    """Pointwise rerank: score each (query, hit.text) pair with the LLM judge.

    The cookbook default uses an LLM scorer to keep the recipe runnable without
    a model download. Production teams swap this for a real cross-encoder by
    calling `sentence-transformers.CrossEncoder` directly — the function
    signature stays the same, only the scorer changes.
    """
    if chat is None:
        from .providers import LLMClient

        chat = LLMClient().chat

    scored: list[tuple[Hit, float]] = []
    for h in hits:
        prompt = POINTWISE_PROMPT.format(query=query, text=h.text[:1000])
        raw = chat(prompt)
        m = re.search(r"[01](?:\.\d+)?", raw or "")
        score = float(m.group()) if m else 0.0
        scored.append((h, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        scored = scored[:top_k]
    return [
        Hit(doc_id=h.doc_id, text=h.text, score=score, metadata=h.metadata)
        for h, score in scored
    ]


def listwise_llm_rerank(
    query: str,
    hits: Sequence[Hit],
    chat: Callable[[str], str],
    *,
    top_k: int = 10,
) -> list[Hit]:
    """RankGPT-style listwise reorder via a single LLM call."""
    rendered = "\n".join(
        f"[{i}] {h.text[:400].replace(chr(10), ' ')}" for i, h in enumerate(hits)
    )
    prompt = (
        "You are a passage ranker. Reorder the passages below from most to "
        "least relevant for the query. Respond with a JSON object of the form "
        '{"order": [3, 1, 7, ...]} listing passage indices. No commentary.\n\n'
        f"Query: {query}\n\nPassages:\n{rendered}"
    )
    raw = chat(prompt)
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return list(hits[:top_k])
    try:
        parsed = json.loads(m.group())
        order = [int(i) for i in parsed.get("order", []) if 0 <= int(i) < len(hits)]
    except (json.JSONDecodeError, ValueError):
        return list(hits[:top_k])
    seen: set[int] = set()
    ordered: list[Hit] = []
    for i in order:
        if i in seen:
            continue
        seen.add(i)
        ordered.append(hits[i])
    for i, h in enumerate(hits):
        if i not in seen:
            ordered.append(h)
    return ordered[:top_k]


__all__ = ["cross_encoder_rerank", "listwise_llm_rerank"]
