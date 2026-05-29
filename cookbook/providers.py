"""Provider abstraction for chat, embedding, and reranking calls.

The whole cookbook flows through `LLMClient`. Each recipe constructs one in
its setup cell, the rest of the recipe never touches a vendor SDK directly.
This is what lets every notebook be Nebius-by-default and OpenAI-or-Anthropic-
or-Groq-by-one-line-change.

Design notes:
    - Chat is routed through LiteLLM, which speaks every OpenAI-compatible
      backend (Nebius, OpenAI, Groq, OpenRouter, Together, Anthropic, …).
    - Embedding has two paths: a hosted path for Nebius / OpenAI / Voyage /
      Cohere / Jina, and a local path that wraps `sentence-transformers` for
      BGE-M3, Qwen3-Embedding, and nomic-embed.
    - Reranking is similarly hybrid: hosted via Cohere / Voyage / Jina,
      local via `FlagEmbedding` and `sentence-transformers` cross-encoders.

This module is intentionally tiny — under ~250 lines — so a reader can audit
the whole provider surface in one sitting.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Literal, Sequence

from dotenv import load_dotenv

from . import _cache

load_dotenv(override=False)

Provider = Literal[
    "nebius", "openai", "anthropic", "groq", "openrouter", "together", "local"
]


# ---------------------------------------------------------------------------
# Provider routing table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProviderSpec:
    """Routing metadata for a single LLM backend."""

    litellm_prefix: str
    base_url_env: str | None
    api_key_env: str
    default_chat_model: str
    default_embed_model: str | None


PROVIDERS: dict[str, ProviderSpec] = {
    "nebius": ProviderSpec(
        litellm_prefix="openai",
        base_url_env="NEBIUS_BASE_URL",
        api_key_env="NEBIUS_API_KEY",
        default_chat_model="meta-llama/Llama-3.3-70B-Instruct",
        default_embed_model="Qwen/Qwen3-Embedding-8B",
    ),
    "openai": ProviderSpec(
        litellm_prefix="openai",
        base_url_env=None,
        api_key_env="OPENAI_API_KEY",
        default_chat_model="gpt-4o-mini",
        default_embed_model="text-embedding-3-large",
    ),
    "anthropic": ProviderSpec(
        litellm_prefix="anthropic",
        base_url_env=None,
        api_key_env="ANTHROPIC_API_KEY",
        default_chat_model="claude-haiku-4-5",
        default_embed_model=None,
    ),
    "groq": ProviderSpec(
        litellm_prefix="groq",
        base_url_env=None,
        api_key_env="GROQ_API_KEY",
        default_chat_model="llama-3.3-70b-versatile",
        default_embed_model=None,
    ),
    "openrouter": ProviderSpec(
        litellm_prefix="openrouter",
        base_url_env=None,
        api_key_env="OPENROUTER_API_KEY",
        default_chat_model="meta-llama/llama-3.3-70b-instruct",
        default_embed_model=None,
    ),
    "together": ProviderSpec(
        litellm_prefix="together_ai",
        base_url_env=None,
        api_key_env="TOGETHER_API_KEY",
        default_chat_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        default_embed_model="BAAI/bge-large-en-v1.5",
    ),
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
@dataclass
class LLMClient:
    """One client to rule them all.

    Parameters
    ----------
    provider:
        Backend key (one of `PROVIDERS`). Falls back to `$PROVIDER`, then nebius.
    chat_model, embed_model, rerank_model:
        Optional overrides for the default model on the chosen provider.
    extra_headers:
        Forwarded on every call (useful for OpenRouter `HTTP-Referer`).
    """

    provider: Provider = field(default_factory=lambda: os.getenv("PROVIDER", "nebius"))
    chat_model: str | None = None
    embed_model: str | None = None
    rerank_model: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unknown provider '{self.provider}'. Options: {list(PROVIDERS)}"
            )
        spec = PROVIDERS[self.provider]
        self._spec = spec
        self.chat_model = self.chat_model or spec.default_chat_model
        self.embed_model = self.embed_model or spec.default_embed_model

    # -- chat ---------------------------------------------------------------
    def chat(
        self,
        prompt: str | list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Single-turn chat. Returns the assistant string."""
        if isinstance(prompt, str):
            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = prompt

        cache_key = {
            "op": "chat",
            "provider": self.provider,
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        cached = _cache.get(cache_key)
        if cached is not None:
            return str(cached)

        import litellm

        route = f"{self._spec.litellm_prefix}/{self.chat_model}"
        kwargs: dict[str, Any] = {
            "model": route,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": os.getenv(self._spec.api_key_env),
        }
        if self._spec.base_url_env:
            kwargs["api_base"] = os.getenv(self._spec.base_url_env)
        if response_format:
            kwargs["response_format"] = response_format
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        resp = litellm.completion(**kwargs)
        text = resp.choices[0].message.content or ""
        _cache.put(cache_key, text)
        return text

    # -- embeddings ---------------------------------------------------------
    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> list[list[float]]:
        """Embed a sequence of texts; returns one float vector per text."""
        if self.provider == "local":
            return self._embed_local(texts)
        if self.embed_model is None:
            raise RuntimeError(
                f"Provider '{self.provider}' has no default embedding model. "
                "Pass `embed_model=` explicitly or use provider='local'."
            )

        out: list[list[float]] = []
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        for i, t in enumerate(texts):
            cached = _cache.get(
                {"op": "embed", "provider": self.provider, "model": self.embed_model, "text": t}
            )
            if cached is not None:
                out.append(list(cached))
            else:
                out.append([])  # placeholder
                missing_idx.append(i)
                missing_texts.append(t)

        if not missing_texts:
            return out

        import litellm

        live: list[list[float]] = []
        for start in range(0, len(missing_texts), batch_size):
            chunk = missing_texts[start : start + batch_size]
            kwargs: dict[str, Any] = {
                "model": f"{self._spec.litellm_prefix}/{self.embed_model}",
                "input": chunk,
                "api_key": os.getenv(self._spec.api_key_env),
            }
            if self._spec.base_url_env:
                kwargs["api_base"] = os.getenv(self._spec.base_url_env)
            resp = litellm.embedding(**kwargs)
            live.extend(d["embedding"] for d in resp["data"])

        for idx, t, vec in zip(missing_idx, missing_texts, live):
            out[idx] = list(vec)
            _cache.put(
                {"op": "embed", "provider": self.provider, "model": self.embed_model, "text": t},
                vec,
            )
        return out

    def _embed_local(self, texts: Sequence[str]) -> list[list[float]]:
        model = _local_encoder(self.embed_model or "BAAI/bge-m3")
        return model.encode(list(texts), normalize_embeddings=True).tolist()

    # -- reranking ----------------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank `documents` against `query`; returns (index, score) pairs."""
        if self.provider == "local" or self.rerank_model is None or "/" in (
            self.rerank_model or ""
        ):
            return _local_rerank(
                query,
                documents,
                model_name=self.rerank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_k=top_k,
            )
        # Hosted Cohere/Voyage/Jina paths go through their own SDKs; left as a
        # focused exercise in the reranking recipes.
        raise NotImplementedError(
            "Hosted rerankers are exercised in recipe 22 with their native SDKs."
        )


# ---------------------------------------------------------------------------
# Local model caches
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _local_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def _local_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _local_rerank(
    query: str,
    documents: Sequence[str],
    *,
    model_name: str,
    top_k: int | None,
) -> list[tuple[int, float]]:
    ce = _local_cross_encoder(model_name)
    scores = ce.predict([(query, d) for d in documents])
    ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


def list_providers() -> Iterable[str]:
    """Helper for the cookbook-tour recipe."""
    return PROVIDERS.keys()
