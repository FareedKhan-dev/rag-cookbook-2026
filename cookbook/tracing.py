"""Tracing setup — one call, every recipe.

Phoenix is the default because it runs locally and stays out of your way.
LangSmith is used inside the agentic recipes when graph state is interesting.
Both back-ends speak OpenTelemetry; the rest of the cookbook is unaware of
which one is on.
"""
from __future__ import annotations

import os
from typing import Literal

Backend = Literal["phoenix", "langsmith", "off"]


def init_tracing(
    backend: Backend | None = None,
    *,
    project: str = "rag-cookbook-2026",
) -> str:
    """Idempotent setup. Returns a one-line status string for display."""
    chosen = backend or os.getenv("COOKBOOK_TRACING", "phoenix")
    if chosen == "off":
        return "Tracing disabled."
    if chosen == "phoenix":
        return _init_phoenix(project)
    if chosen == "langsmith":
        return _init_langsmith(project)
    raise ValueError(f"Unknown tracing backend '{chosen}'")


def _init_phoenix(project: str) -> str:
    try:
        import phoenix as px
        from phoenix.otel import register
    except ImportError as e:  # pragma: no cover
        return f"Phoenix not installed ({e})."

    if not getattr(px, "_session", None):
        px.launch_app()
    register(project_name=project, auto_instrument=True)
    return f"Phoenix tracing on — project '{project}', UI at http://localhost:6006"


def _init_langsmith(project: str) -> str:
    if not os.getenv("LANGSMITH_API_KEY"):
        return "LANGSMITH_API_KEY missing — tracing disabled."
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ["LANGSMITH_PROJECT"] = project
    return f"LangSmith tracing on — project '{project}'"


__all__ = ["init_tracing"]
