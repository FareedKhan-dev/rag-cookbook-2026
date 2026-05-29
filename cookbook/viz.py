"""Tiny visualization helpers used in the "Discuss the Output" sections.

Kept intentionally small: a markdown-table renderer for hit lists, a
matplotlib bar comparator for metric sweeps, and a graphviz-style preview
for the graph recipes. Everything here is optional — notebooks render fine
without it.
"""
from __future__ import annotations

from typing import Sequence

from .stores import Hit


def hits_to_markdown(hits: Sequence[Hit], *, max_chars: int = 240) -> str:
    """Render a ranked hit list as a markdown table for the notebook output."""
    rows = ["| Rank | Score | Doc | Snippet |", "|---:|---:|---|---|"]
    for i, h in enumerate(hits, 1):
        snippet = h.text.replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 1] + "…"
        rows.append(f"| {i} | {h.score:.3f} | {h.doc_id} | {snippet} |")
    return "\n".join(rows)


def metric_bar(metrics: dict[str, float], *, title: str = "Metrics"):
    """Quick comparison bar chart — returns the matplotlib Figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, max(2, 0.4 * len(metrics))))
    names = list(metrics.keys())
    values = [metrics[n] for n in names]
    ax.barh(names, values)
    ax.set_xlim(0, max(values + [1.0]))
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def preview_graph(graph, *, max_edges: int = 60) -> str:
    """ASCII preview of the first N edges in a NetworkX graph."""
    lines: list[str] = []
    for i, (u, v, data) in enumerate(graph.edges(data=True)):
        if i >= max_edges:
            lines.append(f"... ({graph.number_of_edges() - max_edges} more edges)")
            break
        lines.append(f"{u} --[{data.get('predicate', '?')}]--> {v}")
    return "\n".join(lines)


__all__ = ["hits_to_markdown", "metric_bar", "preview_graph"]
