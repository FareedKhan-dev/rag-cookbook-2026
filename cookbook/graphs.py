"""Knowledge-graph helpers for recipes 31–33.

Builds and queries small, in-memory knowledge graphs from chunked corpora:
LLM-extracted triples → NetworkX graph → community detection → PageRank
walks. Recipes that need a "real" KG store (Neo4j, Memgraph) extend these
helpers; the defaults work on a laptop.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import networkx as nx


TRIPLE_PROMPT = (
    "Extract factual triples from the text. Each triple is "
    '`{{"subject": ..., "predicate": ..., "object": ...}}`. '
    "Prefer concise canonical noun phrases. Skip opinions and questions. "
    'Respond with `{{"triples": [...]}}` and nothing else.\n\nText:\n{text}'
)


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    source_id: str


def extract_triples(
    texts: Sequence[tuple[str, str]],
    chat: Callable[[str], str],
) -> list[Triple]:
    """Extract triples from `(source_id, text)` pairs."""
    triples: list[Triple] = []
    for source_id, text in texts:
        raw = chat(TRIPLE_PROMPT.format(text=text[:4000]))
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            continue
        try:
            parsed = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        for t in parsed.get("triples", []):
            subj = (t.get("subject") or "").strip().lower()
            pred = (t.get("predicate") or "").strip().lower()
            obj = (t.get("object") or "").strip().lower()
            if subj and pred and obj:
                triples.append(Triple(subj, pred, obj, source_id))
    return triples


def build_graph(triples: Iterable[Triple]) -> nx.MultiDiGraph:
    """Turn a list of triples into a multi-edge directed graph."""
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    for t in triples:
        g.add_node(t.subject)
        g.add_node(t.object)
        g.add_edge(t.subject, t.object, predicate=t.predicate, source=t.source_id)
    return g


def community_summaries(
    g: nx.MultiDiGraph,
    chat: Callable[[str], str],
    *,
    max_communities: int = 12,
) -> dict[int, str]:
    """Leiden-style communities; LLM-summarized. Falls back to greedy modularity."""
    undirected = g.to_undirected()
    try:
        import igraph as ig

        nodes = list(undirected.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for u, v in undirected.edges()]
        graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
        partition = graph.community_leiden(objective_function="modularity")
        clusters = [[nodes[i] for i in cluster] for cluster in partition]
    except Exception:
        clusters = [list(c) for c in nx.algorithms.community.greedy_modularity_communities(undirected)]
    clusters = sorted(clusters, key=len, reverse=True)[:max_communities]

    summaries: dict[int, str] = {}
    for i, members in enumerate(clusters):
        sample_edges = []
        for u in members[:30]:
            for v in undirected.neighbors(u):
                if v in members:
                    for _, data in g.get_edge_data(u, v, default={}).items():
                        sample_edges.append(f"{u} —{data.get('predicate', '?')}→ {v}")
                        break
                if len(sample_edges) >= 40:
                    break
            if len(sample_edges) >= 40:
                break
        prompt = (
            "Summarize this knowledge-graph community in 2-3 sentences. "
            "Focus on the dominant theme.\n\n" + "\n".join(sample_edges)
        )
        summaries[i] = chat(prompt).strip()
    return summaries


def personalized_pagerank(
    g: nx.MultiDiGraph,
    seeds: Sequence[str],
    *,
    alpha: float = 0.5,
) -> dict[str, float]:
    """HippoRAG-style memory walk seeded by query entities."""
    nodes = list(g.nodes())
    if not nodes:
        return {}
    seed_set = {s.lower() for s in seeds}
    personalization = {n: (1.0 if n in seed_set else 0.0) for n in nodes}
    if sum(personalization.values()) == 0:
        personalization = {n: 1.0 / len(nodes) for n in nodes}
    try:
        return nx.pagerank(g.to_undirected(), alpha=alpha, personalization=personalization)
    except nx.PowerIterationFailedConvergence:
        return {n: 1.0 / len(nodes) for n in nodes}


__all__ = [
    "Triple",
    "extract_triples",
    "build_graph",
    "community_summaries",
    "personalized_pagerank",
]
