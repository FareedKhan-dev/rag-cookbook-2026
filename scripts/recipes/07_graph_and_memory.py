"""Category 7 — Graph & Memory."""
from __future__ import annotations

from authoring import CodeStep, Recipe, Reference


# =============================================================================
# Microsoft GraphRAG Pipeline
# =============================================================================
MSGRAPH = Recipe(
    path="recipes/07-graph-and-memory/microsoft-graphrag-pipeline.ipynb",
    title="GraphRAG — Knowledge Graph + Community Summaries",
    category="graph-and-memory",
    corpus_filter="wikipedia-superconductors",

    theory_problem=[
        "Vector retrieval is excellent at finding *individual* passages that match a query. It is structurally bad at answering questions that require summarising the whole corpus — \"what are the main themes here?\", \"what entities recur across documents?\", \"what's the overall structure of this knowledge?\" The answer to those questions is not in any single chunk; it lives in the graph of relationships *between* chunks.",
        "GraphRAG (Microsoft Research, April 2024) builds an explicit knowledge graph from the corpus. Triples (subject, predicate, object) are extracted from each chunk. The graph is partitioned into communities. Each community gets an LLM-written summary. At query time, the system routes between *local* search (entity-anchored) and *global* search (community-summary based). The result is dramatically better answers on corpus-spanning questions.",
    ],
    theory_origin=[
        "Microsoft Research published GraphRAG in April 2024 along with an open-source implementation. The paper showed that on broad questions over the Russian-Ukraine-war news corpus, GraphRAG beat vanilla RAG decisively — answer quality measured by LLM-as-judge improved by 70-85% on global questions. The community-summary approach was the key contribution; the rest of the pipeline (triple extraction, Leiden clustering) was standard graph machinery.",
        "By 2026 GraphRAG has fragmented into many variants — LightRAG (Recipe 32), HippoRAG (Recipe 33), and others. The original Microsoft implementation remains the most influential, and the community-summary pattern is the most-imitated piece.",
    ],
    theory_landscape=[
        "Graph-based RAG has settled into a family of variants by 2026:",
        "",
        "- **GraphRAG (this recipe).** Communities + global/local routing. Best for broad questions.",
        "- **LightRAG (Recipe 32).** Dual-level retrieval on the same KG. Cheaper, incremental.",
        "- **HippoRAG (Recipe 33).** Personalised PageRank over the KG. Multi-hop strength.",
        "- **MemoRAG / Mem0 (Recipe 34).** Memory-as-graph for long conversations.",
        "",
        "Pick by query shape: broad questions want GraphRAG, multi-hop wants HippoRAG, fast incremental wants LightRAG, conversational memory wants Mem0.",
    ],
    theory_when_to_use=[
        "Use GraphRAG on corpora where users ask broad questions that span many documents. Research synthesis, internal wikis where questions like \"what's our overall position on X?\" are common, exploratory data analysis, and anywhere users need to *understand* the corpus before they query it.",
        "Skip it for narrow factoid queries. Vanilla RAG is faster and cheaper for those.",
        "Skip it when the corpus has no relational structure. A pile of independent FAQ answers has no graph to build.",
        "Skip it when budget matters more than quality. GraphRAG is expensive — triple extraction is one LLM call per chunk, community summaries are more.",
    ],
    theory_intuition=[
        "Four intuitions to carry:",
        "",
        "**Triples are the atoms.** Every fact in your corpus becomes a (subject, predicate, object) triple. The graph is the assembly of triples; each chunk contributes a handful of edges.",
        "",
        "**Communities are topics.** Graph community detection groups densely-connected entities. These groups correspond roughly to topical clusters — the LLM-written summaries are essentially topic summaries.",
        "",
        "**Global vs local is a routing decision.** A question naming a specific entity goes local (find that entity, walk neighbours). A question asking about themes goes global (read community summaries, synthesise).",
        "",
        "**The graph is an artifact.** Building it is a one-time cost. Once built, queries are fast and cheap — you're searching summaries and walking edges, not embedding-and-searching billions of chunks.",
    ],

    architecture_mermaid="""
flowchart TB
  D[Documents] --> C[Chunks]
  C --> T[LLM: extract<br/>triples]
  T --> G[(Knowledge Graph)]
  G --> CD[Leiden<br/>community detection]
  CD --> SUM[LLM: summarise<br/>each community]
  SUM --> CI[(Community<br/>summaries)]
  Q[Query] --> R{Local or<br/>Global?}
  R -->|local| LOC[Walk graph<br/>from entities]
  R -->|global| GLO[Search community<br/>summaries]
  G --> LOC
  CI --> GLO
""",

    references=[
        Reference(
            title="From Local to Global — A Graph RAG Approach to Query-Focused Summarization (Edge et al., 2024)",
            url="https://arxiv.org/abs/2404.16130",
            kind="paper",
            note="The Microsoft GraphRAG paper.",
        ),
        Reference(
            title="microsoft/graphrag reference implementation",
            url="https://github.com/microsoft/graphrag",
            kind="repo",
            note="The official open-source release.",
        ),
        Reference(
            title="Leiden community detection algorithm",
            url="https://www.nature.com/articles/s41598-019-41695-z",
            kind="paper",
            note="The clustering algorithm we use.",
        ),
        Reference(
            title="LightRAG (Recipe 32)",
            url="https://arxiv.org/abs/2410.05779",
            kind="paper",
            note="A cheaper graph-RAG variant.",
        ),
        Reference(
            title="HippoRAG (Recipe 33)",
            url="https://arxiv.org/abs/2405.14831",
            kind="paper",
            note="PageRank-based graph traversal alternative.",
        ),
        Reference(
            title="NetworkX documentation",
            url="https://networkx.org/documentation/stable/",
            kind="docs",
            note="The Python graph library we use.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load and chunk the corpus",
                "",
                "Wikipedia superconductors has good relational structure — entities, theories, scientists, materials — perfect for building a knowledge graph.",
            ],
            code=[
                "from cookbook.corpora import load_wikipedia_superconductors",
                "from cookbook.chunkers import sentence_window",
                "",
                "docs = list(load_wikipedia_superconductors())[:12]",
                "chunks = sentence_window(docs, sentences_per_chunk=4)[:60]",
                "print(f'Working with {len(chunks)} chunks across {len(docs)} articles.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Extract triples from each chunk",
                "",
                "We use `cookbook.graphs.extract_triples`. It prompts the LLM to return canonical (subject, predicate, object) triples and parses the JSON.",
            ],
            code=[
                "from cookbook.graphs import extract_triples",
                "",
                "triples = extract_triples([(c.chunk_id, c.text) for c in chunks], client.chat)",
                "print(f'Extracted {len(triples)} triples.')",
                "print()",
                "print('Sample triples:')",
                "for t in triples[:5]:",
                "    print(f'  ({t.subject}, {t.predicate}, {t.object})  [from {t.source_id}]')",
            ],
        ),
        CodeStep(
            tag="index",
            lead_md=[
                "### Step 3 — Build the NetworkX graph",
                "",
                "Edges carry predicate labels and source chunk IDs. The same entities can connect via many edges; we use a MultiDiGraph to keep them all.",
            ],
            code=[
                "from cookbook.graphs import build_graph",
                "",
                "g = build_graph(triples)",
                "print(f'Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.')",
                "",
                "from cookbook.viz import preview_graph",
                "print(preview_graph(g, max_edges=10))",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Detect communities and summarise each",
                "",
                "`cookbook.graphs.community_summaries` runs Leiden clustering (or falls back to greedy modularity) and asks the LLM to summarise each community.",
            ],
            code=[
                "from cookbook.graphs import community_summaries",
                "",
                "summaries = community_summaries(g, client.chat, max_communities=6)",
                "print(f'Found {len(summaries)} communities.')",
                "print()",
                "for i, s in summaries.items():",
                "    print(f'[community {i}]')",
                "    print(f'  {s[:240]}')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Build the global-search function",
                "",
                "Given a query, ask the LLM to answer using only the community summaries. This is the \"global\" GraphRAG path.",
            ],
            code=[
                "def graph_rag_global(question: str) -> tuple[str, list[str]]:",
                "    rendered = '\\n'.join(f'Community {i}: {s}' for i, s in summaries.items())",
                "    answer = client.chat(",
                "        'Use these community summaries from a knowledge graph built over Wikipedia '",
                "        'articles about superconductivity. Answer the question synthesising across communities.\\n'",
                "        + rendered + f'\\n\\nQuestion: {question}\\nAnswer:'",
                "    )",
                "    return answer, list(summaries.values())",
                "",
                "ans, _ = graph_rag_global('Across these articles, what are the major eras of superconductivity research?')",
                "print(ans[:400])",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 6 — Wrap as `answer_question`",
                "",
                "Cookbook contract. We use the global path for the default contract.",
            ],
            code=[
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    return graph_rag_global(question)",
                "",
                "ans, _ = answer_question('What are the major themes of superconductivity research?')",
                "print(ans[:400])",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — entity degree distribution",
                "",
                "Which entities are most central in the graph? High-degree entities are the hubs — often the names of theories, materials, or canonical phenomena.",
            ],
            code=[
                "from collections import Counter",
                "degrees = Counter(dict(g.degree()))",
                "for entity, deg in degrees.most_common(10):",
                "    print(f'  {deg:3d}  {entity}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what are the community sizes?",
                "",
                "Are communities balanced or skewed? A few giant communities and many tiny ones means clustering is struggling.",
            ],
            code=[
                "import matplotlib.pyplot as plt",
                "sizes = [len(s) for s in summaries.values()]",
                "fig, ax = plt.subplots(figsize=(6, 2.5))",
                "ax.bar(range(len(sizes)), [len(s.split()) for s in summaries.values()])",
                "ax.set_xlabel('Community index')",
                "ax.set_ylabel('Summary length (words)')",
                "ax.set_title('Community summary sizes')",
                "plt.tight_layout()",
                "plt.show()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — read one community summary in full",
                "",
                "Quality of the summaries is the technique. Read one in full to see whether it captures a coherent topic.",
            ],
            code=[
                "first_idx, first_summary = next(iter(summaries.items()))",
                "print(f'Community {first_idx}:')",
                "print(first_summary)",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost",
                "",
                "GraphRAG is expensive. Count the LLM calls: triple extraction (one per chunk), community summaries (one per community).",
            ],
            code=[
                "print(f'Triple extraction calls: ~{len(chunks)} (one per chunk)')",
                "print(f'Community summary calls: ~{len(summaries)} (one per community)')",
                "print(f'Total LLM calls to build the graph: ~{len(chunks) + len(summaries)}')",
                "print()",
                "print('Query-time cost: 1 LLM call per question (global path).')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a synthesis question.",
    ],
    run_code=[
        "ans, _ = answer_question('What are the major eras of superconductivity research and what discoveries marked each?')",
        "print('=== GraphRAG answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla baseline vs GraphRAG on a broad synthesis question. Vanilla retrieves chunks; GraphRAG synthesises across communities. The shape of the answers should be visibly different.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What are the major eras of superconductivity research?'",
        "base = vanilla_pipeline(q, corpus='wikipedia-superconductors', top_k=5)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:200]},",
        "    {'pipeline': 'graphrag', 'preview': ours_a[:200]},",
        "])",
    ],

    tuning_md=[
        "Six knobs in priority order to consider:",
        "",
        "1. **Triple-extraction prompt.** The quality of triples sets the ceiling for everything downstream. Tune carefully.",
        "2. **Community size cap.** Too few communities lose nuance; too many overwhelm the global synthesis.",
        "3. **Summary length.** Each community summary is one LLM call; longer summaries cost more but carry more signal.",
        "4. **Clustering algorithm.** Leiden is preferred; greedy modularity is a fallback. The library handles both.",
        "5. **Local vs global routing.** Add a router upstream that picks based on whether the query names a specific entity.",
        "6. **Incremental updates.** GraphRAG was originally batch-only. LightRAG (Recipe 32) handles incremental updates more elegantly.",
    ],

    discussion_md=[
        "Four failure modes you'll meet:",
        "",
        "- **Triple noise.** The extraction LLM occasionally hallucinates triples. Validate against the source chunk.",
        "- **Community fragmentation.** Small datasets produce many tiny communities. Increase the chunk window or merge small communities post hoc.",
        "- **Summary drift.** Community summaries can over-generalise. The synthesis prompt should ask the LLM to stay grounded in the summaries.",
        "- **Cost explosion at scale.** GraphRAG over millions of chunks is genuinely expensive. Consider LightRAG (Recipe 32) when budget matters.",
        "",
        "Compose with semantic chunking (Recipe 5) for cleaner triple extraction. Compose with reranking (Recipe 22) for the local-search path.",
    ],
)


# =============================================================================
# LightRAG Dual-Level
# =============================================================================
LIGHTRAG = Recipe(
    path="recipes/07-graph-and-memory/lightrag-dual-level.ipynb",
    title="LightRAG — Dual-Level Graph Retrieval",
    category="graph-and-memory",
    corpus_filter="wikipedia-superconductors",

    theory_problem=[
        "GraphRAG (Recipe 31) is powerful but expensive: triple extraction, full graph construction, community detection, and per-community summaries are all paid up-front. For large or incrementally-updated corpora, those costs compound. LightRAG (HKUDS, Oct 2024) keeps the same knowledge-graph idea but rearranges it to be roughly 10x cheaper.",
        "LightRAG's trick: build the graph once, but expose two indexed views over it — one over entity descriptions, one over relation descriptions. Queries retrieve from both views. Low-level (entity) retrieval handles factual queries; high-level (relation) retrieval handles synthesis. No community detection, no community summaries, no per-question synthesis call.",
    ],
    theory_origin=[
        "LightRAG was published by the HKU DS lab in October 2024. The paper showed comparable quality to GraphRAG at roughly 10x lower indexing cost and 5-10x lower query-time cost. The dual-level abstraction was the key innovation: two views over the same graph, retrieved independently, concatenated at query time. The paper also showed strong incremental-update performance — adding new documents touches only the affected entities, not the whole graph.",
        "By 2026 LightRAG has become the practical default for graph-RAG deployments where cost matters. The cookbook implementation follows the paper's structure but uses simpler retrieval (cosine over indexed descriptions) instead of the paper's custom dual-cross-attention setup.",
    ],
    theory_landscape=[
        "Within the graph-RAG family:",
        "",
        "- **GraphRAG (Recipe 31).** Full community detection + summaries. Best quality on global questions, highest cost.",
        "- **LightRAG (this recipe).** Dual-level views, no communities. ~10x cheaper, comparable quality.",
        "- **HippoRAG (Recipe 33).** PageRank over the KG. Best for multi-hop chains.",
        "",
        "Pick by cost budget: GraphRAG when quality matters more than cost; LightRAG for most production cases; HippoRAG for multi-hop-heavy workloads.",
    ],
    theory_when_to_use=[
        "Use LightRAG when you want graph-RAG quality without GraphRAG cost. Most production deployments fit here — the dual-level retrieval handles a wide mix of query types competently.",
        "Skip it when global synthesis quality matters more than cost. Use GraphRAG for that.",
        "Skip it when multi-hop reasoning dominates. Use HippoRAG (Recipe 33) for that.",
        "Skip it for purely factoid queries. Vanilla RAG is faster and the graph machinery adds no value.",
    ],
    theory_intuition=[
        "Five intuitions to carry:",
        "",
        "**Two views from one graph.** The graph is built once; two text-view indexes (entity descriptions and relation descriptions) are derived from it. Both index the same underlying knowledge in different shapes.",
        "",
        "**Low-level vs high-level.** Low-level retrieval (entity descriptions) catches factual queries — \"What is YBCO?\". High-level retrieval (relation descriptions) catches synthesis queries — \"How are these things connected?\".",
        "",
        "**Both retrievals run per query.** Each query embeds and searches against both views. Results are concatenated, not fused — both kinds of context inform the answer simultaneously.",
        "",
        "**No per-question synthesis call.** Unlike GraphRAG's global path, LightRAG doesn't synthesise community summaries on every query. Cost stays low because the expensive work happens at index time, not at query time.",
        "",
        "**Incremental updates are cheap.** Adding new documents only adds new entity and relation views to the indexes; no community detection to redo. This is the structural advantage over GraphRAG.",
    ],

    architecture_mermaid="""
flowchart LR
  D[Documents] --> T[Extract triples]
  T --> G[(Knowledge Graph)]
  G --> EV[Entity views<br/>description per node]
  G --> RV[Relation views<br/>description per edge]
  EV --> ES[(Entity index)]
  RV --> RS[(Relation index)]
  Q[Query] --> R1[Search entity index]
  Q --> R2[Search relation index]
  ES --> R1
  RS --> R2
  R1 --> G2[LLM answer]
  R2 --> G2
""",

    references=[
        Reference(
            title="LightRAG — Simple and Fast Retrieval-Augmented Generation",
            url="https://arxiv.org/abs/2410.05779",
            kind="paper",
            note="The HKUDS paper.",
        ),
        Reference(
            title="HKUDS/LightRAG official repository",
            url="https://github.com/HKUDS/LightRAG",
            kind="repo",
            note="Reference implementation.",
        ),
        Reference(
            title="GraphRAG (Recipe 31)",
            url="https://arxiv.org/abs/2404.16130",
            kind="paper",
            note="The full-cost cousin.",
        ),
        Reference(
            title="HippoRAG (Recipe 33)",
            url="https://arxiv.org/abs/2405.14831",
            kind="paper",
            note="PageRank-based variant.",
        ),
        Reference(
            title="NetworkX documentation",
            url="https://networkx.org/documentation/stable/",
            kind="docs",
            note="Our graph library.",
        ),
        Reference(
            title="HKU Data Science Lab blog",
            url="https://github.com/HKUDS",
            kind="repo",
            note="More from the LightRAG authors.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load + chunk + extract triples",
                "",
                "Same opening as GraphRAG (Recipe 31). The graph construction is shared; the divergence is what we do with it.",
            ],
            code=[
                "from cookbook.corpora import load_wikipedia_superconductors",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.graphs import extract_triples, build_graph",
                "",
                "docs = list(load_wikipedia_superconductors())[:14]",
                "chunks = sentence_window(docs, sentences_per_chunk=4)[:80]",
                "triples = extract_triples([(c.chunk_id, c.text) for c in chunks], client.chat)",
                "g = build_graph(triples)",
                "print(f'Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.')",
            ],
        ),
        CodeStep(
            tag="index",
            lead_md=[
                "### Step 2 — Build entity-level views",
                "",
                "For each node, describe it by listing its top neighbours. This is the low-level retrieval surface.",
            ],
            code=[
                "entity_views = {",
                "    n: f'Entity: {n}. Connected to: ' + ', '.join(sorted(set(g.successors(n)))[:8])",
                "    for n in g.nodes()",
                "}",
                "print(f'Built {len(entity_views)} entity views.')",
                "print()",
                "print('Sample entity views:')",
                "for n, v in list(entity_views.items())[:3]:",
                "    print(f'  {v}')",
            ],
        ),
        CodeStep(
            tag="index",
            lead_md=[
                "### Step 3 — Build relation-level views",
                "",
                "For each edge, describe it as a single sentence \"subject predicate object\". This is the high-level retrieval surface.",
            ],
            code=[
                "edge_views = []",
                "for u, v, data in g.edges(data=True):",
                "    edge_views.append(f'{u} {data.get(\"predicate\", \"relates to\")} {v}')",
                "print(f'Built {len(edge_views)} edge views.')",
                "print()",
                "print('Sample edge views:')",
                "for ev in edge_views[:6]:",
                "    print(f'  {ev}')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 4 — Index both view types",
                "",
                "Standard Qdrant indexes, one per view.",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "",
                "ent_v = client.embed(list(entity_views.values()))",
                "ent_store = QdrantBackend('lr-ent', dim=len(ent_v[0]))",
                "ent_store.add(list(entity_views.values()), ent_v, ids=list(entity_views.keys()))",
                "",
                "edge_v = client.embed(edge_views) if edge_views else []",
                "edge_store = QdrantBackend('lr-edge', dim=len(edge_v[0]) if edge_v else 384)",
                "if edge_v:",
                "    edge_store.add(edge_views, edge_v, ids=[f'e{i}' for i in range(len(edge_views))])",
                "",
                "print('Both stores indexed.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 5 — The dual-level retrieve function",
                "",
                "For each query: hit both indexes, take top-k from each, format as context for the LLM.",
            ],
            code=[
                "def light_rag(question: str, top_k: int = 4) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    low = ent_store.search(qv, top_k=top_k)",
                "    high = edge_store.search(qv, top_k=top_k) if edge_views else []",
                "    rendered = (",
                "        'Entity neighbourhoods:\\n' + '\\n'.join(h.text for h in low)",
                "        + '\\n\\nKey relations:\\n' + '\\n'.join(h.text for h in high)",
                "    )",
                "    answer = client.chat(rendered + f'\\n\\nQuestion: {question}\\nAnswer:')",
                "    contexts = [h.text for h in low] + [h.text for h in high]",
                "    return answer, contexts",
                "",
                "ans, _ = light_rag('Which materials are connected to high-temperature superconductivity?')",
                "print(ans[:400])",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 6 — Wrap as `answer_question`",
                "",
                "Standard contract.",
            ],
            code=[
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    return light_rag(question)",
                "",
                "ans, _ = answer_question('How are Cooper pairs related to BCS theory?')",
                "print(ans[:400])",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — when do entity vs relation views win?",
                "",
                "On factual queries, entity views should dominate. On relational queries, relation views.",
            ],
            code=[
                "for q in [",
                "    'What is YBCO?',                                          # factual",
                "    'How are Cooper pairs related to BCS theory?',            # relational",
                "    'When was superconductivity discovered?',                 # factual",
                "    'How does flux pinning enable levitation?',               # relational",
                "]:",
                "    qv = client.embed([q])[0]",
                "    ent_top = ent_store.search(qv, top_k=1)[0].score",
                "    edge_top = edge_store.search(qv, top_k=1)[0].score if edge_views else 0.0",
                "    label = 'entity' if ent_top > edge_top else 'relation'",
                "    print(f'  {q[:55]:55s}  entity={ent_top:.3f}  relation={edge_top:.3f}  winner={label}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — top entity views for a synthesis query",
                "",
                "Look at the entity neighbourhoods that get retrieved. Quality of the neighbourhood description directly determines retrieval quality.",
            ],
            code=[
                "q = 'Which materials connect high-Tc superconductivity to applications in MRI?'",
                "qv = client.embed([q])[0]",
                "for h in ent_store.search(qv, top_k=5):",
                "    print(f'  {h.score:.3f}  {h.text[:120]}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost vs GraphRAG",
                "",
                "LightRAG skips community detection and per-community summaries. We confirm by counting LLM calls.",
            ],
            code=[
                "print(f'LightRAG indexing LLM calls: ~{len(chunks)} (triple extraction only)')",
                "print(f'LightRAG query-time LLM calls: 1 (just the answer)')",
                "print()",
                "print('GraphRAG (Recipe 31) by comparison:')",
                "print(f'  Indexing: ~{len(chunks)} triple-extraction + ~6 community-summary calls')",
                "print(f'  Query-time: 1 (similar)')",
                "print()",
                "print('Indexing savings: ~6 LLM calls (modest at this scale; large at production scale).')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — incremental updates",
                "",
                "Adding a new chunk requires re-extracting triples for that chunk and updating entity views. We show the pattern.",
            ],
            code=[
                "new_chunk_text = 'Magnesium diboride (MgB2) is an inexpensive superconductor with Tc around 39 K.'",
                "new_triples = extract_triples([('new-chunk', new_chunk_text)], client.chat)",
                "for t in new_triples:",
                "    g.add_node(t.subject)",
                "    g.add_node(t.object)",
                "    g.add_edge(t.subject, t.object, predicate=t.predicate, source=t.source_id)",
                "print(f'Graph after update: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.')",
                "print('Incremental update touched only the new chunk; full graph rebuild not required.')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a synthesis query.",
    ],
    run_code=[
        "ans, _ = answer_question('Which experimental phenomena connect Cooper pairs to applications like MRI and quantum computing?')",
        "print('=== LightRAG answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla vs LightRAG on a synthesis query.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'Which experimental phenomena connect Cooper pairs to applications in MRI?'",
        "base = vanilla_pipeline(q, corpus='wikipedia-superconductors', top_k=5)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'lightrag', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Six knobs in priority order:",
        "",
        "1. **Entity view template.** \"Entity: X. Connected to: Y, Z\" is the cookbook default. Richer templates (top-k neighbours by degree) improve retrieval.",
        "2. **Edge view template.** \"subject predicate object\" is minimal. Adding source-chunk context can help retrieval quality on long edges.",
        "3. **Top-k per view.** Cookbook uses 4 each. Higher catches more context, longer prompts.",
        "4. **Triple-extraction model.** Same lever as GraphRAG — quality of triples sets the ceiling for everything.",
        "5. **Embedder.** Same lever as everywhere; LightRAG benefits from a strong embedder because retrieval is the only quality signal.",
        "6. **Incremental update cadence.** Run extraction on new documents on a schedule. The graph can incrementally absorb new triples without rebuild.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Sparse graphs.** A graph with few edges produces few relation views, which means high-level retrieval gives up. Build a richer extraction prompt.",
        "- **Vague entity descriptions.** \"Entity: X. Connected to: Y\" is sometimes too thin. Enrich.",
        "- **No-graph fallback.** When the graph is empty or wrong, LightRAG silently degrades to entity-only search. Always test on a labelled slice.",
        "",
        "Compose with semantic chunking (Recipe 5) and reranking (Recipe 22). For multi-hop questions, route to HippoRAG (Recipe 33) instead.",
    ],
)


# =============================================================================
# HippoRAG PageRank Memory
# =============================================================================
HIPPORAG = Recipe(
    path="recipes/07-graph-and-memory/hipporag-pagerank-memory.ipynb",
    title="HippoRAG — Hippocampus-Inspired Memory via Personalised PageRank",
    category="graph-and-memory",
    corpus_filter="wikipedia-superconductors",

    theory_problem=[
        "Multi-hop questions are structurally hard for vector retrieval. \"Which experimental technique that uses Cooper pairs powers MRI machines?\" requires linking Cooper pairs → Josephson junctions → SQUID → magnetometry → MRI. No single chunk contains this whole chain; vector retrieval finds chunks near \"Cooper pairs\" or \"MRI\" but not the bridges between them.",
        "HippoRAG (OSU, NeurIPS 2024) borrows from neuroscience: the hippocampus retrieves memories via associative activation spreading from cue entities. Translated to RAG: extract entities from the query, seed personalised PageRank over the knowledge graph, return the highest-scored nodes plus the chunks they came from. Bridge nodes naturally surface because PageRank assigns importance through the graph's structure.",
    ],
    theory_origin=[
        "HippoRAG was published at NeurIPS 2024 by Gutiérrez et al. (Ohio State). The paper showed multi-hop benchmarks (MuSiQue, 2WikiMultiHopQA) improved by 6-20 points over vanilla RAG. HippoRAG 2 followed in 2025 with a learned scoring head; the cookbook uses the original PageRank-only variant for clarity. The neuroscience framing was more than metaphor — the paper argued PageRank diffusion behaves like associative recall in the hippocampus, which makes the technique a structural fit for multi-hop.",
        "By 2026 HippoRAG has become the default choice for multi-hop-heavy workloads. The cookbook composes it with GraphRAG-style graph construction so the underlying graph can serve both kinds of queries; the per-query strategy is what differs.",
    ],
    theory_landscape=[
        "Three graph-traversal RAG variants to know:",
        "",
        "- **GraphRAG (Recipe 31).** Communities + summaries. Good for synthesis.",
        "- **LightRAG (Recipe 32).** Dual-level views. Cheap, balanced.",
        "- **HippoRAG (this recipe).** PageRank from query entities. Best for multi-hop.",
        "",
        "All three share the underlying knowledge graph; the difference is in retrieval strategy. Production systems often build the graph once and route between strategies per query.",
    ],
    theory_when_to_use=[
        "Use HippoRAG when your queries are multi-hop. Research questions that span topics, comparative questions, anything that requires linking facts across documents.",
        "Skip it for factoid queries. Vanilla RAG is faster.",
        "Skip it when the corpus has no graph structure. A pile of FAQ answers has no edges to walk.",
        "Skip it when entity extraction is unreliable. The technique seeds PageRank on extracted entities; bad extraction means bad seeds means bad walks.",
    ],
    theory_intuition=[
        "Three intuitions to carry:",
        "",
        "**Seeds matter more than the algorithm.** PageRank with good seeds finds bridge nodes. PageRank with bad seeds returns random structurally-popular entities. Spend time on entity extraction.",
        "",
        "**Bridge nodes are the win.** Nodes connecting query entities are exactly what multi-hop questions need. PageRank's diffusion process surfaces them naturally.",
        "",
        "**alpha controls spread.** PageRank's damping factor controls how far the walk travels. Low alpha (0.3) stays near seeds; high alpha (0.85) spreads broadly. The cookbook default is 0.4.",
        "",
        "**The chunks behind the nodes matter.** PageRank returns entities. Map back to the source chunks to feed the generator. The graph remembers which chunk introduced each node.",
    ],

    architecture_mermaid="""
flowchart TB
  Q[Question] --> EE[LLM: extract<br/>entities]
  EE --> S[Seed nodes]
  G[(Knowledge Graph)] --> PR[Personalised PageRank<br/>from seeds]
  S --> PR
  PR --> R[Ranked nodes]
  R --> CH[Map nodes to<br/>source chunks]
  G --> CH
  CH --> A[LLM answers]
""",

    references=[
        Reference(
            title="HippoRAG — Neurobiologically Inspired Long-Term Memory for LLMs (Gutiérrez et al., 2024)",
            url="https://arxiv.org/abs/2405.14831",
            kind="paper",
            note="The NeurIPS 2024 paper.",
        ),
        Reference(
            title="OSU-NLP-Group/HippoRAG reference implementation",
            url="https://github.com/OSU-NLP-Group/HippoRAG",
            kind="repo",
            note="The paper's code.",
        ),
        Reference(
            title="HippoRAG 2 (2025)",
            url="https://arxiv.org/abs/2502.14802",
            kind="paper",
            note="The follow-up with a learned scoring head.",
        ),
        Reference(
            title="Personalised PageRank — Haveliwala et al., 2002",
            url="https://www.cs.cornell.edu/~bindel/papers/2014-toplas.pdf",
            kind="paper",
            note="The PageRank variant we use.",
        ),
        Reference(
            title="GraphRAG (Recipe 31)",
            url="https://arxiv.org/abs/2404.16130",
            kind="paper",
            note="Cousin technique for global synthesis.",
        ),
        Reference(
            title="MuSiQue multi-hop benchmark",
            url="https://github.com/StonyBrookNLP/musique",
            kind="repo",
            note="The benchmark where HippoRAG shines.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build the knowledge graph",
                "",
                "Same graph construction as GraphRAG and LightRAG.",
            ],
            code=[
                "from cookbook.corpora import load_wikipedia_superconductors",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.graphs import extract_triples, build_graph, personalized_pagerank",
                "",
                "docs = list(load_wikipedia_superconductors())[:14]",
                "chunks = sentence_window(docs, sentences_per_chunk=4)[:80]",
                "triples = extract_triples([(c.chunk_id, c.text) for c in chunks], client.chat)",
                "g = build_graph(triples)",
                "print(f'Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Build the entity extractor",
                "",
                "For each query, ask the LLM to list the entities or concepts mentioned. These become PageRank seeds.",
            ],
            code=[
                "import re",
                "",
                "def query_entities(question: str) -> list[str]:",
                "    raw = client.chat(",
                "        'List, one per line, the named entities or concepts in this question. '",
                "        'Use lowercase and keep them short.\\n\\n' + question",
                "    )",
                "    out = []",
                "    for line in raw.splitlines():",
                "        cleaned = re.sub(r'[^a-z0-9 \\-]', '', line.strip().lower()).strip()",
                "        if cleaned:",
                "            out.append(cleaned)",
                "    return out",
                "",
                "for q in [",
                "    'How is BCS theory connected to superconducting magnets used in MRI?',",
                "    'What materials show flux pinning?',",
                "]:",
                "    print(f'Q: {q}')",
                "    print(f'  entities: {query_entities(q)}')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 3 — Run personalised PageRank",
                "",
                "`cookbook.graphs.personalized_pagerank` seeds the walk with query entities and runs PageRank. The result is a score per node; we pick the top.",
            ],
            code=[
                "q = 'Which experimental phenomena connect Cooper pairs to SQUID measurements?'",
                "seeds = query_entities(q)",
                "print(f'Seeds: {seeds}')",
                "scores = personalized_pagerank(g, seeds, alpha=0.4)",
                "top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]",
                "for node, sc in top_nodes:",
                "    print(f'  {sc:.4f}  {node}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Build the HippoRAG answer function",
                "",
                "Use the top nodes' neighbourhoods as context for the answer LLM.",
            ],
            code=[
                "def hippo(question: str, top_k: int = 6) -> tuple[str, list[str]]:",
                "    seeds = query_entities(question)",
                "    scores = personalized_pagerank(g, seeds, alpha=0.4)",
                "    top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]",
                "    rendered = []",
                "    for node, sc in top_nodes:",
                "        neighbours = list(g.neighbors(node))[:4]",
                "        rendered.append(f'{node} (score {sc:.4f}) connects to: {neighbours}')",
                "    ctx = '\\n'.join(rendered)",
                "    answer = client.chat(",
                "        'Use this graph-walk context to answer.\\n' + ctx + f'\\nQuestion: {question}\\nAnswer:'",
                "    )",
                "    return answer, rendered",
                "",
                "ans, _ = hippo('Which experimental phenomena connect Cooper pairs to SQUID measurements?')",
                "print(ans[:400])",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap as `answer_question`",
                "",
                "Standard contract.",
            ],
            code=[
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    return hippo(question)",
                "",
                "ans, _ = answer_question('How does BCS theory connect to applications in MRI?')",
                "print(ans[:400])",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — bridge nodes for a multi-hop query",
                "",
                "For a multi-hop question, the top PageRank nodes should include intermediate entities not explicitly named in the query. Those are the bridges.",
            ],
            code=[
                "q = 'How does BCS theory connect to applications in MRI?'",
                "seeds = query_entities(q)",
                "print(f'Seeds (explicitly named): {seeds}')",
                "scores = personalized_pagerank(g, seeds, alpha=0.4)",
                "for node, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:8]:",
                "    is_seed = node in seeds",
                "    print(f'  {sc:.4f}  {node:35s}  {\"(seed)\" if is_seed else \"(bridge)\"}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — alpha (damping factor) sweep",
                "",
                "Low alpha stays near seeds; high alpha spreads broadly. Look at how the top-5 changes.",
            ],
            code=[
                "import pandas as pd",
                "rows = []",
                "for alpha in (0.15, 0.3, 0.5, 0.7, 0.85):",
                "    sc = personalized_pagerank(g, seeds, alpha=alpha)",
                "    top = sorted(sc.items(), key=lambda x: x[1], reverse=True)[:3]",
                "    rows.append({'alpha': alpha, 'top3': [t[0] for t in top]})",
                "pd.DataFrame(rows)",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what happens with no extracted entities?",
                "",
                "If the entity extractor returns nothing, PageRank falls back to uniform distribution over all nodes. The cookbook handles this gracefully.",
            ],
            code=[
                "scores = personalized_pagerank(g, [], alpha=0.4)",
                "top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]",
                "print('With empty seeds (uniform fallback), top-5:')",
                "for node, sc in top:",
                "    print(f'  {sc:.4f}  {node}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost",
                "",
                "HippoRAG adds one LLM call per query (entity extraction) on top of the answer. The PageRank itself is cheap (NetworkX, < 1 s on most graphs).",
            ],
            code=[
                "import time",
                "t0 = time.perf_counter()",
                "_ = personalized_pagerank(g, ['cooper pairs', 'squid'], alpha=0.4)",
                "pr_ms = (time.perf_counter() - t0) * 1000",
                "print(f'PageRank: {pr_ms:.1f} ms')",
                "print('Per-query cost: 1 entity-extraction LLM call + 1 answer LLM call.')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a multi-hop question.",
    ],
    run_code=[
        "ans, _ = answer_question('What phenomena link Cooper pairs to the operation of MRI scanners and particle accelerators?')",
        "print('=== HippoRAG answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla vs HippoRAG on a multi-hop question.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What phenomena link Cooper pairs to the operation of MRI scanners?'",
        "base = vanilla_pipeline(q, corpus='wikipedia-superconductors', top_k=5)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'hipporag', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Six knobs in priority order:",
        "",
        "1. **Entity extractor prompt.** The seeds are the technique. Tune; bad seeds produce bad walks.",
        "2. **alpha (PageRank damping).** 0.4 is the cookbook default. Lower stays near seeds, higher spreads more.",
        "3. **top_k nodes.** We use 6. Higher gathers more context but inflates the prompt.",
        "4. **Neighbour count per node.** \"4 neighbours\" is the cookbook default in the answer prompt; can be increased.",
        "5. **Compose with the graph.** GraphRAG and HippoRAG can share the same underlying graph; route per query.",
        "6. **Source-chunk recall.** Map PageRank-winning nodes back to their source chunks for the answer prompt — adds grounding.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Bad entity extraction.** If the extractor produces vague seeds or misses key entities, PageRank walks the wrong neighbourhoods. Tune the extractor.",
        "- **Disconnected components.** PageRank within a disconnected component can't reach the other component. Build a better graph or treat components separately.",
        "- **alpha mistuning.** Too low and you don't explore. Too high and you get noise. Sweep.",
        "",
        "Compose with GraphRAG (Recipe 31) for global queries and HippoRAG for multi-hop; route per query. Compose with LightRAG (Recipe 32) for cheap factoid queries.",
    ],
)


# =============================================================================
# Mem0 Long-Term Memory
# =============================================================================
MEM0 = Recipe(
    path="recipes/07-graph-and-memory/mem0-long-term-memory.ipynb",
    title="Mem0 — Long-Term Memory for Assistants",
    category="graph-and-memory",
    corpus_filter=None,

    theory_problem=[
        "Conversational assistants need to remember things across sessions: user preferences, prior decisions, ongoing projects. Naive approaches — stuffing the whole conversation history into the prompt — break down past a few turns. The model can't pick out what matters; the prompt balloons; tokens get expensive; and the same facts get repeated turn after turn.",
        "Mem0 (and similar systems like Letta/MemGPT, Zep) treats memory as a small retrievable knowledge base. After each turn, an extractor pulls durable facts from the exchange. Facts are deduplicated, scored, and stored. Before each turn, relevant facts are retrieved and prepended. The model sees a tight prompt with only what matters, regardless of how long the conversation has been going on.",
    ],
    theory_origin=[
        "Mem0 was released as a hosted product and open-source library in early 2024. The conceptual roots — extract-store-recall memory for agents — go back to MemGPT (Stanford, 2023) and earlier work on external memory for language models. The practical contribution was packaging it as a clean API: `add(turn)`, `search(query)`, with deduplication and scoring built in.",
        "By 2026 long-term memory for assistants is table stakes. Mem0 has competitors (Letta, Zep, Cognee) but the pattern is largely the same across all of them: extract durable facts, embed for retrieval, recall by relevance. The cookbook implementation matches that shape directly so the pattern is visible without any framework magic.",
    ],
    theory_landscape=[
        "Memory layers complement the rest of the RAG stack and live alongside retrieval rather than replacing it:",
        "",
        "- **Mem0 (this recipe).** Persistent per-user fact store with extract-and-recall.",
        "- **Letta / MemGPT.** Memory + tool-use unified agent runtime.",
        "- **Zep.** Memory store with stronger temporal reasoning and time-windowed recall.",
        "- **Cognee.** Knowledge-graph-based memory; closer to GraphRAG (Recipe 31).",
        "- **Native conversation history.** Cheapest, breaks past a few turns.",
        "",
        "Memory composes with retrieval (RAG): the memory layer holds *who the user is*; the retrieval layer holds *what's in the corpus*. Production assistants use both.",
    ],
    theory_when_to_use=[
        "Use Mem0-style memory in conversational assistants where users have ongoing relationships with the system. Coding assistants, customer-support bots, personal assistants — anywhere a user comes back and expects the system to know things about them.",
        "Skip it for stateless Q&A systems. There's no relationship to remember.",
        "Skip it when privacy regimes prohibit persistent user data. GDPR/CCPA require careful design and deletion mechanisms.",
    ],
    theory_intuition=[
        "Four intuitions:",
        "",
        "**Extract is the bottleneck.** What you extract determines what you can recall. Bad extraction gives bad recall.",
        "",
        "**Recall is a retrieval.** Standard cosine search over stored facts. The same machinery as RAG.",
        "",
        "**Deduplication is non-trivial.** Two facts that say the same thing in different words should merge. The cookbook implementation is naive; production systems use embedding-similarity deduplication.",
        "",
        "**Forgetting is a feature.** Not every fact is durable. Score by recency and reuse, prune the long tail.",
    ],

    architecture_mermaid="""
flowchart LR
  T[Conversation turn] --> EX[LLM:<br/>extract durable facts]
  EX --> D[Dedupe vs<br/>existing memory]
  D --> M[(Memory store)]
  Q[New question] --> RC[Recall:<br/>retrieve relevant facts]
  M --> RC
  RC --> AN[LLM answers<br/>with memory context]
""",

    references=[
        Reference(
            title="Mem0 documentation",
            url="https://docs.mem0.ai/",
            kind="docs",
            note="Hosted product docs.",
        ),
        Reference(
            title="mem0ai/mem0 repository",
            url="https://github.com/mem0ai/mem0",
            kind="repo",
            note="Open-source implementation.",
        ),
        Reference(
            title="MemGPT — Towards LLMs as Operating Systems (Packer et al., 2023)",
            url="https://arxiv.org/abs/2310.08560",
            kind="paper",
            note="The earlier paper that framed the problem.",
        ),
        Reference(
            title="Letta (formerly MemGPT) framework",
            url="https://docs.letta.com/",
            kind="docs",
            note="Memory + agent runtime.",
        ),
        Reference(
            title="Zep memory store",
            url="https://www.getzep.com/",
            kind="docs",
            note="Alternative with strong temporal reasoning.",
        ),
        Reference(
            title="LlamaIndex chat memory abstractions",
            url="https://developers.llamaindex.ai/python/framework/module_guides/storing/chat_stores/",
            kind="docs",
            note="The framework-level abstraction.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build a memory store",
                "",
                "Memory is just a Qdrant collection of fact strings. Each fact has metadata (category, recency, score).",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "",
                "MEM_DIM = len(client.embed(['probe'])[0])",
                "mem = QdrantBackend('mem0', dim=MEM_DIM)",
                "print(f'Memory store ready (dim={MEM_DIM}).')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — The fact extractor",
                "",
                "Given a conversation turn, extract durable facts about the user worth remembering. We ask the LLM for JSON.",
            ],
            code=[
                "import json",
                "import re",
                "",
                "EXTRACT = (",
                "    'Extract durable facts about the user worth remembering from this turn. '",
                '    \'Respond as JSON {{\"facts\": [{{\"text\": str, \"category\": str}}]}}.\\n\\n\'',
                "    'Turn: {turn}'",
                ")",
                "",
                "def remember(turn: str):",
                "    raw = client.chat(EXTRACT.format(turn=turn))",
                "    m = re.search(r'\\{.*\\}', raw, flags=re.DOTALL)",
                "    if not m:",
                "        return []",
                "    try:",
                "        facts = json.loads(m.group()).get('facts', [])",
                "    except json.JSONDecodeError:",
                "        return []",
                "    texts = [f['text'] for f in facts if isinstance(f, dict) and f.get('text')]",
                "    if not texts:",
                "        return []",
                "    vecs = client.embed(texts)",
                "    mem.add(",
                "        texts, vecs,",
                "        metadatas=[{'category': f.get('category', 'general')} for f in facts]",
                "    )",
                "    return texts",
                "",
                "stored = remember('I prefer concise, one-paragraph answers and I work in Rust most of the time.')",
                "print('Stored:')",
                "for t in stored:",
                "    print(f'  - {t}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Simulate a few conversation turns",
                "",
                "Store facts across turns. Each turn's extraction adds to the memory.",
            ],
            code=[
                "turns = [",
                "    'I prefer terse one-paragraph answers.',",
                "    'I work primarily in Rust and avoid Python unless required.',",
                "    'My current research project is on high-temperature superconductors.',",
                "    'I find verbose explanations annoying — stick to the point.',",
                "]",
                "for t in turns:",
                "    facts = remember(t)",
                "    print(f'Turn: {t[:60]}')",
                "    for f in facts:",
                "        print(f'  +  {f}')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 4 — Recall relevant memories",
                "",
                "Standard cosine retrieval over the memory store.",
            ],
            code=[
                "def recall(question: str, top_k: int = 5):",
                "    qv = client.embed([question])[0]",
                "    return mem.search(qv, top_k=top_k)",
                "",
                "for h in recall('Suggest a weekend reading list for me.'):",
                "    print(f'  {h.score:.3f}  {h.text}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap as `answer_question`",
                "",
                "Standard contract. We recall first, then answer with memory in context.",
            ],
            code=[
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    facts = [h.text for h in recall(question)]",
                "    answer = client.chat(",
                "        'Known facts about user:\\n' + '\\n'.join(facts)",
                "        + f'\\n\\nQ: {question}\\nA:'",
                "    )",
                "    return answer, facts",
                "",
                "ans, _ = answer_question('What kind of programming book should I read this weekend?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what's in the memory store?",
                "",
                "Recall everything to see the full state.",
            ],
            code=[
                "from cookbook import _cache",
                "print(f'Memory entries: {_cache.stats()[\"entries\"]} (cache; not memory directly)')",
                "print()",
                "print('Recall against \"user preferences\":')",
                "for h in recall('user preferences', top_k=10):",
                "    print(f'  {h.score:.3f}  {h.text}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — recall on different questions",
                "",
                "Different queries should recall different facts. Verify that retrieval is selective.",
            ],
            code=[
                "for q in [",
                "    'What programming language should I focus on?',",
                "    'What research topic am I interested in?',",
                "    'How verbose should my replies be?',",
                "]:",
                "    facts = [h.text for h in recall(q, top_k=2)]",
                "    print(f'Q: {q}')",
                "    for f in facts:",
                "        print(f'  +  {f}')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — duplicate detection",
                "",
                "Add a near-duplicate fact and see what happens. Naive Mem0 doesn't dedupe; production Mem0 does.",
            ],
            code=[
                "before_count = mem.client.count(mem.collection).count",
                "remember('I dislike verbose answers and prefer concise ones.')",
                "after_count = mem.client.count(mem.collection).count",
                "print(f'Memory size: {before_count} -> {after_count}')",
                "print('Naive Mem0 stores duplicates; production Mem0 detects and merges them.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost",
                "",
                "Each turn: 1 LLM call (extraction) + N embeddings (one per fact). Each query: 1 embedding + 1 LLM call.",
            ],
            code=[
                "print('Per-turn cost: 1 extraction LLM call + N embeddings.')",
                "print('Per-query cost: 1 query embedding + 1 answer LLM call.')",
                "print('Memory stays cheap until you scale to thousands of users.')",
            ],
        ),
    ],

    run_md=[
        "End-to-end with memory.",
    ],
    run_code=[
        "ans, _ = answer_question('Recommend a reading list aligned with my interests.')",
        "print('=== Memory-grounded answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla (no memory) vs Mem0 (with memory). The vanilla call lacks the user context Mem0 retrieves; the answer should be visibly less personalised.",
    ],
    comparison_code=[
        "q = 'Recommend a reading list aligned with my interests.'",
        "base_ans = client.chat(q)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'no-memory', 'preview': base_ans[:160]},",
        "    {'pipeline': 'mem0', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Six knobs in priority order:",
        "",
        "1. **Extraction prompt.** Quality of stored facts dominates everything downstream. Tune carefully and review what gets stored.",
        "2. **Top-k recall.** Cookbook uses 5. Higher catches more memories at the cost of prompt length.",
        "3. **Deduplication.** Production systems dedupe by embedding similarity. The cookbook doesn't to keep the implementation small.",
        "4. **Decay / forgetting.** Score recency and reuse; prune low-score facts periodically to keep the store relevant.",
        "5. **Multi-user isolation.** Use Qdrant payload filters to isolate per-user memories in a shared collection.",
        "6. **Categorisation.** Stored facts can carry a category (preferences, projects, history). Recall can filter by category.",
    ],

    discussion_md=[
        "Four failure modes you'll meet in production:",
        "",
        "- **Fact drift.** The extractor invents facts from ambiguous turns. Validate before storing — at minimum, check that the fact is grounded in the turn text.",
        "- **Stale memories.** \"I'm working on X\" recorded today is wrong six months later. Decay; categorise as time-bounded.",
        "- **Privacy.** Persistent per-user storage triggers regulatory questions. Plan for deletion (GDPR right-to-be-forgotten) from day one.",
        "- **Cross-user contamination.** A shared memory store leaks data between users. Use strict per-user isolation.",
        "",
        "Compose with retrieval (RAG): memory holds who the user is; RAG holds what's in the corpus. Compose with LangGraph (Recipe 28): a memory node and a retrieval node, both feeding the answer node.",
    ],
)


RECIPES = [MSGRAPH, LIGHTRAG, HIPPORAG, MEM0]
