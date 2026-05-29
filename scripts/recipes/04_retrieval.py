"""Category 4 — Retrieval.

Four recipes total. Batch 3 ships hybrid and ColBERT; Batch 6 adds the rest.
"""
from __future__ import annotations

from authoring import CodeStep, Recipe, Reference


# =============================================================================
# Hybrid Dense + BM25
# =============================================================================
HYBRID = Recipe(
    path="recipes/04-retrieval/hybrid-dense-plus-bm25.ipynb",
    title="Hybrid Retrieval — Dense + BM25 with Reciprocal Rank Fusion",
    category="retrieval",
    corpus_filter="sec-10k-pltr",

    theory_problem=[
        "Dense retrieval is great at paraphrase. \"How does the company protect its customers' data?\" finds passages about \"data protection\" and \"information security\". But dense retrieval is awful at exact-match terms — product codes, error codes, regulation citations (\"15 U.S.C. § 78m\"), specific dollar figures. A vector of `15 U.S.C. § 78m` is just numbers and characters with no obvious semantic neighbours.",
        "BM25 is the opposite. It is a token-frequency model, so it locks onto exact matches like a guided missile. It fails on paraphrase. The pragmatic answer is to run both, fuse the rankings, and let each retriever cover the other's weakness. Reciprocal Rank Fusion is the canonical fuse: parameter-free, robust to score-scale differences, takes any two rankings and produces a single ranking.",
    ],
    theory_origin=[
        "BM25 dates to the 1990s — Robertson and Spärck Jones at Cambridge. RRF was introduced by Cormack, Clarke, and Büttcher in 2009 as a parameter-free way to combine TREC submissions. Their experiment showed that simple RRF beat learned combinations on most tasks. The hybrid dense + BM25 pattern emerged with the first wave of dense retrieval (Karpukhin et al., DPR 2020) and became standard in 2023 when LangChain and LlamaIndex shipped one-line hybrid retrievers.",
        "By 2026 every serious vector store has a native hybrid mode: Qdrant's BM42, Weaviate's BlockMax, Pinecone's hybrid index, Milvus' sparse-dense. We use a portable RRF implementation here so the technique reads clearly; production code should prefer the database's native hybrid path.",
    ],
    theory_landscape=[
        "Three options for combining sparse and dense:",
        "",
        "- **Reciprocal Rank Fusion (this recipe).** Sum `1 / (k + rank)` across both rankings. Parameter-free, robust. Default choice.",
        "- **Score normalisation + linear combination.** Min-max-normalise each ranking's scores, weighted-sum. Tunable but fragile when score distributions shift.",
        "- **Cross-encoder reranker over the union.** Take top-N from both, send all to a cross-encoder. Most accurate; slowest. Often combined with RRF as a two-stage pipeline.",
        "",
        "Stack hybrid with reranking (Recipe 22) and you have the Anthropic contextual-retrieval recommended pipeline. Stack with contextual chunking (Recipe 7) and you have what most production systems are running by 2026.",
    ],
    theory_when_to_use=[
        "Use hybrid retrieval whenever your queries mix paraphrase and exact-match. SEC filings (lots of citations and codes), customer support (product SKUs), legal text (statute references), codebases (function names) — all benefit obviously.",
        "Skip hybrid only when your corpus is purely paraphrase-friendly (Wikipedia summary paragraphs) and queries never contain exact-match tokens. In practice this is rare; even Wikipedia queries often ask about specific dates or names.",
        "Skip BM25 entirely if you cannot tokenise sanely. Languages without whitespace, or domains where stemming changes meaning (chemistry names, legal text), need careful BM25 setup. A bad BM25 hurts hybrid more than no BM25.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**RRF turns disagreement into evidence.** When dense and sparse agree on a passage, that passage scores high under both rankings, which RRF rewards heavily. When they disagree, RRF blends — neither retriever's strong but wrong picks dominate.",
        "",
        "**The constant `k` (default 60) matters less than people think.** RRF is robust to `k` in the range 30–100. Pick 60 and move on; do not waste a week tuning it.",
        "",
        "**Hybrid pays off most at the bottom of the ranking.** Top-1 is usually the same chunk under dense, sparse, or hybrid. The difference shows up at ranks 3–10, where hybrid pulls in chunks that one retriever ranked low and the other ranked high. That is where a downstream reranker (Recipe 22) earns its keep.",
    ],

    architecture_mermaid="""
flowchart LR
  Q[Query] --> D[Dense retriever<br/>top-20]
  Q --> S[Sparse BM25<br/>top-20]
  D --> F[Reciprocal<br/>Rank Fusion]
  S --> F
  F --> R[Fused top-k<br/>ready for generation]
""",

    references=[
        Reference(
            title="Reciprocal Rank Fusion (Cormack et al., 2009)",
            url="https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf",
            kind="paper",
            note="The original RRF paper. Two pages of math, a lifetime of utility.",
        ),
        Reference(
            title="BM25 — Probabilistic Information Retrieval",
            url="https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf",
            kind="paper",
            note="Robertson and Zaragoza's canonical retrospective on BM25.",
        ),
        Reference(
            title="Dense Passage Retrieval for Open-Domain QA (Karpukhin et al., 2020)",
            url="https://arxiv.org/abs/2004.04906",
            kind="paper",
            note="The DPR paper that popularised dense retrieval; established hybrid as a baseline.",
        ),
        Reference(
            title="Qdrant Hybrid Search documentation",
            url="https://qdrant.tech/documentation/concepts/hybrid-queries/",
            kind="docs",
            note="Native hybrid in our default vector store.",
        ),
        Reference(
            title="rank-bm25 Python library",
            url="https://github.com/dorianbrown/rank_bm25",
            kind="repo",
            note="The lightweight BM25 implementation we use here.",
        ),
        Reference(
            title="Anthropic Contextual Retrieval (uses hybrid)",
            url="https://www.anthropic.com/news/contextual-retrieval",
            kind="blog",
            note="Anthropic's recommended pipeline is hybrid + contextual + reranking.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Index Palantir's 10-K",
                "",
                "The SEC filing is a perfect testbed: dense prose mixed with statute citations, dollar figures, and product names. The mix exercises both halves of the hybrid retriever.",
            ],
            code=[
                "from cookbook.corpora import load_sec_10k",
                "from cookbook.chunkers import fixed_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_sec_10k())",
                "chunks = fixed_window(docs, target_tokens=384, overlap_tokens=48)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('hybrid', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 2 — Build the BM25 side",
                "",
                "BM25 is a token-frequency model. We tokenise each chunk by lowercasing and splitting on whitespace — naïve but fine for English prose. Production setups stem and remove stopwords; the cookbook keeps it simple so the implementation fits in one cell.",
            ],
            code=[
                "from rank_bm25 import BM25Okapi",
                "",
                "tokenised = [c.text.lower().split() for c in chunks]",
                "bm25 = BM25Okapi(tokenised)",
                "",
                "def bm25_search(query: str, top_k: int = 20):",
                "    scores = bm25.get_scores(query.lower().split())",
                "    import numpy as np",
                "    order = np.argsort(scores)[::-1][:top_k]",
                "    return [(int(i), float(scores[i])) for i in order]",
                "",
                "print('Top-3 BM25 hits for \"AIP commercial revenue concentration\":')",
                "for i, s in bm25_search('AIP commercial revenue concentration', top_k=3):",
                "    print(f'  score={s:6.2f}  {chunks[i].text[:160]}')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 3 — Build the dense side",
                "",
                "Same Qdrant store we built above. Wrap it in a small function that returns `(chunk_index, score)` tuples so it has the same shape as the BM25 side. RRF doesn't care about scores beyond rank order, but having a consistent shape makes the code symmetric.",
            ],
            code=[
                "id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}",
                "",
                "def dense_search(query: str, top_k: int = 20):",
                "    qv = client.embed([query])[0]",
                "    hits = store.search(qv, top_k=top_k)",
                "    return [(id_to_idx[h.doc_id], float(h.score)) for h in hits]",
                "",
                "print('Top-3 dense hits for the same query:')",
                "for i, s in dense_search('AIP commercial revenue concentration', top_k=3):",
                "    print(f'  score={s:.4f}  {chunks[i].text[:160]}')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 4 — Reciprocal Rank Fusion",
                "",
                "Take both rankings, for each chunk sum `1 / (k + rank)`. Sort. That is the whole algorithm. `k=60` is the canonical constant from the Cormack paper.",
            ],
            code=[
                "def rrf(rankings, k: int = 60, top_k: int = 10):",
                "    scores: dict[int, float] = {}",
                "    for ranking in rankings:",
                "        for rank, (idx, _) in enumerate(ranking):",
                "            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)",
                "    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]",
                "",
                "q = 'AIP commercial revenue concentration'",
                "fused = rrf([dense_search(q, top_k=20), bm25_search(q, top_k=20)], top_k=5)",
                "print('Top-5 hybrid hits:')",
                "for idx, s in fused:",
                "    print(f'  rrf_score={s:.4f}  {chunks[idx].text[:160]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap as `answer_question`",
                "",
                "The contract. Hybrid retrieval, then standard stuffed-context generation.",
            ],
            code=[
                "PROMPT = (",
                "    'Use only the passages below to answer the question. '",
                "    'If they do not contain the answer, say so plainly.\\n\\n'",
                "    'Passages:\\n{context}\\n\\nQuestion: {question}\\nAnswer:'",
                ")",
                "",
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    fused = rrf(",
                "        [dense_search(question, top_k=20), bm25_search(question, top_k=20)],",
                "        top_k=k,",
                "    )",
                "    contexts = [chunks[i].text for i, _ in fused]",
                "    return client.chat(PROMPT.format(context='\\n\\n'.join(contexts), question=question)), contexts",
                "",
                "ans, _ = answer_question('What concentration risks does Palantir disclose for government customers?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — which retriever's choices dominate the fused ranking?",
                "",
                "For a representative query, mark each fused-top-10 chunk with how dense and BM25 ranked it. When a chunk was top-3 for both, RRF promotes it strongly. When it was top-3 for only one, it still makes the fused top-10 if the other ranked it somewhere reasonable.",
            ],
            code=[
                "q = 'AIP commercial revenue concentration'",
                "dense = dense_search(q, top_k=20)",
                "sparse = bm25_search(q, top_k=20)",
                "dense_rank = {idx: r for r, (idx, _) in enumerate(dense)}",
                "sparse_rank = {idx: r for r, (idx, _) in enumerate(sparse)}",
                "fused = rrf([dense, sparse], top_k=10)",
                "",
                "import pandas as pd",
                "rows = []",
                "for idx, s in fused:",
                "    rows.append({",
                "        'rrf_rank': len(rows) + 1,",
                "        'dense_rank': dense_rank.get(idx, '>20'),",
                "        'bm25_rank': sparse_rank.get(idx, '>20'),",
                "        'preview': chunks[idx].text[:80],",
                "    })",
                "pd.DataFrame(rows)",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — exact-match query that should favour BM25",
                "",
                "A query with a precise citation or product name. BM25 should rank the exact match first; dense often misses it. Hybrid recovers the right answer in either case.",
            ],
            code=[
                "q = 'Foundry'",
                "print('Dense top-3:')",
                "for i, s in dense_search(q, top_k=3):",
                "    print(f'  {s:.3f}  {chunks[i].text[:140]}')",
                "print()",
                "print('BM25 top-3:')",
                "for i, s in bm25_search(q, top_k=3):",
                "    print(f'  {s:6.2f}  {chunks[i].text[:140]}')",
                "print()",
                "print('Hybrid top-3:')",
                "for i, s in rrf([dense_search(q, top_k=20), bm25_search(q, top_k=20)], top_k=3):",
                "    print(f'  {s:.4f}  {chunks[i].text[:140]}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — paraphrase query that should favour dense",
                "",
                "The opposite test: a query that uses none of the exact words in the answer. BM25 will miss; dense should land it.",
            ],
            code=[
                "q = 'How does the company keep customer data safe from unauthorised access?'",
                "print('Dense top-3:')",
                "for i, s in dense_search(q, top_k=3):",
                "    print(f'  {s:.3f}  {chunks[i].text[:140]}')",
                "print()",
                "print('BM25 top-3:')",
                "for i, s in bm25_search(q, top_k=3):",
                "    print(f'  {s:6.2f}  {chunks[i].text[:140]}')",
                "print()",
                "print('Hybrid top-3:')",
                "for i, s in rrf([dense_search(q, top_k=20), bm25_search(q, top_k=20)], top_k=3):",
                "    print(f'  {s:.4f}  {chunks[i].text[:140]}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — how RRF's `k` constant changes the ranking",
                "",
                "Sweep `k` from small (10) to large (200). The top-5 usually does not move; the bottom of the top-20 shifts. This is why RRF is considered parameter-free — the constant matters at the tail, not at the head.",
            ],
            code=[
                "q = 'AIP commercial revenue concentration'",
                "d = dense_search(q, top_k=20); s = bm25_search(q, top_k=20)",
                "for k_val in (10, 30, 60, 100, 200):",
                "    top5 = [idx for idx, _ in rrf([d, s], k=k_val, top_k=5)]",
                "    print(f'  k={k_val:3d}  top-5 ids: {top5}')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a question that needs both halves of hybrid: government-concentration risk wording (paraphrase) plus specific revenue language (exact-match).",
    ],
    run_code=[
        "q = 'What does the company say about concentration risks tied to specific government customers and how is AIP positioned in the commercial segment?'",
        "ans, ctxs = answer_question(q)",
        "print('=== Hybrid answer ===')",
        "print(ans)",
        "print()",
        "print(f'(used {len(ctxs)} contexts)')",
    ],

    comparison_md=[
        "Vanilla baseline vs hybrid. Vanilla uses dense alone; hybrid adds BM25. The interesting case is a question with both paraphrase and exact-match components.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What does the company say about concentration risks tied to specific government customers and how is AIP positioned in the commercial segment?'",
        "base = vanilla_pipeline(q, corpus='sec-10k-pltr', top_k=5)",
        "ours_a, ours_c = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla (dense only)', 'top_preview': base.contexts[0][:140]},",
        "    {'pipeline': 'hybrid (dense + bm25)', 'top_preview': ours_c[0][:140]},",
        "])",
    ],

    tuning_md=[
        "Five knobs:",
        "",
        "1. **Top-N per retriever.** We use top-20 for both before fusing. Larger N catches more recall; the cost is the BM25 scoring loop (cheap) and the dense retrieval (cheap too). Production systems often go to 50 or 100.",
        "2. **RRF `k` constant.** Default 60. Lower `k` rewards top-rank agreement more sharply; higher `k` flattens the curve. Most sweeps find the result is insensitive to `k` in [30, 100].",
        "3. **Tokeniser for BM25.** Whitespace split is the cookbook default. For multi-word entities (\"New York\") or stemmed forms, use a smarter tokeniser. Snowball stemming usually adds 1–3 points on technical corpora.",
        "4. **Weight on the dense ranking.** Vanilla RRF treats both rankings equally. Weighted RRF gives one ranking more influence; useful when you know your corpus favours one side. Tune with a held-out eval.",
        "5. **Native hybrid in the database.** Qdrant BM42, Weaviate BlockMax, Pinecone hybrid. The performance and ergonomic wins are real at scale. The cookbook code uses portable RRF for clarity; production code should use the native path.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Bad BM25 tokenisation.** Whitespace splitting misses casing, multi-word terms, and hyphens. The cookbook does not bother to fix this because the corpus is simple; production must.",
        "- **One retriever dominates the union.** If your dense scores are well-calibrated and BM25 is noisy, the fused ranking essentially equals the dense ranking. Either narrow the BM25 source corpus (per-section BM25) or weight the dense side down.",
        "- **Score scale leakage.** If you ever sum raw scores from dense and BM25 (instead of using RRF), score-scale differences will destroy your ranking. Stick to RRF; do not try to be clever with score arithmetic.",
        "",
        "Hybrid is the cheapest single retrieval upgrade in this cookbook. Anthropic's contextual retrieval pipeline (contextual chunks + BM25 + reranker) is built on it. Add cross-encoder reranking (Recipe 22) on top and you have most of what production systems use in 2026.",
    ],
)


# =============================================================================
# ColBERT Late Interaction
# =============================================================================
COLBERT = Recipe(
    path="recipes/04-retrieval/colbert-late-interaction.ipynb",
    title="ColBERT Late Interaction — Multi-Vector MaxSim Retrieval",
    category="retrieval",
    corpus_filter="arxiv-mamba",

    theory_problem=[
        "Single-vector retrieval loses information. A 1024-dim vector cannot remember every named entity, every relationship, every nuanced phrase. When you embed a 400-word chunk into one vector, you average it. Some signal survives, some does not.",
        "ColBERT — Contextualised Late Interaction over BERT, 2020 — keeps every token's embedding. At search time, for each query token, find its maximum cosine over chunk tokens, sum those maxes. The chunk score is the sum of per-query-token MaxSims. State-of-the-art recall on BEIR; ColBERT v2 cut storage by 80x with quantisation. By 2026 every major vector store (Qdrant, Weaviate, Vespa, LanceDB) supports multi-vector retrieval natively.",
    ],
    theory_origin=[
        "ColBERT was introduced by Khattab and Zaharia at Stanford in 2020. The original paper showed near-cross-encoder quality at bi-encoder speed. ColBERT v2 (2022) added quantisation, residual compression, and approximate MaxSim that brought storage costs into the range where multi-vector indexes are practical.",
        "By 2024 ColBERT-style late interaction had become the dominant pattern for high-recall retrieval, with the multimodal variant (ColPali — Recipe 35) extending it to vision-language models. The 2026 wave of multi-vector-native vector stores (Qdrant 1.10+, Weaviate v1.27+) made it trivial to deploy without DIY token-level indexing.",
    ],
    theory_landscape=[
        "Three levels of retrieval expressivity:",
        "",
        "- **Single-vector dense (Recipe 2)** — one vector per chunk. Cheapest, fastest, lowest ceiling.",
        "- **Multi-vector late interaction (this recipe)** — per-token vectors, MaxSim aggregation. Higher recall, ~30x storage, comparable inference latency.",
        "- **Cross-encoder reranker (Recipe 22)** — joint scoring at query time. Highest quality, slowest, used as a reranker over a shortlist.",
        "",
        "Production systems usually combine: shortlist with single-vector dense, rerank with ColBERT MaxSim or a cross-encoder. The cookbook shows the pure ColBERT path here so the technique is visible end to end; recipe 22 shows the shortlist+rerank composition.",
    ],
    theory_when_to_use=[
        "Use ColBERT when retrieval quality matters more than storage cost. Search, e-discovery, anything where a 5–10 percent recall lift translates directly to business value. Multi-vector storage at 30x the size of a single-vector index is a few extra dollars at gigabyte scale and a real budget conversation at terabyte scale.",
        "Skip it when storage cost dominates and a cheaper reranker is available. A cross-encoder rerank over single-vector dense is often almost as good for half the storage budget.",
        "Skip it when your corpus is small. With under 10k chunks, single-vector dense is already accurate enough that ColBERT's marginal lift may not pay for the engineering complexity.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**MaxSim is per-token best-matching.** For each query token, you find the chunk token that matches it best. The chunk score is the sum of those best matches. The chunk \"wins\" if it has *some* token that matches each query token well, not if its average vector is similar.",
        "",
        "**Late interaction means \"compute interactions late, not early\".** A cross-encoder interacts query and chunk inside the model. A bi-encoder interacts not at all — it just produces independent vectors. Late interaction is the middle: independent token vectors, interacted via MaxSim at search time.",
        "",
        "**Token-level precision is what wins on rare entities.** A query mentioning \"selective scan\" should retrieve chunks that contain the words \"selective\" and \"scan\" near each other. Single-vector embedders average those away; ColBERT keeps the token signals separate and compares them individually.",
    ],

    architecture_mermaid="""
flowchart TB
  Q[Query] --> QT[Per-query-token<br/>embeddings]
  C[Chunks] --> CT[Per-chunk-token<br/>embeddings]
  CT --> S[(Multi-vector<br/>index)]
  QT --> MS[For each query token<br/>find max sim over chunk tokens]
  S --> MS
  MS --> SUM[Sum MaxSims<br/>= chunk score]
  SUM --> R[Ranked chunks]
""",

    references=[
        Reference(
            title="ColBERT — Efficient and Effective Passage Search (Khattab and Zaharia, 2020)",
            url="https://arxiv.org/abs/2004.12832",
            kind="paper",
            note="The original paper.",
        ),
        Reference(
            title="ColBERTv2 — Compressed Late Interaction (Santhanam et al., 2022)",
            url="https://arxiv.org/abs/2112.01488",
            kind="paper",
            note="The version that made storage costs practical.",
        ),
        Reference(
            title="ColBERT v2 reference repository",
            url="https://github.com/stanford-futuredata/ColBERT",
            kind="repo",
            note="Stanford's canonical implementation.",
        ),
        Reference(
            title="Qdrant Multi-Vector documentation",
            url="https://qdrant.tech/documentation/concepts/vectors/#multivectors",
            kind="docs",
            note="Native multi-vector retrieval in our default vector store.",
        ),
        Reference(
            title="fastembed Late-Interaction Embedding",
            url="https://qdrant.github.io/fastembed/examples/Supported_Models/",
            kind="docs",
            note="The ColBERTv2 loader used in this recipe.",
        ),
        Reference(
            title="BEIR Benchmark",
            url="https://github.com/beir-cellar/beir",
            kind="repo",
            note="The standard recall benchmark on which ColBERT's lift was measured.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load the corpus and chunk it",
                "",
                "We use a small slice of the Mamba paper. ColBERT's storage scales as `n_chunks * tokens_per_chunk * dim`, so notebook-scale demonstrations need small chunk counts. The technique generalises directly; only the indexing cost grows.",
            ],
            code=[
                "from cookbook.corpora import load_arxiv_mamba",
                "from cookbook.chunkers import sentence_window",
                "",
                "docs = list(load_arxiv_mamba())[:6]",
                "chunks = sentence_window(docs, sentences_per_chunk=4, overlap=1)",
                "print(f'Working with {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 2 — Load the late-interaction embedder",
                "",
                "We use `fastembed`'s ColBERT v2 — small, fast, CPU-runnable. The model emits token-level embeddings; `embed(text)` returns one vector per token rather than one vector per text.",
            ],
            code=[
                "from fastembed import LateInteractionTextEmbedding",
                "",
                "encoder = LateInteractionTextEmbedding('colbert-ir/colbertv2.0')",
                "print('Loaded ColBERT v2.')",
                "sample = list(encoder.embed(['selective scan in state-space models']))[0]",
                "print(f'Sample produced {len(sample)} token vectors of dim {len(sample[0])}.')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 3 — Index per-chunk token embeddings",
                "",
                "Compute token embeddings for every chunk and keep them in memory. For 100+ chunks this is the slow step; once cached it is reused freely.",
            ],
            code=[
                "import numpy as np",
                "",
                "chunk_token_arrays = []",
                "for v in encoder.embed([c.text for c in chunks]):",
                "    arr = np.asarray(v, dtype=np.float32)",
                "    arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-9)",
                "    chunk_token_arrays.append(arr)",
                "print(f'Indexed {len(chunk_token_arrays)} chunks with token-level vectors.')",
                "print(f'First chunk: {chunk_token_arrays[0].shape[0]} token vectors, dim={chunk_token_arrays[0].shape[1]}.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 4 — Implement MaxSim",
                "",
                "For a query, embed it (per-token), then for each query token find its best cosine over all token vectors of a chunk, sum those, that is the chunk score. Iterate over chunks. This is the search loop — slow in pure Python, fast in C inside Qdrant's multi-vector index.",
            ],
            code=[
                "def maxsim(query_tokens: np.ndarray, chunk_token_lists) -> list[float]:",
                "    scores = []",
                "    for tlist in chunk_token_lists:",
                "        sim = query_tokens @ tlist.T",
                "        scores.append(float(sim.max(axis=1).sum()))",
                "    return scores",
                "",
                "q = 'What is selective scan and why does it matter?'",
                "q_tokens = np.asarray(list(encoder.query_embed([q]))[0], dtype=np.float32)",
                "q_tokens /= np.linalg.norm(q_tokens, axis=1, keepdims=True).clip(min=1e-9)",
                "scores = maxsim(q_tokens, chunk_token_arrays)",
                "ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)[:5]",
                "for s, c in ranked:",
                "    print(f'  score={s:6.2f}  {c.text[:160]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap as `answer_question`",
                "",
                "Standard cookbook contract. Internally uses MaxSim retrieval; externally identical to every other recipe.",
            ],
            code=[
                "def colbert_search(query: str, top_k: int = 5):",
                "    qt = np.asarray(list(encoder.query_embed([query]))[0], dtype=np.float32)",
                "    qt /= np.linalg.norm(qt, axis=1, keepdims=True).clip(min=1e-9)",
                "    s = maxsim(qt, chunk_token_arrays)",
                "    return sorted(zip(s, chunks), key=lambda x: x[0], reverse=True)[:top_k]",
                "",
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    hits = colbert_search(question, top_k=k)",
                "    contexts = [c.text for _, c in hits]",
                "    answer = client.chat(",
                "        'Use these passages.\\n' + '\\n\\n'.join(contexts) + f'\\nQ: {question}\\nA:'",
                "    )",
                "    return answer, contexts",
                "",
                "ans, _ = answer_question('What is selective scan?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — per-token attention pattern",
                "",
                "For a query, look at the per-query-token MaxSim contributions to the top-1 chunk. Each query token finds its best matching chunk token; summing those is the chunk's score. The shape of those contributions tells you which query tokens carried the chunk into the top-1.",
            ],
            code=[
                "q = 'selective scan in state-space models'",
                "qt = np.asarray(list(encoder.query_embed([q]))[0], dtype=np.float32)",
                "qt /= np.linalg.norm(qt, axis=1, keepdims=True).clip(min=1e-9)",
                "scores = maxsim(qt, chunk_token_arrays)",
                "best_chunk = int(np.argmax(scores))",
                "sims = qt @ chunk_token_arrays[best_chunk].T",
                "print(f'Best chunk index: {best_chunk}  total MaxSim: {scores[best_chunk]:.3f}')",
                "print()",
                "print('Per-query-token MaxSim contribution to the best chunk:')",
                "for q_i in range(min(8, qt.shape[0])):",
                "    contrib = float(sims[q_i].max())",
                "    print(f'  q_token {q_i:2d}: max={contrib:.3f}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — storage comparison",
                "",
                "How much bigger is the multi-vector index than the single-vector equivalent? Multiply per-chunk tokens by dim to get the per-chunk cost.",
            ],
            code=[
                "single_vec_dim = 1024  # Nebius default",
                "single_vec_bytes = single_vec_dim * 4",
                "multi_vec_bytes = sum(arr.size * 4 for arr in chunk_token_arrays) / len(chunk_token_arrays)",
                "print(f'Single-vector per chunk: {single_vec_bytes:,} bytes')",
                "print(f'Multi-vector  per chunk: {int(multi_vec_bytes):,} bytes')",
                "print(f'Ratio: ~{int(multi_vec_bytes / single_vec_bytes)}x')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — MaxSim score distribution",
                "",
                "For one query, plot the score distribution across all chunks. The shape tells you whether ColBERT is confidently ranking (sharp peak) or uncertain (flat curve).",
            ],
            code=[
                "import matplotlib.pyplot as plt",
                "",
                "q = 'selective scan'",
                "qt = np.asarray(list(encoder.query_embed([q]))[0], dtype=np.float32)",
                "qt /= np.linalg.norm(qt, axis=1, keepdims=True).clip(min=1e-9)",
                "scores = sorted(maxsim(qt, chunk_token_arrays), reverse=True)",
                "fig, ax = plt.subplots(figsize=(6, 2.8))",
                "ax.plot(range(1, len(scores) + 1), scores, marker='o')",
                "ax.set_xlabel('Rank')",
                "ax.set_ylabel('MaxSim score')",
                "ax.set_title('Per-chunk MaxSim distribution')",
                "ax.grid(alpha=0.3)",
                "plt.tight_layout()",
                "plt.show()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — does ColBERT change the top-1 on rare-term queries?",
                "",
                "Single-vector dense embedders often miss queries with rare technical terms. ColBERT's per-token matching is supposed to catch them. Compare on a few such queries.",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "dense_vectors = client.embed([c.text for c in chunks])",
                "dense_store = QdrantBackend('dense-cmp', dim=len(dense_vectors[0]))",
                "dense_store.add([c.text for c in chunks], dense_vectors, ids=[c.chunk_id for c in chunks])",
                "",
                "for q in [",
                "    'HiPPO initialization',",
                "    'selective scan',",
                "    'discretization rule',",
                "    'long-range arena',",
                "]:",
                "    dense_top = dense_store.search(client.embed([q])[0], top_k=1)[0]",
                "    colbert_top = colbert_search(q, top_k=1)[0][1]",
                "    same = dense_top.text[:60] == colbert_top.text[:60]",
                "    print(f'  {\"same\":>9s} ' if same else f'  {\"DIFFER\":>9s} ', q)",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a representative Mamba question.",
    ],
    run_code=[
        "q = 'What is HiPPO initialization and why does it matter for selective state-space models?'",
        "ans, ctxs = answer_question(q)",
        "print('=== ColBERT answer ===')",
        "print(ans)",
        "print()",
        "print('Top context preview:')",
        "print(ctxs[0][:300])",
    ],

    comparison_md=[
        "Vanilla single-vector dense vs ColBERT MaxSim on the same question. The interesting case is a question with rare technical terms where dense averaging dilutes the signal.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What is HiPPO initialization and why does it matter for selective state-space models?'",
        "base = vanilla_pipeline(q, corpus='arxiv-mamba', top_k=5)",
        "ours_a, ours_c = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla dense', 'top_preview': base.contexts[0][:140]},",
        "    {'pipeline': 'colbert maxsim', 'top_preview': ours_c[0][:140]},",
        "])",
    ],

    tuning_md=[
        "Five knobs:",
        "",
        "1. **Backbone model.** ColBERT v2 is the workhorse; v2.5 and ColBERTv3 (under development as of 2026) bring quality and storage improvements. For multi-lingual workloads, multilingual ColBERT models exist (Jina ColBERT v2, BGE ColBERT-M3).",
        "2. **Quantisation.** ColBERT v2 introduced residual compression: int8 quantisation cuts storage by 4x with negligible recall loss. Production deployments quantise.",
        "3. **Max tokens per chunk.** Longer chunks mean more token vectors and more MaxSim compute. Cap at 256 tokens per chunk for indexing speed; semantic chunks (Recipe 5) tend to fit nicely.",
        "4. **Shortlist + rerank.** In production, retrieve top-100 with single-vector dense, rerank those 100 with ColBERT MaxSim. Cuts the per-query compute from `n_chunks * n_tokens` to `100 * n_tokens`.",
        "5. **Native multi-vector store.** Qdrant 1.10+ has `MultiVectorConfig` that stores per-token vectors and runs MaxSim in C. Weaviate and Vespa offer the same. Always prefer the native path for production.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Storage shock.** ColBERT indexes are ~30x bigger than single-vector indexes. At a billion chunks, the difference matters; budget for it before deciding to deploy.",
        "- **Latency at the tail.** Pure MaxSim is `O(n_chunks * tokens_per_chunk)` per query in Python. Native multi-vector stores reduce this to ANN-time retrieval, but vanilla implementations are slow without a shortlist.",
        "- **Quality below ColBERT v2.** ColBERT v1 was state-of-the-art in 2020 but is now beaten by larger single-vector models with reranking. The v2 recipe shines; the v1 recipe is just expensive.",
        "",
        "The right production pattern is shortlist with single-vector dense (Qdrant default), rerank with ColBERT MaxSim or a cross-encoder. The cookbook isolates ColBERT here so the technique is visible; recipe 22 shows the composition.",
    ],
)


# =============================================================================
# Auto-Retrieval with Metadata
# =============================================================================
AUTO_RETRIEVAL = Recipe(
    path="recipes/04-retrieval/auto-retrieval-with-metadata.ipynb",
    title="Auto-Retrieval — LLM-Extracted Filters Over Structured Metadata",
    category="retrieval",
    corpus_filter="rust-book",

    theory_problem=[
        "When chunks carry structured metadata (chapter, section, date, author), a query that names one of those fields should narrow retrieval to chunks matching that field. \"In chapter 17, how does async work?\" should not scan all 18 chapters — it should filter to chapter 17 first.",
        "Auto-retrieval is the LLM-driven implementation: an LLM reads the query, extracts a filter spec (a JSON object naming fields and values), and applies it to the vector store's structured-filter API. The chunks pre-filtered by metadata are then ranked by vector similarity. The combination is dramatically more precise than vector-only retrieval.",
    ],
    theory_origin=[
        "Auto-retrieval was packaged by LlamaIndex's `VectorIndexAutoRetriever` in 2023, building on a long IR tradition of structured-filter retrieval that goes back to the SQL-and-keyword search systems of the 1990s. The LLM extraction step is what made it accessible — earlier systems required users to write filter syntax by hand, which broke the natural-language UX of LLM agents.",
        "By 2026 every major vector store supports server-side metadata filters; the LLM extraction layer is the standard pattern for letting users issue natural-language queries that implicitly carry structured constraints. Modern variants use JSON-schema-constrained function calling to get the LLM to produce valid filter objects directly, which makes the extraction step both more reliable and more developer-ergonomic.",
    ],
    theory_landscape=[
        "Metadata-driven retrieval lives in the broader routing family:",
        "",
        "- **Semantic router (Recipe 17).** Pick the right index.",
        "- **Document-summary routing (Recipe 11).** Pick the right document.",
        "- **Auto-retrieval (this recipe).** Pick the right metadata slice within an index.",
        "",
        "Stack them: route to the right index, then auto-retrieve with metadata filters inside that index. Each layer cuts the search space.",
    ],
    theory_when_to_use=[
        "Use auto-retrieval when chunks carry meaningful, query-relevant metadata. Dated documents (filter by year), versioned docs (filter by version), structured corpora (filter by section or chapter), multi-tenant systems (filter by tenant ID).",
        "Skip it when chunks are flat. If metadata is just an internal ID with no user-meaningful semantics, there's nothing to filter on and the LLM extraction is wasted.",
        "Skip it when filter extraction is unreliable. A model that mis-extracts the filter loses the right chunks entirely. Always have a fallback that retrieves without the filter when the filtered set is empty.",
    ],
    theory_intuition=[
        "Four intuitions to carry:",
        "",
        "**Metadata is a hard constraint, embeddings are a soft one.** A filter says \"this chunk MUST come from chapter 17\". The embedding ranks within that constraint. The hard constraint cuts the search space by 50-100x for free.",
        "",
        "**The LLM extracts; you don't write filter syntax.** Users say \"in chapter 17\" and the LLM produces `{chapter: 'ch17-...'}`. Users don't need to know the schema, which is the whole point of natural-language UX.",
        "",
        "**Always validate the extracted filter.** A model may invent a chapter that doesn't exist. Constrain the prompt with the actual list of valid values and reject invented fields client-side.",
        "",
        "**Fall back to unfiltered search.** If the filtered set is empty, don't refuse. Retrieve without the filter and let vector similarity find the right chunks. The fallback is the safety net that makes auto-retrieval safe to deploy.",
    ],

    architecture_mermaid="""
flowchart TB
  Q[Query] --> EX[LLM:<br/>extract filter spec]
  EX --> F{Filter<br/>extracted?}
  F -->|yes| FR[Filtered retrieval]
  F -->|no| UR[Unfiltered retrieval]
  V[(Vector store<br/>with metadata)] --> FR
  V --> UR
  FR --> A[Top-k]
  UR --> A
""",

    references=[
        Reference(
            title="LlamaIndex VectorIndexAutoRetriever",
            url="https://developers.llamaindex.ai/python/framework-api-reference/llama_index/core/indices/vector_store/retrievers/VectorIndexAutoRetriever/",
            kind="docs",
            note="Reference implementation.",
        ),
        Reference(
            title="Qdrant filter conditions",
            url="https://qdrant.tech/documentation/concepts/filtering/",
            kind="docs",
            note="The server-side filter syntax we use.",
        ),
        Reference(
            title="LangChain SelfQueryRetriever",
            url="https://python.langchain.com/docs/how_to/self_query/",
            kind="docs",
            note="LangChain's auto-retrieval cousin.",
        ),
        Reference(
            title="Weaviate Auto-Cut",
            url="https://weaviate.io/blog/distance-thresholds",
            kind="blog",
            note="A different shape of automatic retrieval narrowing.",
        ),
        Reference(
            title="Function-Calling Schemas — OpenAI",
            url="https://platform.openai.com/docs/guides/function-calling",
            kind="docs",
            note="The JSON-schema approach used by some auto-retrievers.",
        ),
        Reference(
            title="Vector + Metadata Hybrid — Pinecone",
            url="https://www.pinecone.io/learn/vector-search-filtering/",
            kind="blog",
            note="The architectural case for hybrid vector + metadata.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load and chunk with metadata",
                "",
                "We need chunks with structured metadata for auto-retrieval to filter on. The Rust book chunks carry `chapter` metadata automatically.",
            ],
            code=[
                "from cookbook.corpora import load_rust_book",
                "from cookbook.chunkers import sentence_window",
                "",
                "docs = list(load_rust_book())",
                "chunks = sentence_window(docs, sentences_per_chunk=4, overlap=1)",
                "print(f'Chunks: {len(chunks)}')",
                "print(f'Sample metadata: {chunks[0].metadata}')",
            ],
        ),
        CodeStep(
            tag="index",
            lead_md=[
                "### Step 2 — Index with metadata payload",
                "",
                "We pass per-chunk metadata to Qdrant so it can filter server-side at query time.",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('auto', dim=len(vectors[0]))",
                "store.add(",
                "    [c.text for c in chunks],",
                "    vectors,",
                "    ids=[c.chunk_id for c in chunks],",
                "    metadatas=[{'chapter': c.metadata.get('chapter'), 'doc_id': c.doc_id} for c in chunks],",
                ")",
                "print(f'Indexed {len(chunks)} chunks with metadata.')",
                "",
                "available_chapters = sorted({c.metadata.get('chapter') for c in chunks if c.metadata.get('chapter')})",
                "print(f'Available chapters: {len(available_chapters)}')",
                "for c in available_chapters[:5]:",
                "    print(f'  {c}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Build the filter-extraction prompt",
                "",
                "The model gets the list of valid chapters and is asked to extract `(chapter, topic)`. Constraining the prompt with valid values prevents hallucinated chapter names.",
            ],
            code=[
                "import json",
                "import re",
                "",
                "FILTER_PROMPT = (",
                "    'You extract a metadata filter for a Rust-book vector store. '",
                '    \'Respond as JSON: {{\"chapter\": \"ch07-00-...\" or null, \"topic\": \"...\"}}. \'',
                "    'Use only chapters from the list. \"topic\" is a paraphrase of the question used for vector similarity.\\n\\n'",
                "    'Available chapters:\\n{chapters}\\n\\nQuestion: {q}'",
                ")",
                "",
                "def extract_filter(question: str) -> dict:",
                "    raw = client.chat(FILTER_PROMPT.format(",
                "        chapters='\\n'.join(available_chapters),",
                "        q=question,",
                "    ))",
                "    m = re.search(r'\\{.*\\}', raw, flags=re.DOTALL)",
                "    if not m:",
                "        return {'chapter': None, 'topic': question}",
                "    try:",
                "        spec = json.loads(m.group())",
                "        return {",
                "            'chapter': spec.get('chapter') if spec.get('chapter') in available_chapters else None,",
                "            'topic': spec.get('topic') or question,",
                "        }",
                "    except json.JSONDecodeError:",
                "        return {'chapter': None, 'topic': question}",
                "",
                "spec = extract_filter('In which chapter does the book discuss the orphan rule for traits?')",
                "print(spec)",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 4 — Retrieve with the extracted filter",
                "",
                "Pass the filter to Qdrant's `query_filter`. We fall back to unfiltered retrieval when extraction returned no chapter constraint.",
            ],
            code=[
                "from qdrant_client.models import Filter, FieldCondition, MatchValue",
                "",
                "def auto_retrieve(question: str, top_k: int = 5):",
                "    spec = extract_filter(question)",
                "    qv = client.embed([spec['topic']])[0]",
                "    if spec['chapter']:",
                "        flt = Filter(must=[FieldCondition(key='chapter', match=MatchValue(value=spec['chapter']))])",
                "        result = store.client.query_points(",
                "            collection_name=store.collection,",
                "            query=qv,",
                "            limit=top_k,",
                "            query_filter=flt,",
                "            with_payload=True,",
                "        ).points",
                "        return [(p.payload.get('text', ''), spec['chapter']) for p in result], spec",
                "    return [(h.text, None) for h in store.search(qv, top_k=top_k)], spec",
                "",
                "hits, spec = auto_retrieve('In which chapter does the book discuss the orphan rule for traits?')",
                "print(f'Extracted: {spec}')",
                "for text, chapter in hits[:3]:",
                "    print(f'  [{chapter}] {text[:140]}')",
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
                "PROMPT = (",
                "    'Use only the passages below to answer the question.\\n\\n'",
                "    'Passages:\\n{context}\\n\\nQuestion: {question}\\nAnswer:'",
                ")",
                "",
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    hits, _ = auto_retrieve(question, top_k=k)",
                "    contexts = [t for t, _ in hits]",
                "    return client.chat(PROMPT.format(context='\\n\\n'.join(contexts), question=question)), contexts",
                "",
                "ans, _ = answer_question('In which chapter does the book describe the orphan rule for trait implementations?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — extraction behaviour on different queries",
                "",
                "Some queries name a chapter, some don't. The extractor should produce a chapter filter only when the query supports it.",
            ],
            code=[
                "for q in [",
                "    'In which chapter does the book discuss the orphan rule for traits?',",
                "    'How does the borrow checker work?',",
                "    'What does Chapter 5 say about structs?',",
                "    'Compare Rc and Arc.',",
                "]:",
                "    spec = extract_filter(q)",
                "    print(f'  {q[:60]:60s}  -> chapter={spec[\"chapter\"]!r}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — filtered top-3 vs unfiltered top-3",
                "",
                "When a filter applies, the chunks should all come from that chapter. Confirm.",
            ],
            code=[
                "q = 'How are smart pointers used?'",
                "spec = extract_filter(q)",
                "print(f'Extracted: {spec}')",
                "hits, _ = auto_retrieve(q, top_k=5)",
                "for text, chapter in hits[:5]:",
                "    print(f'  [{chapter!r}] {text[:80]}...')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what happens with an invalid chapter name?",
                "",
                "If the LLM hallucinates a chapter, our filter validator should reject it and fall back to unfiltered retrieval.",
            ],
            code=[
                "# Force a hallucinated chapter to confirm fallback works",
                "spec = {'chapter': 'ch99-fake-chapter', 'topic': 'borrow checker'}",
                "validated = spec['chapter'] if spec['chapter'] in available_chapters else None",
                "print(f'Hallucinated chapter \"{spec[\"chapter\"]}\" -> validated={validated}')",
                "print('Fall-back to unfiltered retrieval would have triggered.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — recall@5 with vs without auto-retrieval",
                "",
                "On a battery of chapter-targeting queries, auto-retrieval should improve recall.",
            ],
            code=[
                "import pandas as pd",
                "labelled = [",
                "    'In which chapter does the book discuss async/await?',",
                "    'In Chapter 10, what does the book say about generics?',",
                "    'In which chapter does the book discuss Rc and Arc?',",
                "    'How do lifetimes work?',  # no chapter constraint",
                "    'When should I use a HashMap?',",
                "]",
                "rows = []",
                "for q in labelled:",
                "    spec = extract_filter(q)",
                "    rows.append({'query': q[:55], 'chapter_extracted': spec['chapter']})",
                "pd.DataFrame(rows)",
            ],
        ),
    ],

    run_md=[
        "End-to-end query with auto-retrieval.",
    ],
    run_code=[
        "for q in [",
        "    'In which chapter does the book describe async and await?',",
        "    'How does the book handle the orphan rule for traits?',",
        "    'When should I prefer Arc over Rc?',",
        "]:",
        "    ans, _ = answer_question(q)",
        "    print(f'Q: {q}')",
        "    print(f'  -> {ans[:200]}')",
        "    print()",
    ],

    comparison_md=[
        "Vanilla baseline vs auto-retrieval. The interesting case is a query that names a specific chapter — vanilla scans everything, auto-retrieval narrows first.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'In which chapter does the book describe the orphan rule for trait implementations?'",
        "base = vanilla_pipeline(q, corpus='rust-book', top_k=5)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'auto-retrieval', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Five knobs in priority order:",
        "",
        "1. **Filter schema.** Define carefully and stick to it. Adding fields later means re-indexing every chunk to include the new field, which is the most expensive operation in a vector DB.",
        "2. **Extraction prompt.** Constrain valid values explicitly by listing them in the prompt. \"Use only chapters from the list below\" is not optional, otherwise the model hallucinates plausible-but-invalid chapter names.",
        "3. **Extractor model.** Mid-tier is plenty. Extraction is a constrained task — bigger models don't help much, and the latency saving from a smaller model is worth it.",
        "4. **Fallback strategy.** Empty filtered result → unfiltered retrieval. Always. Refusing on empty filtered sets is the most common newbie mistake in auto-retrieval deployments.",
        "5. **Compose with hybrid (Recipe 18).** Filter + dense + BM25 is the most precise stack for structured corpora. Each layer narrows what the next layer searches.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Hallucinated filter values.** Mitigated by validating against a known list; always validate.",
        "- **Over-filtering.** A model that extracts too many constraints can produce an empty result. Soft-fail by falling back to unfiltered.",
        "- **Missing metadata.** If chunks don't carry the field the user is asking about, extraction is wasted effort.",
        "",
        "Compose with semantic routing (Recipe 17) for cross-index dispatch, with hybrid retrieval (Recipe 18) for dense + sparse over the filtered slice, and with reranking (Recipe 22) for the final top-k ordering.",
    ],
)


# =============================================================================
# MMR
# =============================================================================
MMR = Recipe(
    path="recipes/04-retrieval/mmr-diversity.ipynb",
    title="MMR — Maximum Marginal Relevance for Diverse Top-K",
    category="retrieval",
    corpus_filter="wikipedia-superconductors",

    theory_problem=[
        "Vanilla top-k retrieval returns the k chunks most similar to the query. Those k chunks are also often similar to *each other* — near-duplicates, paraphrases of the same paragraph. The model sees five chunks that say the same thing instead of five chunks that cover different facets of the answer.",
        "Maximum Marginal Relevance (MMR) trades a little query-similarity for a lot of diversity. It picks the most-similar chunk first, then for each subsequent pick maximises `λ * query_similarity - (1-λ) * max_similarity_to_chosen`. The result: a top-k that covers the answer space rather than repeats one part of it.",
    ],
    theory_origin=[
        "MMR was introduced by Carbonell and Goldstein at CMU in 1998 (\"The Use of MMR, Diversity-Based Reranking\"). It long predated dense retrieval; in the IR community it was a standard re-ranking technique for search engines. RAG inherited it via LangChain's `max_marginal_relevance_search` (2023) and LlamaIndex's `MMRPostProcessor`.",
        "By 2026 MMR is a default postprocessor for any retrieval call where the top-k is going to be summarised or composed into an answer. The tuning parameter `lambda_mult` (default 0.5) balances relevance vs diversity; production systems set it once and rarely revisit.",
    ],
    theory_landscape=[
        "Among diversity-aware retrieval techniques in this cookbook:",
        "",
        "- **MMR (this recipe).** Greedy reranking for diversity. Cheap, predictable, one tunable parameter.",
        "- **Clustering-based pickers.** Group top-N candidates, pick one per cluster. Older approach, rare in modern RAG.",
        "- **Listwise LLM rerankers (Recipe 23).** Let an LLM reorder for diversity and relevance jointly. More flexible, more expensive.",
        "- **Cross-encoder + MMR.** Stack: cross-encoder finds the most relevant top-N, MMR picks a diverse subset. The cookbook recommends this composition in production.",
        "",
        "MMR is the cheapest of these and the most predictable. Listwise LLM is the most flexible but pays per-query LLM cost. Clustering is rare in modern RAG because greedy MMR is essentially free.",
    ],
    theory_when_to_use=[
        "Use MMR when the downstream task summarises or composes across the top-k. Anywhere your answer benefits from diverse evidence — research synthesis, comparison Q&A, list-making, broad-topic exploration where the user wants coverage rather than depth.",
        "Skip it when the downstream is extractive QA. There you want the single best chunk; diversity in top-k is wasteful and may push the right chunk out of the result set.",
        "Skip it when query similarity is your only signal. If users want \"the most relevant chunk\", MMR can return something the user would have ranked lower in favour of diversity.",
    ],
    theory_intuition=[
        "Four intuitions:",
        "",
        "**Diversity comes from penalising similarity to chosen chunks.** Each new pick is scored against the chosen set, not just against the query. Picks similar to already-chosen are penalised so the algorithm rewards complementary information.",
        "",
        "**Greedy is enough.** MMR is a greedy algorithm; it picks one chunk at a time. There's no global optimum being computed; the local pick at each step is the algorithm and the implementation is essentially a loop.",
        "",
        "**`λ` controls the trade-off.** λ=1 is pure relevance (no diversity). λ=0 is pure diversity (random spread). λ=0.5 is the cookbook default and works on most corpora without further tuning.",
        "",
        "**Over-retrieve before MMR.** Greedy diversification picks one item at a time from a pool. The bigger the pool, the more material MMR has to express diversity. Always over-retrieve.",
    ],

    architecture_mermaid="""
flowchart LR
  Q[Query] --> R[Retrieve top-N<br/>candidates]
  R --> MMR[For each pick:<br/>maximise λ·sim_query<br/>− 1-λ ·max_sim_chosen]
  MMR --> T[Top-k diverse]
""",

    references=[
        Reference(
            title="The Use of MMR, Diversity-Based Reranking (Carbonell and Goldstein, 1998)",
            url="https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf",
            kind="paper",
            note="The original paper.",
        ),
        Reference(
            title="LangChain max_marginal_relevance_search",
            url="https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.utils.maximal_marginal_relevance.html",
            kind="docs",
            note="Reference implementation.",
        ),
        Reference(
            title="LlamaIndex MMRPostProcessor",
            url="https://developers.llamaindex.ai/python/framework-api-reference/llama_index/core/postprocessor/MetadataReplacementPostProcessor/",
            kind="docs",
            note="LlamaIndex's variant.",
        ),
        Reference(
            title="Sentence-Transformers MMR utilities",
            url="https://www.sbert.net/examples/applications/retrieve_rerank/README.html",
            kind="docs",
            note="Useful reference for the implementation.",
        ),
        Reference(
            title="Pinecone — Search with diversity",
            url="https://www.pinecone.io/learn/series/rag/diversity-rerank/",
            kind="blog",
            note="Production-perspective on MMR.",
        ),
        Reference(
            title="Listwise LLM rerankers (Recipe 23)",
            url="https://arxiv.org/abs/2305.02156",
            kind="paper",
            note="Alternative approach to similar problem.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build the index",
                "",
                "Wikipedia superconductors is a good corpus for MMR — many articles cover overlapping topics, so vanilla top-k returns near-duplicate chunks.",
            ],
            code=[
                "from cookbook.corpora import load_wikipedia_superconductors",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_wikipedia_superconductors())",
                "chunks = sentence_window(docs, sentences_per_chunk=4)",
                "texts = [c.text for c in chunks]",
                "vectors = client.embed(texts)",
                "store = QdrantBackend('mmr', dim=len(vectors[0]))",
                "store.add(texts, vectors, ids=[c.chunk_id for c in chunks])",
                "vec_by_id = {c.chunk_id: v for c, v in zip(chunks, vectors)}",
                "print(f'Indexed {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 2 — Use the cookbook's MMR implementation",
                "",
                "`cookbook.retrievers.mmr` does the greedy reranking. We give it the query vector, candidate vectors, candidate hits, and the lambda multiplier.",
            ],
            code=[
                "from cookbook.retrievers import mmr",
                "",
                "q = 'What discoveries set the milestones of high-temperature superconductivity?'",
                "qv = client.embed([q])[0]",
                "pool = store.search(qv, top_k=30)",
                "pool_vecs = [vec_by_id[h.doc_id] for h in pool]",
                "diverse = mmr(qv, pool_vecs, pool, lambda_mult=0.5, top_k=5)",
                "for h in diverse:",
                "    print(f'  {h.text[:160]}')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 3 — Compare vanilla top-5 to MMR top-5",
                "",
                "Vanilla top-5 will often include three near-duplicates. MMR top-5 will look more diverse.",
            ],
            code=[
                "print('--- Vanilla top-5 ---')",
                "for h in pool[:5]:",
                "    print(f'  {h.score:.3f}  {h.text[:140]}')",
                "print()",
                "print('--- MMR top-5 ---')",
                "for h in diverse:",
                "    print(f'  {h.text[:140]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Sweep `λ` to see the relevance/diversity trade-off",
                "",
                "λ=1 reproduces vanilla top-5. λ=0 maximises diversity at the cost of relevance. The cookbook default is 0.5.",
            ],
            code=[
                "import pandas as pd",
                "rows = []",
                "for lam in (0.0, 0.3, 0.5, 0.7, 1.0):",
                "    picks = mmr(qv, pool_vecs, pool, lambda_mult=lam, top_k=5)",
                "    rows.append({'lambda': lam, 'top1_preview': picks[0].text[:60], 'top5_doc_ids': [p.doc_id for p in picks]})",
                "pd.DataFrame(rows)",
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
                "PROMPT = (",
                "    'Use only the passages below to answer the question.\\n\\n'",
                "    'Passages:\\n{context}\\n\\nQuestion: {question}\\nAnswer:'",
                ")",
                "",
                "def answer_question(question: str, k: int = 5, lam: float = 0.5) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    pool = store.search(qv, top_k=20)",
                "    pool_vecs = [vec_by_id[h.doc_id] for h in pool]",
                "    diverse = mmr(qv, pool_vecs, pool, lambda_mult=lam, top_k=k)",
                "    contexts = [h.text for h in diverse]",
                "    return client.chat(PROMPT.format(context='\\n\\n'.join(contexts), question=question)), contexts",
                "",
                "ans, _ = answer_question('What are the major eras of superconductivity research?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — pairwise similarity within top-5",
                "",
                "Vanilla top-5 chunks should have high pairwise similarity; MMR top-5 should have lower. We confirm with a heatmap.",
            ],
            code=[
                "import numpy as np",
                "import matplotlib.pyplot as plt",
                "",
                "vanilla_picks = pool[:5]",
                "mmr_picks = mmr(qv, pool_vecs, pool, lambda_mult=0.5, top_k=5)",
                "vanilla_v = np.asarray([vec_by_id[h.doc_id] for h in vanilla_picks])",
                "vanilla_v /= np.linalg.norm(vanilla_v, axis=1, keepdims=True).clip(min=1e-9)",
                "mmr_v = np.asarray([vec_by_id[h.doc_id] for h in mmr_picks])",
                "mmr_v /= np.linalg.norm(mmr_v, axis=1, keepdims=True).clip(min=1e-9)",
                "",
                "fig, axes = plt.subplots(1, 2, figsize=(8, 3))",
                "axes[0].imshow(vanilla_v @ vanilla_v.T, vmin=0, vmax=1, cmap='viridis')",
                "axes[0].set_title('Vanilla top-5 pairwise')",
                "axes[1].imshow(mmr_v @ mmr_v.T, vmin=0, vmax=1, cmap='viridis')",
                "axes[1].set_title('MMR top-5 pairwise')",
                "plt.tight_layout()",
                "plt.show()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — how many unique doc_ids appear?",
                "",
                "Diversity often shows up as different source documents. We count unique doc_ids in each top-5.",
            ],
            code=[
                "vanilla_ids = {h.doc_id for h in pool[:5]}",
                "mmr_ids = {h.doc_id for h in mmr(qv, pool_vecs, pool, lambda_mult=0.5, top_k=5)}",
                "print(f'Vanilla unique doc_ids: {len(vanilla_ids)}')",
                "print(f'MMR unique doc_ids:     {len(mmr_ids)}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — does MMR help on factoid queries?",
                "",
                "MMR doesn't always help. For factoid queries, the single best chunk is what you want, and diversity doesn't matter.",
            ],
            code=[
                "factoid_q = 'In what year did Heike Kamerlingh Onnes discover superconductivity?'",
                "fq_vec = client.embed([factoid_q])[0]",
                "fq_pool = store.search(fq_vec, top_k=20)",
                "fq_vecs = [vec_by_id[h.doc_id] for h in fq_pool]",
                "print('--- Vanilla top-3 ---')",
                "for h in fq_pool[:3]:",
                "    print(f'  {h.text[:120]}')",
                "print()",
                "print('--- MMR top-3 ---')",
                "for h in mmr(fq_vec, fq_vecs, fq_pool, lambda_mult=0.5, top_k=3):",
                "    print(f'  {h.text[:120]}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost",
                "",
                "MMR is essentially free at this scale. The whole rerank is matrix operations over 20-50 vectors.",
            ],
            code=[
                "import time",
                "t0 = time.perf_counter()",
                "_ = mmr(qv, pool_vecs, pool, lambda_mult=0.5, top_k=5)",
                "dt = (time.perf_counter() - t0) * 1000",
                "print(f'MMR over 30 candidates took {dt:.1f} ms.')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a synthesis query.",
    ],
    run_code=[
        "ans, _ = answer_question('What are the major eras and key discoveries that shaped superconductivity research?')",
        "print('=== MMR-diverse answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla vs MMR on a synthesis query. Vanilla often gives the model five repeated facts; MMR gives five complementary ones.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What are the major eras and key discoveries that shaped superconductivity research?'",
        "base = vanilla_pipeline(q, corpus='wikipedia-superconductors', top_k=5)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'mmr', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Five knobs in priority order:",
        "",
        "1. **`lambda_mult`.** Default 0.5. Higher (0.7) for more relevance, lower (0.3) for more diversity. Sweep on synthesis queries.",
        "2. **Candidate pool size.** We use 20-30. Bigger pool gives MMR more material to choose from; smaller pool may not have enough diversity to express.",
        "3. **Where MMR sits.** Cookbook puts MMR after dense retrieval. Some implementations put it after cross-encoder reranking (Recipe 22) — the rerank picks the most relevant, MMR picks a diverse subset.",
        "4. **When to use it.** Synthesis queries yes; factoid queries no. Adaptive routing (Recipe 26) can decide per query whether to apply MMR.",
        "5. **Similarity metric for diversity.** Cosine is the cookbook default; max-inner-product also works. Choose to match your embedding normalisation.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Wrong query type.** MMR on factoid retrieval can hurt: it may demote the single best chunk in favour of diverse-but-weaker alternatives.",
        "- **Too few candidates.** With a 5-candidate pool, MMR can't diversify. Always over-retrieve.",
        "- **λ at extremes.** λ=0 is mostly noise; λ=1 is vanilla. Stay in [0.3, 0.7].",
        "",
        "Compose with cross-encoder reranking (Recipe 22): use the cross-encoder to find the top-N most relevant, then MMR over those N. Compose with adaptive routing (Recipe 26): apply MMR only on synthesis queries.",
    ],
)


RECIPES = [HYBRID, COLBERT, AUTO_RETRIEVAL, MMR]
