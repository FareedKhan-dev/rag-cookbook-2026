"""Category 5 — Reranking & Fusion."""
from __future__ import annotations

from authoring import CodeStep, Recipe, Reference


# =============================================================================
# Cross-Encoder Rerank
# =============================================================================
CROSS_ENCODER = Recipe(
    path="recipes/05-reranking-and-fusion/cross-encoder-rerank.ipynb",
    title="Cross-Encoder Reranking — The Default Production Quality Lift",
    category="reranking-and-fusion",
    corpus_filter="arxiv-mamba",

    theory_problem=[
        "Bi-encoders (the embedding models we've used in every recipe so far) score queries and chunks independently. The query's vector is produced without seeing any chunk; each chunk's vector is produced without seeing the query. The model never gets to ask \"how well does this query line up against this chunk?\" — it only computes their vectors separately and compares with cosine.",
        "Cross-encoders score query and chunk *jointly*. The transformer sees the concatenated `(query, chunk)` pair and outputs a single relevance score. Joint scoring catches signals bi-encoders miss — semantic mismatches, contradictions, partial-match nuances. The trade-off is speed: cross-encoders can't precompute an index. The pattern: shortlist with a fast bi-encoder, rerank the top-N with a slow cross-encoder.",
    ],
    theory_origin=[
        "Cross-encoder reranking has been a standard IR technique since the late 2010s; sentence-transformers shipped `CrossEncoder` in 2019. The RAG community adopted it from the start. By 2026 the leaders are `bge-reranker-v2-m3` (open weights), Cohere Rerank 4 (hosted), Voyage Rerank 2.5 (hosted), and Jina Reranker v3 (listwise variant).",
        "The technique has stayed remarkably stable for five years because the fundamental shape — joint scoring of (query, chunk) pairs — is what makes it work. Most innovation since 2020 has been in training data (more diverse domains) and architecture (smaller distilled models for cheaper inference).",
        "**This notebook ships with an LLM-based pointwise scorer** so it runs without a heavy model download. The shape — pointwise (query, chunk) scoring then re-sort — is identical to a real cross-encoder. Production teams swap `cross_encoder_rerank()` to call `sentence-transformers.CrossEncoder` with `BAAI/bge-reranker-v2-m3` or use a hosted reranker like Cohere Rerank; the surrounding pipeline does not change.",
    ],
    theory_landscape=[
        "Reranking sits between retrieval and generation. Three levels of expressivity:",
        "",
        "- **Bi-encoder dense (Recipe 2).** Fastest. Index-friendly. Cosine over precomputed vectors.",
        "- **Cross-encoder (this recipe).** Slower. Per-query. Joint (query, chunk) scoring.",
        "- **LLM-as-reranker (Recipe 23).** Slowest. Most flexible. Reorders a candidate list as one task.",
        "",
        "Production patterns: shortlist with bi-encoder, rerank with cross-encoder, optionally apply LLM rerank on top-5. Each stage cuts the search space and adds a layer of quality.",
    ],
    theory_when_to_use=[
        "Use cross-encoder reranking on any production RAG system where quality matters more than latency. The pattern adds 50-150 ms of latency and a few cents of cost per query for a 5-15 point recall@5 improvement on hard queries — a near-universal win.",
        "Skip it when the shortlist is already perfect. A FAQ Q&A system with crisp questions and clear answers may not benefit from another layer.",
        "Skip it when latency is hard-capped under 200 ms. Cross-encoders are not free; budget them carefully and consider a distilled model if latency is tight.",
    ],
    theory_intuition=[
        "Four intuitions:",
        "",
        "**Joint scoring sees nuance.** A bi-encoder represents query and chunk as separate points in vector space. A cross-encoder reads them together. Joint reading lets the model notice that a chunk mentions the query's entity but contradicts the question, or matches the topic but answers a different sub-question.",
        "",
        "**Cross-encoders can't be indexed.** Every (query, chunk) pair must be scored at query time. That's why the shortlist+rerank pattern exists — you can't run a cross-encoder over a billion chunks.",
        "",
        "**Shortlist size is the latency knob.** Reranking 20 candidates is fast. Reranking 500 is slow. Tune the shortlist to fit your latency budget.",
        "",
        "**Rerankers don't generate.** A reranker only ranks; it never produces text. That's why they can be small, distilled, and fast. Don't reach for a frontier model when bge-reranker-v2-m3 suffices.",
    ],

    architecture_mermaid="""
flowchart LR
  Q[Query] --> BE[Bi-encoder<br/>retrieval]
  BE --> SL[Shortlist top-N<br/>e.g. 30]
  SL --> CE[Cross-encoder<br/>scores each pair]
  CE --> R[Reordered top-k<br/>e.g. 5]
  R --> G[Generator]
""",

    references=[
        Reference(
            title="Sentence-Transformers CrossEncoder documentation",
            url="https://www.sbert.net/examples/applications/cross-encoder/README.html",
            kind="docs",
            note="The canonical Python implementation.",
        ),
        Reference(
            title="BAAI BGE Reranker v2-m3",
            url="https://huggingface.co/BAAI/bge-reranker-v2-m3",
            kind="repo",
            note="Open-weight leader; we use this model.",
        ),
        Reference(
            title="Cohere Rerank documentation",
            url="https://docs.cohere.com/docs/rerank-overview",
            kind="docs",
            note="Hosted leader on quality.",
        ),
        Reference(
            title="Jina Reranker v3",
            url="https://jina.ai/news/jina-reranker-v3-deep-research-multilingual-and-listwise/",
            kind="blog",
            note="Listwise variant; cousin to Recipe 23.",
        ),
        Reference(
            title="Pinecone reranking guide",
            url="https://www.pinecone.io/learn/series/rag/rerankers/",
            kind="blog",
            note="Production-perspective overview.",
        ),
        Reference(
            title="Anthropic Contextual Retrieval",
            url="https://www.anthropic.com/news/contextual-retrieval",
            kind="blog",
            note="The contextual-chunks + BM25 + reranker stack relies heavily on this step.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build the index",
                "",
                "Standard setup on the Mamba paper. We need a working bi-encoder index to shortlist from.",
            ],
            code=[
                "from cookbook.corpora import load_arxiv_mamba",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_arxiv_mamba())",
                "chunks = sentence_window(docs, sentences_per_chunk=4)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('rerank', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 2 — Shortlist with the bi-encoder",
                "",
                "Standard dense retrieval to get a candidate pool of 30. The shortlist size is the main latency knob.",
            ],
            code=[
                "q = 'Why is the parallel scan implementation crucial for Mamba on GPUs?'",
                "qv = client.embed([q])[0]",
                "shortlist = store.search(qv, top_k=30)",
                "print(f'Shortlisted {len(shortlist)} candidates.')",
                "print('Top-3 by bi-encoder:')",
                "for h in shortlist[:3]:",
                "    print(f'  {h.score:.3f}  {h.text[:140]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Rerank with the cross-encoder",
                "",
                "`cookbook.rerankers.cross_encoder_rerank` loads the cross-encoder model, scores each (query, chunk) pair, returns the reordered list.",
            ],
            code=[
                "from cookbook.rerankers import cross_encoder_rerank",
                "",
                "reranked = cross_encoder_rerank(q, shortlist, top_k=5, chat=client.chat)",
                "print('Top-5 after cross-encoder rerank:')",
                "for h in reranked:",
                "    print(f'  {h.score:.3f}  {h.text[:140]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Wrap as `answer_question`",
                "",
                "Standard contract. Shortlist + rerank inside.",
            ],
            code=[
                "PROMPT = (",
                "    'Use only the passages below to answer the question.\\n\\n'",
                "    'Passages:\\n{context}\\n\\nQuestion: {question}\\nAnswer:'",
                ")",
                "",
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    short = store.search(qv, top_k=30)",
                "    top = cross_encoder_rerank(question, short, top_k=k, chat=client.chat)",
                "    contexts = [h.text for h in top]",
                "    return client.chat(PROMPT.format(context='\\n\\n'.join(contexts), question=question)), contexts",
                "",
                "ans, _ = answer_question('Why is HiPPO initialization useful for state-space models?')",
                "print(ans)",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 5 — Compare bi-encoder top-5 to reranked top-5",
                "",
                "On hard queries the order changes substantially. Note which chunks moved.",
            ],
            code=[
                "q = 'Why is the parallel scan implementation crucial for Mamba on GPUs?'",
                "qv = client.embed([q])[0]",
                "shortlist = store.search(qv, top_k=30)",
                "reranked = cross_encoder_rerank(q, shortlist, top_k=5, chat=client.chat)",
                "shortlist_ids = [h.doc_id for h in shortlist[:5]]",
                "reranked_ids = [h.doc_id for h in reranked]",
                "print(f'Bi-encoder top-5 ids: {shortlist_ids}')",
                "print(f'Reranked   top-5 ids: {reranked_ids}')",
                "moved_in = set(reranked_ids) - set(shortlist_ids)",
                "moved_out = set(shortlist_ids) - set(reranked_ids)",
                "print(f'Moved into top-5 by rerank: {moved_in}')",
                "print(f'Moved out of top-5 by rerank: {moved_out}')",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — how do scores compare?",
                "",
                "Bi-encoder scores are cosine similarities (0-1 typically). Cross-encoder scores are logits — different scale. Both are monotonic; ranks tell you everything.",
            ],
            code=[
                "import pandas as pd",
                "rows = []",
                "for i, h in enumerate(shortlist[:10]):",
                "    bi_score = h.score",
                "    rows.append({'bi_rank': i+1, 'bi_score': bi_score, 'doc_id': h.doc_id[:20]})",
                "",
                "rerank_top10 = cross_encoder_rerank(q, shortlist, top_k=10, chat=client.chat)",
                "ce_scores = {h.doc_id: h.score for h in rerank_top10}",
                "",
                "for r in rows:",
                "    r['ce_score'] = ce_scores.get(r['doc_id'], None)",
                "pd.DataFrame(rows)",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what's the latency cost?",
                "",
                "Measure how long the cross-encoder rerank takes on a 30-candidate shortlist.",
            ],
            code=[
                "import time",
                "t0 = time.perf_counter()",
                "_ = cross_encoder_rerank(q, shortlist, top_k=5, chat=client.chat)",
                "ce_ms = (time.perf_counter() - t0) * 1000",
                "",
                "t0 = time.perf_counter()",
                "_ = store.search(qv, top_k=30)",
                "be_ms = (time.perf_counter() - t0) * 1000",
                "",
                "print(f'Bi-encoder retrieval (30 hits): {be_ms:.1f} ms')",
                "print(f'Cross-encoder rerank   (30 pairs): {ce_ms:.1f} ms')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — sweep shortlist size",
                "",
                "Bigger shortlist gives the cross-encoder more material but slows reranking linearly. Sweep.",
            ],
            code=[
                "import time, pandas as pd",
                "rows = []",
                "for n in (10, 20, 40, 80):",
                "    short = store.search(qv, top_k=n)",
                "    t0 = time.perf_counter()",
                "    top = cross_encoder_rerank(q, short, top_k=5, chat=client.chat)",
                "    dt = (time.perf_counter() - t0) * 1000",
                "    rows.append({'shortlist_size': n, 'rerank_ms': round(dt, 1), 'top1_doc_id': top[0].doc_id[:30]})",
                "pd.DataFrame(rows)",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — recall@5 on the eval slice",
                "",
                "Loose recall proxy comparing bi-encoder-only vs reranked.",
            ],
            code=[
                "from cookbook.corpora import load_eval_questions",
                "qs = [q for q in load_eval_questions() if q['corpus'] == 'arxiv-mamba'][:8]",
                "def recall_fn(retr_fn):",
                "    hits = 0",
                "    for row in qs:",
                "        retr = retr_fn(row['question'])",
                "        gold = [w.lower() for w in row['answer'].split() if len(w) >= 4]",
                "        if any(any(w[:6] in r.text.lower() for w in gold) for r in retr):",
                "            hits += 1",
                "    return hits / max(1, len(qs))",
                "be = lambda question: store.search(client.embed([question])[0], top_k=5)",
                "ce = lambda question: cross_encoder_rerank(question, store.search(client.embed([question])[0], top_k=30), top_k=5)",
                "print(f'bi-encoder recall@5 = {recall_fn(be):.2f}')",
                "print(f'reranked   recall@5 = {recall_fn(ce):.2f}')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a representative question.",
    ],
    run_code=[
        "q = 'In plain language, what does selective scan do that earlier SSMs could not?'",
        "ans, _ = answer_question(q)",
        "print('=== Reranked answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla baseline (bi-encoder only) vs reranked.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'In plain language, what does selective scan do that earlier SSMs could not?'",
        "base = vanilla_pipeline(q, corpus='arxiv-mamba', top_k=5)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'reranked', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Six knobs in priority order:",
        "",
        "1. **Shortlist size.** Default 30. Larger improves recall at linear latency cost.",
        "2. **Reranker model.** bge-reranker-v2-m3 is the open-weight default. Cohere Rerank 4 and Voyage Rerank 2.5 are stronger hosted options.",
        "3. **Where to put the rerank.** After dense, after hybrid, or as a fall-back when dense top-1 score is below a confidence threshold.",
        "4. **Cache reranker scores.** When you reuse a query, the rerank scores are cached automatically; treat cache size like an embedding cache.",
        "5. **GPU for the reranker.** A modest GPU (T4 / L4) keeps rerank latency under 50ms for 100 pairs in production.",
        "6. **Chunk-length cap.** Reranker latency scales with chunk length. Cap chunks at ~512 tokens to keep latency predictable across queries.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Shortlist misses the answer.** The cross-encoder can only rerank what the bi-encoder gave it. Use HyDE (Recipe 13) or multi-query fusion (Recipe 14) upstream to widen the funnel.",
        "- **Reranker out-of-domain.** A reranker trained on web QA may underperform on legal or medical text. Pick a domain-appropriate model when one exists.",
        "- **Latency spike on long chunks.** Reranker latency scales with chunk length. Cap chunk size at ~512 tokens to keep latency predictable.",
        "",
        "Compose with hybrid (Recipe 18): hybrid produces the shortlist, cross-encoder picks the top-k. Compose with MMR (Recipe 21): rerank picks the most relevant top-N, MMR picks a diverse subset.",
    ],
)


# =============================================================================
# Listwise LLM Rerank
# =============================================================================
LISTWISE = Recipe(
    path="recipes/05-reranking-and-fusion/listwise-llm-rerank.ipynb",
    title="Listwise LLM Reranking — RankGPT-style Reordering",
    category="reranking-and-fusion",
    corpus_filter="sec-10k-pltr",

    theory_problem=[
        "Cross-encoder rerankers score one (query, chunk) pair at a time. They are powerful but blind to global context — they can't see what the other top-N candidates look like. When two candidates compete for the same factual role, or when one candidate's relevance only makes sense given another's absence, pointwise scoring is structurally limited.",
        "Listwise LLM reranking asks a single language model to reorder the entire candidate set in one pass. The model sees all N candidates together, can compare them, and can reason about consistency, contradiction, and complementarity. The trade-off: one LLM call per query, with prompt sizes that grow with N.",
    ],
    theory_origin=[
        "Listwise reranking via LLMs was crystallised by RankGPT (Sun et al., 2023). The original paper showed that a single GPT-3.5 call reordering 20 candidates beat cross-encoder rerankers on TREC and BEIR by 2-6 points. The technique fits the natural strengths of LLMs — global reasoning over text — and bypasses the awkwardness of training a cross-encoder for every new domain or language.",
        "By 2026 the technique has been productised. Jina Reranker v3 is listwise by default. Cohere Rerank 4 offers a listwise variant. Open implementations live in `llama-index` and `langchain`. The cookbook implements it directly so the mechanism is visible — under the hood it's just a prompt that asks the LLM to output a JSON array of indices in rank order.",
    ],
    theory_when_to_use=[
        "Use listwise LLM reranking when quality justifies the cost. Research assistants, expert systems, tools where the user expects the top-1 to be obviously right. Production systems handling high-stakes queries (legal, medical, financial) often pay the listwise cost.",
        "Skip it when latency or cost matters more than the marginal quality. The technique adds 500-2000 ms per query and a real LLM bill.",
        "Skip it on very short shortlists. With 5 candidates a cross-encoder is already nearly optimal; the listwise call adds little.",
    ],
    theory_landscape=[
        "Listwise reranking is the most expensive of the three reranking tiers:",
        "",
        "- **Cross-encoder (Recipe 22).** Pointwise; fast; per-pair scoring.",
        "- **Listwise LLM (this recipe).** Full-list; slow; global reasoning.",
        "- **Listwise dedicated models (Jina Reranker v3).** Same shape, smaller dedicated model, cheaper than full LLM.",
        "",
        "Many production systems stack cross-encoder + listwise LLM: cross-encoder picks the top-20 from the shortlist, listwise picks the top-5 from the 20. Each tier adds quality at a known cost.",
    ],
    theory_intuition=[
        "Four intuitions:",
        "",
        "**Global context catches contradictions.** When two candidates contradict each other, the LLM can prefer the one consistent with other top candidates. Pointwise scorers can't see that consistency signal.",
        "",
        "**Prompt size grows with N.** Each candidate adds chunks to the prompt. The cookbook caps candidates to 20 and chunk previews to 400 chars; production stacks may go further.",
        "",
        "**A smaller LLM is enough.** The reranker doesn't need to *answer* — only to order. Use a fast small model (gpt-4o-mini, Llama-3.3-8B-instant, Qwen-2.5-7B). Save your frontier-tier budget for the final answer.",
        "",
        "**Listwise rerankers handle redundancy gracefully.** If three candidates are near-duplicates, the LLM tends to rank one near the top and demote the others. Pointwise scorers would rank all three high and waste top-k slots.",
    ],

    architecture_mermaid="""
flowchart LR
  Q[Query] --> S[Shortlist<br/>top-N candidates]
  S --> P[LLM:<br/>read all N,<br/>output order JSON]
  P --> R[Reordered top-k]
  R --> G[Generator]
""",

    references=[
        Reference(
            title="Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent (Sun et al., 2023)",
            url="https://arxiv.org/abs/2304.09542",
            kind="paper",
            note="The RankGPT paper.",
        ),
        Reference(
            title="Jina Reranker v3 — listwise multilingual",
            url="https://jina.ai/news/jina-reranker-v3-deep-research-multilingual-and-listwise/",
            kind="blog",
            note="Production listwise reranker.",
        ),
        Reference(
            title="LlamaIndex RankGPTRerank",
            url="https://developers.llamaindex.ai/python/examples/node_postprocessor/rankGPT/",
            kind="docs",
            note="Reference implementation.",
        ),
        Reference(
            title="Cohere Rerank documentation",
            url="https://docs.cohere.com/docs/rerank-overview",
            kind="docs",
            note="The pointwise leader; some endpoints offer listwise.",
        ),
        Reference(
            title="Cross-encoder rerank (Recipe 22)",
            url="https://www.sbert.net/examples/applications/cross-encoder/README.html",
            kind="docs",
            note="The faster cousin.",
        ),
        Reference(
            title="MMR (Recipe 21)",
            url="https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf",
            kind="paper",
            note="Diversity-aware alternative.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build the index",
                "",
                "Standard SEC 10-K setup. Multi-part queries are listwise reranking's home turf.",
            ],
            code=[
                "from cookbook.corpora import load_sec_10k",
                "from cookbook.chunkers import fixed_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_sec_10k())",
                "chunks = fixed_window(docs, target_tokens=320, overlap_tokens=32)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('llm-rerank', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 2 — Shortlist with the bi-encoder",
                "",
                "Top-20 candidates for the listwise to reorder.",
            ],
            code=[
                "q = 'What competitive risk factors does Palantir attribute to consulting firms?'",
                "qv = client.embed([q])[0]",
                "shortlist = store.search(qv, top_k=20)",
                "print(f'Shortlisted {len(shortlist)} candidates.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Apply listwise LLM rerank",
                "",
                "`cookbook.rerankers.listwise_llm_rerank` packages the prompt, parses the JSON output, and produces the reordered list.",
            ],
            code=[
                "from cookbook.rerankers import listwise_llm_rerank",
                "",
                "reranked = listwise_llm_rerank(q, shortlist, client.chat, top_k=5)",
                "print('Listwise top-5:')",
                "for h in reranked:",
                "    print(f'  {h.text[:160]}')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 4 — Compare bi-encoder, cross-encoder, listwise",
                "",
                "Three rankings, one chart. The orderings should differ in interesting ways — listwise often promotes chunks that are *complementary* to the top-1 rather than redundant with it.",
            ],
            code=[
                "from cookbook.rerankers import cross_encoder_rerank",
                "",
                "be_ids = [h.doc_id for h in shortlist[:5]]",
                "ce_ids = [h.doc_id for h in cross_encoder_rerank(q, shortlist, top_k=5, chat=client.chat)]",
                "lw_ids = [h.doc_id for h in listwise_llm_rerank(q, shortlist, client.chat, top_k=5)]",
                "",
                "import pandas as pd",
                "rows = [",
                "    {'rank': r+1, 'bi_encoder': be_ids[r][:20] if r < len(be_ids) else None,",
                "     'cross_encoder': ce_ids[r][:20] if r < len(ce_ids) else None,",
                "     'listwise_llm': lw_ids[r][:20] if r < len(lw_ids) else None}",
                "    for r in range(5)",
                "]",
                "pd.DataFrame(rows)",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap as `answer_question`",
                "",
                "Cookbook contract.",
            ],
            code=[
                "PROMPT = (",
                "    'Use only the passages below to answer the question.\\n\\n'",
                "    'Passages:\\n{context}\\n\\nQuestion: {question}\\nAnswer:'",
                ")",
                "",
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    short = store.search(qv, top_k=20)",
                "    top = listwise_llm_rerank(question, short, client.chat, top_k=k)",
                "    contexts = [h.text for h in top]",
                "    return client.chat(PROMPT.format(context='\\n\\n'.join(contexts), question=question)), contexts",
                "",
                "ans, _ = answer_question('What customer concentration metrics does Palantir disclose?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — latency comparison",
                "",
                "Listwise costs one LLM call per query. Cross-encoder costs N pair scorings. Bi-encoder is essentially free. Measure.",
            ],
            code=[
                "import time",
                "from cookbook.rerankers import cross_encoder_rerank",
                "",
                "t0 = time.perf_counter()",
                "_ = store.search(qv, top_k=20)",
                "be_ms = (time.perf_counter() - t0) * 1000",
                "",
                "t0 = time.perf_counter()",
                "_ = cross_encoder_rerank(q, shortlist, top_k=5, chat=client.chat)",
                "ce_ms = (time.perf_counter() - t0) * 1000",
                "",
                "t0 = time.perf_counter()",
                "_ = listwise_llm_rerank(q, shortlist, client.chat, top_k=5)",
                "lw_ms = (time.perf_counter() - t0) * 1000",
                "",
                "print(f'Bi-encoder retrieval : {be_ms:7.1f} ms')",
                "print(f'Cross-encoder rerank : {ce_ms:7.1f} ms')",
                "print(f'Listwise LLM rerank  : {lw_ms:7.1f} ms')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — does listwise change the top-1?",
                "",
                "Run several queries and count how often listwise picks a different top-1 than cross-encoder.",
            ],
            code=[
                "battery = [",
                "    'What competitive risks does Palantir disclose?',",
                "    'How does Palantir generate revenue from AIP?',",
                "    'What governance structures protect founders?',",
                "    'What cybersecurity controls does the filing describe?',",
                "]",
                "same, diff = 0, 0",
                "for q in battery:",
                "    qv = client.embed([q])[0]",
                "    short = store.search(qv, top_k=20)",
                "    ce_top = cross_encoder_rerank(q, short, top_k=1, chat=client.chat)[0]",
                "    lw_top = listwise_llm_rerank(q, short, client.chat, top_k=1)[0]",
                "    if ce_top.doc_id == lw_top.doc_id:",
                "        same += 1",
                "    else:",
                "        diff += 1",
                "print(f'Same top-1: {same}/{len(battery)}; Different: {diff}/{len(battery)}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does the LLM prompt look like?",
                "",
                "Read the prompt we send to the LLM. It's just a numbered list of passages plus a directive.",
            ],
            code=[
                "rendered = '\\n'.join(f'[{i}] {h.text[:200]}' for i, h in enumerate(shortlist[:10]))",
                "preview = (",
                "    'Reorder the following passages from most to least relevant for the query.\\n'",
                "    f'Query: {q}\\n\\nPassages:\\n{rendered}\\n\\n'",
                "    'Respond as JSON: {\"order\": [3, 1, 7, ...]} with passage indices.'",
                ")",
                "print(preview[:1200])",
                "print('...')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost",
                "",
                "Listwise adds one full LLM call. Cache helps but only on repeated queries.",
            ],
            code=[
                "from cookbook import _cache",
                "before = _cache.stats()['entries']",
                "_ = answer_question('What does the company say about consulting competitors?')",
                "after = _cache.stats()['entries']",
                "print(f'New cache entries: {after - before}')",
                "print('Rough breakdown:')",
                "print('  1   query embed')",
                "print('  1   listwise rerank LLM call')",
                "print('  1   final-answer LLM call')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a representative SEC question.",
    ],
    run_code=[
        "q = 'What does Palantir say about its competition with consulting firms and large platform companies?'",
        "ans, _ = answer_question(q)",
        "print('=== Listwise-reranked answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla vs listwise.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What does Palantir say about its competition with consulting firms and large platform companies?'",
        "base = vanilla_pipeline(q, corpus='sec-10k-pltr', top_k=5)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'listwise-llm', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Six knobs in priority order:",
        "",
        "1. **Shortlist size.** Default 20. Higher gives the LLM more material but inflates prompt length and per-query cost.",
        "2. **Reranker model.** A small fast model is fine. Reserve your best model for the final answer — the reranker only needs to order, not to answer.",
        "3. **Prompt structure.** RankGPT-style numbered passages plus JSON output. The cookbook uses this format; it's robust across providers.",
        "4. **Snippet length per passage.** We use 400 chars. Longer gives the LLM more to compare; shorter keeps the prompt small and the LLM call cheap.",
        "5. **Composition.** Stack on top of cross-encoder rerank: cross-encoder picks top-20 from 100, listwise picks top-5 from 20.",
        "6. **JSON fallback.** Always have a fallback for malformed JSON. The cookbook returns the original ordering on parse failure.",
    ],

    discussion_md=[
        "Three failure modes you'll meet in production:",
        "",
        "- **JSON parse failures.** The LLM occasionally returns malformed JSON. Cookbook's parser falls back to the original ranking on parse failure, which is safer than refusing.",
        "- **Prompt length blow-up.** With N=50 and long chunks the prompt is huge, slow, and expensive. Trim snippets aggressively.",
        "- **Cost.** Two LLM calls per query (rerank + answer). For high-volume systems, listwise may not pay off — use a dedicated listwise reranker model like Jina v3 instead.",
        "",
        "Compose with cross-encoder rerank (Recipe 22) for a two-tier rerank. Compose with hybrid retrieval (Recipe 18) for the shortlist. Compose with adaptive routing (Recipe 26) to only apply listwise on queries that benefit.",
    ],
)


RECIPES = [CROSS_ENCODER, LISTWISE]
