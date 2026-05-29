"""Category 3 — Query Transformation."""
from __future__ import annotations

from authoring import CodeStep, Recipe, Reference


# =============================================================================
# HyDE
# =============================================================================
HYDE = Recipe(
    path="recipes/03-query-transformation/hypothetical-document-embeddings.ipynb",
    title="HyDE — Hypothetical Document Embeddings",
    category="query-transformation",
    corpus_filter="arxiv-mamba",

    theory_problem=[
        "User queries are short. Documents are long. Their embeddings live in different parts of vector space, and cosine similarity often fails to bridge that gap. A query that asks \"why does this model scale linearly?\" rarely lands near a passage that *answers* that question — because answers are written in a different register, with more entities, with the topic stated as fact rather than as a question.",
        "HyDE (Hypothetical Document Embeddings) fixes the asymmetry by having the LLM write a fake answer first, then embedding that. The fake answer is in answer-register, so it lives close to real answers in embedding space. Retrieve against the hypothetical, return the actual passages it matches, generate the real answer. The first call is wasted compute; the retrieval lift typically justifies it.",
    ],
    theory_origin=[
        "HyDE was published by Luyu Gao and colleagues at CMU in late 2022 (\"Precise Zero-Shot Dense Retrieval without Relevance Labels\"). The paper showed gains of 4–8 points on zero-shot retrieval benchmarks where no labelled training data was available. The mechanism was simple enough that everyone implemented it within months; by 2024 it was a standard baseline in any RAG comparison.",
        "By 2026 HyDE is rarely deployed alone — it tends to be one branch of multi-query fusion (Recipe 14) or as a fallback when standard retrieval misses. But it remains the cleanest demonstration of why query transformation matters, and the implementation is twenty lines of code.",
    ],
    theory_landscape=[
        "Three approaches to closing the query/document register gap:",
        "",
        "- **HyDE (this recipe).** Hallucinate an answer, embed that.",
        "- **Multi-query rewriting (Recipe 14).** Paraphrase the question N times, fuse the rankings.",
        "- **Step-back abstraction (Recipe 15).** Ask a more general version of the question.",
        "",
        "All three can be combined. The cookbook's `cookbook.retrievers.reciprocal_rank_fusion` is the standard glue. Production systems often run HyDE alongside the original query and fuse — best of both worlds.",
    ],
    theory_when_to_use=[
        "Use HyDE when your queries are abstract or when the corpus uses a different vocabulary than your users. Zero-shot retrieval (no labelled query/document pairs to fine-tune against) is HyDE's home turf. Cross-domain Q&A bots, exploratory search interfaces, anywhere the user's phrasing doesn't match the corpus's phrasing.",
        "Skip HyDE when your queries already look like answers — factoid lookup over a Wikipedia-like corpus, FAQ Q&A. There the original query is already in the right register.",
        "Skip it when latency is tight. HyDE doubles the LLM cost (one call to hallucinate, one to answer) and adds 200–500 ms before retrieval even starts.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**The hallucination doesn't need to be correct.** It needs to be in the right *register* — to look like the kind of text the real answer would be. Even if the fake claims wrong things, it usually retrieves the right chunks because chunks live in answer-space.",
        "",
        "**Embedders cluster by register, not by truth.** Two paraphrases of the same answer are close in embedding space whether or not they are true. HyDE exploits this — the fake answer's embedding clusters with real answers regardless of fact-correctness.",
        "",
        "**Combine with the raw query for safety.** A pure-HyDE pipeline can drift if the model hallucinates badly. Run both — raw query and HyDE query — and fuse the rankings. You get HyDE's lift on hard queries and the original query's safety on easy ones.",
    ],

    architecture_mermaid="""
flowchart LR
  Q[Question] --> H[LLM hallucinates<br/>fake answer]
  H --> E[Embed the<br/>fake answer]
  E --> R[Retrieve top-k<br/>real chunks]
  R --> G[LLM answers from<br/>real chunks]
""",

    references=[
        Reference(
            title="Precise Zero-Shot Dense Retrieval without Relevance Labels (Gao et al., 2022)",
            url="https://arxiv.org/abs/2212.10496",
            kind="paper",
            note="The HyDE paper.",
        ),
        Reference(
            title="LlamaIndex HyDE Query Transform",
            url="https://developers.llamaindex.ai/python/examples/query_transformations/HyDEQueryTransformDemo/",
            kind="docs",
            note="Reference implementation.",
        ),
        Reference(
            title="LangChain HyDE retriever",
            url="https://python.langchain.com/docs/integrations/retrievers/hyde/",
            kind="docs",
            note="Cousin implementation.",
        ),
        Reference(
            title="Multi-query RAG-fusion (Recipe 14)",
            url="https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1",
            kind="blog",
            note="Stack HyDE inside RAG-fusion for the best of both.",
        ),
        Reference(
            title="Step-back prompting (Recipe 15)",
            url="https://arxiv.org/abs/2310.06117",
            kind="paper",
            note="Different way to abstract the query.",
        ),
        Reference(
            title="Query2Doc — Microsoft",
            url="https://arxiv.org/abs/2303.07678",
            kind="paper",
            note="Microsoft's contemporary variant of the same idea.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build a chunked index of the Mamba paper",
                "",
                "Standard setup. The Mamba survey is good for HyDE because it's technical — the gap between casual question vocabulary and the paper's vocabulary is wide.",
            ],
            code=[
                "from cookbook.corpora import load_arxiv_mamba",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_arxiv_mamba())",
                "chunks = sentence_window(docs, sentences_per_chunk=4, overlap=1)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('hyde', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — The hypothetical-answer prompt",
                "",
                "Short and confident. The prompt asks the model to write what a passage answering the question would look like — not to actually answer it, but to mimic a passage's tone.",
            ],
            code=[
                "HYDE_PROMPT = (",
                "    'Write a 4-sentence excerpt from a technical paper that would directly answer the question below. '",
                "    'Use a confident, declarative tone — as if you are quoting a paper. '",
                "    'Do not hedge or say you are uncertain.\\n\\n'",
                "    'Question: {q}\\n\\nExcerpt:'",
                ")",
                "",
                "q = 'Why does selective scan recover content-based reasoning that earlier SSMs lacked?'",
                "fake = client.chat(HYDE_PROMPT.format(q=q))",
                "print(fake)",
            ],
            expected_output_md=[
                "Look at the register. The fake answer reads like a paragraph from the paper — declarative claims, technical vocabulary, no questions. That register is what makes it retrieve well.",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 3 — Retrieve against the hypothetical",
                "",
                "Embed the fake answer, search Qdrant. We package the whole flow as `hyde_retrieve()` so the rest of the recipe is short.",
            ],
            code=[
                "def hyde_retrieve(question: str, top_k: int = 5):",
                "    fake = client.chat(HYDE_PROMPT.format(q=question))",
                "    qv = client.embed([fake])[0]",
                "    return store.search(qv, top_k=top_k), fake",
                "",
                "hits, fake = hyde_retrieve(q)",
                "print('HYPOTHETICAL ANSWER:')",
                "print(fake[:400])",
                "print()",
                "print('TOP RETRIEVED CHUNKS:')",
                "for h in hits[:3]:",
                "    print(f'  {h.score:.3f}  {h.text[:160]}')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 4 — Compare HyDE retrieval to direct retrieval",
                "",
                "Run the same query both ways. The interesting cases are queries where HyDE finds a chunk that direct retrieval missed.",
            ],
            code=[
                "raw_qv = client.embed([q])[0]",
                "raw_hits = store.search(raw_qv, top_k=5)",
                "",
                "print('--- RAW QUERY retrieval ---')",
                "for h in raw_hits[:3]:",
                "    print(f'  {h.score:.3f}  {h.text[:140]}')",
                "print()",
                "print('--- HyDE retrieval ---')",
                "for h in hits[:3]:",
                "    print(f'  {h.score:.3f}  {h.text[:140]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap as `answer_question`",
                "",
                "Standard contract. The HyDE retrieval feeds real chunks to the answer LLM — the model never sees the fake answer except as a routing aid.",
            ],
            code=[
                "PROMPT = (",
                "    'Use only the passages below to answer the question. '",
                "    'If they do not contain the answer, say so plainly.\\n\\n'",
                "    'Passages:\\n{context}\\n\\nQuestion: {question}\\nAnswer:'",
                ")",
                "",
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    hits, _ = hyde_retrieve(question, top_k=k)",
                "    contexts = [h.text for h in hits]",
                "    return client.chat(PROMPT.format(context='\\n\\n'.join(contexts), question=question)), contexts",
                "",
                "ans, _ = answer_question('What complexity advantage motivates state-space models?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — read three hypothetical answers",
                "",
                "Quality of the hallucinated text matters. A vague hallucination retrieves noisy chunks; a specific one retrieves sharp chunks.",
            ],
            code=[
                "for question in [",
                "    'Explain HiPPO initialization in one paragraph.',",
                "    'How does Mamba compare to RWKV on long contexts?',",
                "    'Why is parallel scan important for GPU efficiency?',",
                "]:",
                "    fake = client.chat(HYDE_PROMPT.format(q=question))",
                "    print(f'Q: {question}')",
                "    print(f'  fake: {fake[:240]}...')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — does HyDE change the top-1?",
                "",
                "Across a battery, count how often HyDE retrieves a different top-1 than the raw query. Often the top-1 is the same (both lands on the right chunk), but on hard queries HyDE finds chunks raw retrieval misses.",
            ],
            code=[
                "battery = [",
                "    'What is selective scan?',",
                "    'How does Mamba scale linearly with sequence length?',",
                "    'Compare H3 to S4 architecturally.',",
                "    'Why are SSMs cheaper to run than transformers?',",
                "    'What does the paper say about associative recall?',",
                "]",
                "same, diff = 0, 0",
                "for q in battery:",
                "    raw_top = store.search(client.embed([q])[0], top_k=1)[0]",
                "    hyde_hits, _ = hyde_retrieve(q, top_k=1)",
                "    if raw_top.doc_id == hyde_hits[0].doc_id:",
                "        same += 1",
                "    else:",
                "        diff += 1",
                "print(f'Same top-1: {same}/{len(battery)}')",
                "print(f'Different : {diff}/{len(battery)}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — recall@5 with and without HyDE",
                "",
                "Loose recall proxy on the eval slice.",
            ],
            code=[
                "from cookbook.corpora import load_eval_questions",
                "qs = [q for q in load_eval_questions() if q['corpus'] == 'arxiv-mamba'][:8]",
                "def recall_raw():",
                "    hits = 0",
                "    for q in qs:",
                "        retr = store.search(client.embed([q['question']])[0], top_k=5)",
                "        gold = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "        if any(any(w[:6] in r.text.lower() for w in gold) for r in retr):",
                "            hits += 1",
                "    return hits / max(1, len(qs))",
                "def recall_hyde():",
                "    hits = 0",
                "    for q in qs:",
                "        retr, _ = hyde_retrieve(q['question'], top_k=5)",
                "        gold = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "        if any(any(w[:6] in r.text.lower() for w in gold) for r in retr):",
                "            hits += 1",
                "    return hits / max(1, len(qs))",
                "print(f'raw  recall@5 = {recall_raw():.2f}')",
                "print(f'hyde recall@5 = {recall_hyde():.2f}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost of HyDE vs raw",
                "",
                "HyDE adds one LLM call per query. We measure the cache delta to confirm.",
            ],
            code=[
                "from cookbook import _cache",
                "before = _cache.stats()['entries']",
                "_ = answer_question('When should I use HiPPO over alternative initializations?')",
                "after = _cache.stats()['entries']",
                "print(f'New cache entries: {after - before}')",
                "print('Rough breakdown:')",
                "print('  1 hypothetical-answer LLM call')",
                "print('  1 embed of the hypothetical')",
                "print('  1 embed of the original query (already cached from earlier cells)')",
                "print('  1 final-answer LLM call')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a representative question.",
    ],
    run_code=[
        "ans, ctxs = answer_question('In plain language, why is Mamba cheaper to run than an equivalent transformer at long context?')",
        "print('=== HyDE answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla baseline (raw query retrieval) vs HyDE on the same question.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'In plain language, why is Mamba cheaper to run than an equivalent transformer at long context?'",
        "base = vanilla_pipeline(q, corpus='arxiv-mamba', top_k=5)",
        "ours_a, ours_c = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'hyde', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Five knobs:",
        "",
        "1. **Hypothetical-answer prompt.** The single most impactful lever. \"Write a paragraph from a paper\" produces different embeddings than \"summarize the answer\" — pick the register your corpus actually has.",
        "2. **Length of the hallucination.** Three-four sentences is the cookbook default. Longer means more text to embed and slightly different vector position; usually no quality gain past five sentences.",
        "3. **Hallucinator model.** Smaller models still produce useful hallucinations if the prompt is strong. A Llama-3.3-8B is fine for HyDE; reserve your best model for the final-answer call.",
        "4. **Combine with the raw query.** Fuse `dense(raw_q)` and `dense(hyde_q)` with RRF (Recipe 14) for robustness against bad hallucinations.",
        "5. **Cache the hallucination.** With the cookbook cache enabled, repeated queries reuse the same hallucinated answer. The first call is paid for; later calls are free.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Hallucination drifts off-domain.** The model writes a plausible-sounding answer that points at the wrong topic. The retrieval then retrieves about the wrong topic. Detect with a sanity check (does the fake answer contain query keywords?).",
        "- **Domain mismatch.** Your corpus is news articles; HyDE writes academic-paper-style fakes; retrieval is awkward. Tune the prompt to match the corpus register.",
        "- **Cost.** Two LLM calls per query, often with a long hallucination input. If you have tight cost budgets, HyDE may not pay for itself; multi-query fusion (Recipe 14) is usually a better choice.",
        "",
        "Compose with multi-query fusion (Recipe 14) — fuse HyDE retrieval with N paraphrase retrievals. Compose with reranking (Recipe 22) — HyDE pulls more chunks in, a reranker picks the best.",
    ],
)


# =============================================================================
# Multi-Query RAG-Fusion
# =============================================================================
MULTI_QUERY = Recipe(
    path="recipes/03-query-transformation/multi-query-rag-fusion.ipynb",
    title="Multi-Query RAG-Fusion — N Rewrites + RRF",
    category="query-transformation",
    corpus_filter="wikipedia-superconductors",

    theory_problem=[
        "A single query has one perspective. The Meissner-effect chunk that uses the phrase \"magnetic flux expulsion\" doesn't rank well for a query phrased \"how do superconductors push out magnetic fields\". They mean the same thing, but embedding cosine doesn't quite agree.",
        "Multi-query rewriting asks the LLM to write N paraphrases. Each paraphrase retrieves a slightly different chunk set. Fusing the N rankings with Reciprocal Rank Fusion gives a final ranking that benefits from every paraphrase's strengths. The technique is sometimes called RAG-Fusion; the implementation is twenty lines.",
    ],
    theory_origin=[
        "Multi-query rewriting predates RAG by years (it's how IR researchers handle query variance) but became a standard RAG technique in early 2024 after Adrian Raudaschl's RAG-Fusion blog post went viral. LlamaIndex and LangChain both shipped multi-query retrievers in that window. The RRF fusion step comes from Cormack et al. 2009 and is what makes the combination robust without per-corpus tuning — sum `1/(k+rank)` across rankings, no per-corpus weight learning required.",
        "By 2026 multi-query is a default in many production systems, with N=3 to N=5 as common settings. The cost is N+1 retrievals plus one LLM call to produce the paraphrases; the win is a few points of recall across the eval set. Modern implementations parallelise the N retrievals, which makes the pattern essentially free in latency even at N=5.",
    ],
    theory_landscape=[
        "Three query-transformation strategies (Recipes 13–17):",
        "",
        "- **HyDE (Recipe 13).** One hallucinated answer, embed that.",
        "- **Multi-query fusion (this recipe).** N paraphrases, fuse rankings.",
        "- **Step-back (Recipe 15).** One more-abstract query, retrieve principles.",
        "",
        "Composition: multi-query fusion is the chassis; HyDE and step-back can be branches inside it. Many systems run `[raw_query, hyde_query, step_back_query]` and fuse all three under one RRF. The cookbook factors them as separate recipes for clarity; production stacks blend.",
    ],
    theory_when_to_use=[
        "Use multi-query whenever a single query phrasing isn't enough. Open-domain Q&A, exploratory search, anywhere users phrase questions inconsistently with the corpus's vocabulary. The technique is robust — it rarely hurts.",
        "Skip it when your queries are short, structured, and the corpus matches their vocabulary. Type-ahead search over product catalogues doesn't benefit much.",
        "Skip it when latency is critical and you cannot parallelise. N+1 retrievals are sequential by default; parallelising helps but adds engineering. RRF needs all rankings before fusing, so it's a barrier in the latency budget.",
    ],
    theory_intuition=[
        "Four intuitions to carry with you:",
        "",
        "**Each paraphrase finds different chunks.** A paraphrase that emphasises one keyword retrieves chunks where that keyword dominates. Three paraphrases emphasising different angles cover more of the answer space than the original query alone could.",
        "",
        "**RRF is parameter-free.** Cormack's `1/(k+rank)` with `k=60` works on every fusion you'll ever do. Don't fiddle with it; the algorithm is robust to `k` choice in a wide band.",
        "",
        "**Diminishing returns past N=5.** Three paraphrases catch most of the gain. Five is generous. Ten is wasteful — you're paying for retrievals that contribute nothing new because the additional paraphrases overlap heavily.",
        "",
        "**The rewriter is doing the work.** If your rewriter writes lazy paraphrases that just reorder words, fusion gains nothing. Tune the rewriter prompt before tuning N.",
    ],

    architecture_mermaid="""
flowchart TB
  Q[Original query] --> RW[LLM:<br/>write N paraphrases]
  RW --> P1[Paraphrase 1]
  RW --> P2[Paraphrase 2]
  RW --> P3[Paraphrase 3]
  Q --> R0[Retrieve top-k]
  P1 --> R1[Retrieve top-k]
  P2 --> R2[Retrieve top-k]
  P3 --> R3[Retrieve top-k]
  R0 --> F[Reciprocal<br/>Rank Fusion]
  R1 --> F
  R2 --> F
  R3 --> F
  F --> G[Top-k fused → LLM]
""",

    references=[
        Reference(
            title="RAG-Fusion — A New Approach (Adrian Raudaschl, 2024)",
            url="https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1",
            kind="blog",
            note="The blog post that popularised the name.",
        ),
        Reference(
            title="LangChain Multi Query Retriever",
            url="https://python.langchain.com/docs/how_to/MultiQueryRetriever/",
            kind="docs",
            note="Reference implementation.",
        ),
        Reference(
            title="LlamaIndex Query Transform Cookbook",
            url="https://developers.llamaindex.ai/python/examples/query_transformations/query_transform_cookbook/",
            kind="docs",
            note="Multi-query plus several cousins.",
        ),
        Reference(
            title="Reciprocal Rank Fusion (Cormack et al., 2009)",
            url="https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf",
            kind="paper",
            note="The fusion technique we use.",
        ),
        Reference(
            title="HyDE (Recipe 13)",
            url="https://arxiv.org/abs/2212.10496",
            kind="paper",
            note="A cousin technique; often composed inside fusion.",
        ),
        Reference(
            title="Step-back prompting (Recipe 15)",
            url="https://arxiv.org/abs/2310.06117",
            kind="paper",
            note="Another query-transformation flavour worth composing.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build the index",
                "",
                "Standard Wikipedia superconductors setup.",
            ],
            code=[
                "from cookbook.corpora import load_wikipedia_superconductors",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_wikipedia_superconductors())",
                "chunks = sentence_window(docs, sentences_per_chunk=5, overlap=1)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('fusion', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Generate N paraphrases",
                "",
                "One LLM call; ask for N numbered paraphrases. The prompt explicitly asks for different angles so the paraphrases retrieve different chunks.",
            ],
            code=[
                "REWRITE_PROMPT = (",
                "    'Write {n} different paraphrases of the question, each phrased to surface a different aspect or use different vocabulary. '",
                "    'Number them, one per line. Do not include any preamble.\\n\\n'",
                "    'Question: {q}'",
                ")",
                "",
                "def fan_out(question: str, n: int = 4) -> list[str]:",
                "    raw = client.chat(REWRITE_PROMPT.format(n=n, q=question))",
                "    rewrites = []",
                "    for line in raw.splitlines():",
                "        line = line.strip()",
                "        if '.' in line[:4]:",
                "            line = line.split('.', 1)[1].strip()",
                "        if line:",
                "            rewrites.append(line)",
                "    return [question] + rewrites[:n]",
                "",
                "rewrites = fan_out('How does the Meissner effect distinguish a superconductor from a perfect conductor?', n=4)",
                "for i, r in enumerate(rewrites):",
                "    print(f'  {i}: {r}')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 3 — Retrieve for each paraphrase",
                "",
                "Standard top-10 retrieval, one per paraphrase. We over-retrieve relative to what we'll keep, because RRF promotes consensus and we want enough candidates for the fusion to be meaningful.",
            ],
            code=[
                "rankings = [store.search(client.embed([q])[0], top_k=10) for q in rewrites]",
                "print(f'Built {len(rankings)} rankings.')",
                "print('Top-1 chunk for each paraphrase:')",
                "for i, ranking in enumerate(rankings):",
                "    print(f'  {i}: {ranking[0].doc_id}  score={ranking[0].score:.3f}')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 4 — Reciprocal Rank Fusion",
                "",
                "We use `cookbook.retrievers.reciprocal_rank_fusion`. Sums `1/(k+rank)` across rankings; the highest-scoring chunks are those that ranked well across multiple paraphrases.",
            ],
            code=[
                "from cookbook.retrievers import reciprocal_rank_fusion",
                "",
                "fused = reciprocal_rank_fusion(rankings, top_k=5)",
                "print('Fused top-5:')",
                "for h in fused:",
                "    print(f'  score={h.score:.4f}  {h.text[:160]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap as `answer_question`",
                "",
                "Standard contract. The fan-out and RRF happen inside.",
            ],
            code=[
                "PROMPT = (",
                "    'Use only the passages below to answer the question. '",
                "    'If they do not contain the answer, say so plainly.\\n\\n'",
                "    'Passages:\\n{context}\\n\\nQuestion: {question}\\nAnswer:'",
                ")",
                "",
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    queries = fan_out(question, n=4)",
                "    rankings = [store.search(client.embed([q])[0], top_k=10) for q in queries]",
                "    fused = reciprocal_rank_fusion(rankings, top_k=k)",
                "    contexts = [h.text for h in fused]",
                "    return client.chat(PROMPT.format(context='\\n\\n'.join(contexts), question=question)), contexts",
                "",
                "ans, _ = answer_question('What is the role of Cooper pairs in BCS theory?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — paraphrase quality",
                "",
                "Read three sets of paraphrases. Good paraphrases use different vocabulary. Bad paraphrases just rearrange words.",
            ],
            code=[
                "for q in [",
                "    'How does flux pinning enable Type-II superconductor levitation?',",
                "    'When was YBCO discovered?',",
                "    'What is the connection between BCS theory and the Meissner effect?',",
                "]:",
                "    print(f'Q: {q}')",
                "    for r in fan_out(q, n=3)[1:]:",
                "        print(f'  - {r}')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — how much do the rankings overlap?",
                "",
                "If paraphrase rankings agree completely, fusion adds nothing. The interesting case is partial overlap — fusion promotes the chunks that appear in many rankings.",
            ],
            code=[
                "q = 'How does the Meissner effect distinguish a superconductor from a perfect conductor?'",
                "qs = fan_out(q, n=4)",
                "rankings = [store.search(client.embed([qq])[0], top_k=5) for qq in qs]",
                "from collections import Counter",
                "top1_ids = Counter(r[0].doc_id for r in rankings)",
                "print('Top-1 distribution across rankings:')",
                "for doc_id, count in top1_ids.most_common():",
                "    print(f'  {count}x: {doc_id}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — sweep N (number of paraphrases)",
                "",
                "Diminishing returns past N=4 on most corpora. Measure on the eval slice.",
            ],
            code=[
                "from cookbook.corpora import load_eval_questions",
                "qs = [q for q in load_eval_questions() if q['corpus'] == 'wikipedia-superconductors'][:6]",
                "import pandas as pd",
                "rows = []",
                "for n in (1, 2, 3, 4, 5):",
                "    hits = 0",
                "    for q in qs:",
                "        queries = fan_out(q['question'], n=n)",
                "        rankings = [store.search(client.embed([qq])[0], top_k=10) for qq in queries]",
                "        fused = reciprocal_rank_fusion(rankings, top_k=5)",
                "        gold = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "        if any(any(w[:6] in h.text.lower() for w in gold) for h in fused):",
                "            hits += 1",
                "    rows.append({'n_paraphrases': n, 'recall@5': hits / max(1, len(qs))})",
                "pd.DataFrame(rows)",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost",
                "",
                "Fan-out is 1 LLM call + N retrievals. Each retrieval is 1 embedding + 1 search. We measure cache entries before and after one query.",
            ],
            code=[
                "from cookbook import _cache",
                "before = _cache.stats()['entries']",
                "_ = answer_question('What is critical temperature?')",
                "after = _cache.stats()['entries']",
                "print(f'New cache entries: {after - before}')",
                "print('Rough breakdown for n=4:')",
                "print('  1 fan-out LLM call')",
                "print('  5 query embeds (raw + 4 paraphrases)')",
                "print('  1 final-answer LLM call')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a representative query.",
    ],
    run_code=[
        "ans, ctxs = answer_question('How does the Meissner effect distinguish a superconductor from a perfect conductor?')",
        "print('=== Multi-query fusion answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla baseline (single query) vs multi-query fusion.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'How does the Meissner effect distinguish a superconductor from a perfect conductor?'",
        "base = vanilla_pipeline(q, corpus='wikipedia-superconductors', top_k=5)",
        "ours_a, ours_c = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'multi-query fusion', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Five knobs in priority order:",
        "",
        "1. **N (paraphrase count).** 3–4 is the sweet spot. Higher pays diminishing returns and inflates cost. Lower may not cover the query space adequately.",
        "2. **Rewriter prompt.** Ask explicitly for *different angles*. Without this, the model produces near-duplicates that all retrieve the same chunks and waste calls.",
        "3. **Top-k per retrieval.** We use 10. Higher catches more recall, slower RRF aggregation. Sweep on a held-out eval to find your sweet spot.",
        "4. **RRF `k` constant.** Default 60. Rarely worth tuning — the algorithm is robust to `k` in [30, 100].",
        "5. **Rewriter model.** A cheap fast model is fine. Paraphrasing is easy; don't waste your best model on this step.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Bad paraphrases.** The rewriter produces near-duplicates or off-topic rewordings. Re-tune the prompt.",
        "- **Cost balloon.** N=10 with 10 chunks each is 100 retrievals. Parallelise; otherwise latency suffers.",
        "- **Diminishing returns invisible without measurement.** Always sweep N on a held-out eval before committing.",
        "",
        "Compose with HyDE (Recipe 13) — one of the paraphrases can be a HyDE hallucination. Compose with step-back (Recipe 15) — another can be an abstracted version. Compose with reranking (Recipe 22) — the fused top-20 is a great input to a cross-encoder.",
    ],
)


# =============================================================================
# Step-Back Abstraction
# =============================================================================
STEP_BACK = Recipe(
    path="recipes/03-query-transformation/step-back-abstraction.ipynb",
    title="Step-Back Abstraction — Retrieve Principles, Then Specifics",
    category="query-transformation",
    corpus_filter="arxiv-mamba",

    theory_problem=[
        "A specific reasoning question (\"Given selective scan, why is HiPPO initialization still useful?\") often retrieves narrow passages and misses the background that makes the answer make sense. The model needs both: the specific passage that mentions HiPPO *and* the conceptual passage that explains why initialization matters at all.",
        "Step-back prompting (DeepMind, 2023) abstracts the specific question into a more general one (\"What role does initialization play in state-space models?\"). Retrieving the abstracted query pulls in principles. Combining principles with the specific query gives the model the reasoning ingredients it lacked.",
    ],
    theory_origin=[
        "Step-back was published by DeepMind in late 2023 (\"Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models\"). The paper showed 4–7 point improvements on MMLU-Phys and TimeQA when the abstracted query's retrievals were combined with the original. Most subsequent work extended the technique to multi-turn or multi-hop retrieval, but the core abstraction step is unchanged.",
        "By 2026 step-back is a standard branch inside multi-query fusion. The cookbook implements it standalone here for clarity; in production it composes with HyDE and N paraphrases inside `reciprocal_rank_fusion`. The composition is what makes the pattern feel polished — each query-transformation flavour catches different failure modes.",
    ],
    theory_landscape=[
        "Among query-transformation cousins in this cookbook:",
        "",
        "- **HyDE (Recipe 13).** Concrete fake answer in answer-register.",
        "- **Multi-query (Recipe 14).** N paraphrases that emphasise different vocabulary.",
        "- **Step-back (this recipe).** One more-abstract query that retrieves principles.",
        "- **Sub-question decomposition (Recipe 16).** Break the question into parts and run RAG per part.",
        "",
        "Step-back wins on reasoning questions over textbooks because it pulls in the background concepts the specific query alone wouldn't retrieve. Sub-question decomposition wins on multi-part questions. They compose well — sub-questions can each be step-back-expanded before retrieval.",
    ],
    theory_when_to_use=[
        "Use step-back on reasoning questions over textbooks, surveys, or didactic content. The technique adds background that the specific query alone wouldn't retrieve, which the model needs to construct a complete answer.",
        "Skip it on factoid questions. \"In what year was YBCO discovered?\" doesn't need abstraction; the specific entity is already what retrieval needs.",
        "Skip it when the corpus has no abstract layer to retrieve. If your documents are pure facts (FAQ entries, product specs), the abstracted query retrieves nothing useful and just costs you an extra LLM call.",
        "Skip it when the question is genuinely multi-part rather than reasoning-deep — sub-question decomposition (Recipe 16) is the right tool there.",
    ],
    theory_intuition=[
        "Four intuitions to keep in mind:",
        "",
        "**Reasoning questions hide a principle.** \"Given X, why Y?\" assumes both X-specific knowledge and Y-principle knowledge. Standard retrieval finds X. Step-back retrieves the Y-principle so the model has both ingredients.",
        "",
        "**Abstracted queries match background sections.** Survey papers and textbooks have introduction sections that explain principles. Abstracted queries retrieve those introductions; specific queries retrieve the body. The pair gives the model both.",
        "",
        "**Fusion is essential.** Step-back alone usually under-performs the raw query because it retrieves only principles and may miss the specific entity. The combination is what wins; the cookbook's `step_back_retrieve` does the fusion under the hood.",
        "",
        "**Abstraction is cheap.** The abstractor only writes one short rewrite. The cost is one LLM call plus one extra retrieval — well under double the vanilla pipeline cost.",
    ],

    architecture_mermaid="""
flowchart TB
  Q[Specific question] --> A[LLM: abstract<br/>to a parent question]
  A --> AP[Parent question]
  Q --> R1[Retrieve specific]
  AP --> R2[Retrieve principles]
  R1 --> F[RRF fuse]
  R2 --> F
  F --> G[LLM answers]
""",

    references=[
        Reference(
            title="Take a Step Back — DeepMind, 2023",
            url="https://arxiv.org/abs/2310.06117",
            kind="paper",
            note="The original step-back paper.",
        ),
        Reference(
            title="LangChain step-back retriever tutorial",
            url="https://python.langchain.com/docs/tutorials/qa_chat_history/",
            kind="docs",
            note="Reference implementation pattern.",
        ),
        Reference(
            title="HyDE (Recipe 13)",
            url="https://arxiv.org/abs/2212.10496",
            kind="paper",
            note="Companion query-transformation; composes well.",
        ),
        Reference(
            title="Multi-query fusion (Recipe 14)",
            url="https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1",
            kind="blog",
            note="Step-back is often one branch inside multi-query fusion.",
        ),
        Reference(
            title="Self-Ask prompting",
            url="https://arxiv.org/abs/2210.03350",
            kind="paper",
            note="Adjacent reasoning-via-decomposition technique.",
        ),
        Reference(
            title="Chain-of-Verification (CoVe)",
            url="https://arxiv.org/abs/2309.11495",
            kind="paper",
            note="A different abstraction-then-verification pattern.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Index the Mamba survey",
                "",
                "Standard setup. The survey has both background sections (principles) and core sections (specifics), so step-back has somewhere to land both queries.",
            ],
            code=[
                "from cookbook.corpora import load_arxiv_mamba",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_arxiv_mamba())",
                "chunks = sentence_window(docs, sentences_per_chunk=4)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('stepback', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — The step-back prompt",
                "",
                "Ask the LLM for a more general version of the question. The cookbook's `cookbook.retrievers.step_back_retrieve` uses this prompt under the hood; we replicate it here for clarity.",
            ],
            code=[
                "ABSTRACT_PROMPT = (",
                "    \"Rewrite the user's question into a more general, higher-level question \"",
                "    'that asks about the underlying principle or category. Output only the rewritten question.\\n\\n'",
                "    'Question: {q}'",
                ")",
                "",
                "q = 'Given that selective scan needs input-dependent transitions, why is HiPPO initialization still useful?'",
                "parent = client.chat(ABSTRACT_PROMPT.format(q=q)).strip()",
                "print(f'Specific: {q}')",
                "print(f'Parent  : {parent}')",
            ],
            expected_output_md=[
                "A good abstracted query removes the specific entity (\"selective scan\", \"HiPPO\") and asks about the underlying principle (\"initialization in state-space models\"). If the parent question still names the specific entity, the prompt is too lenient.",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 3 — Retrieve with both, fuse",
                "",
                "Embed both queries, retrieve top-k for each, fuse with RRF. `cookbook.retrievers.step_back_retrieve` does this; we use it directly.",
            ],
            code=[
                "from cookbook.retrievers import step_back_retrieve",
                "",
                "hits = step_back_retrieve(q, store, client.chat, client.embed, top_k=6)",
                "for h in hits:",
                "    print(f'  {h.score:.4f}  {h.text[:160]}')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 4 — Inspect what each branch contributes to the fused result",
                "",
                "RRF fuses both rankings. Some passages come from the specific branch, some from the abstract branch, and some appear in both. We mark the source for transparency.",
            ],
            code=[
                "specific_hits = {h.doc_id for h in store.search(client.embed([q])[0], top_k=10)}",
                "abstract_hits = {h.doc_id for h in store.search(client.embed([parent])[0], top_k=10)}",
                "for h in hits:",
                "    source = []",
                "    if h.doc_id in specific_hits: source.append('specific')",
                "    if h.doc_id in abstract_hits: source.append('abstract')",
                "    print(f'  rrf={h.score:.4f}  source={\",\".join(source) or \"unknown\"}  {h.text[:120]}')",
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
                "    'Use only the passages below to answer the question. '",
                "    'If they do not contain the answer, say so plainly.\\n\\n'",
                "    'Passages:\\n{context}\\n\\nQuestion: {question}\\nAnswer:'",
                ")",
                "",
                "def answer_question(question: str, k: int = 6) -> tuple[str, list[str]]:",
                "    hits = step_back_retrieve(question, store, client.chat, client.embed, top_k=k)",
                "    contexts = [h.text for h in hits]",
                "    return client.chat(PROMPT.format(context='\\n\\n'.join(contexts), question=question)), contexts",
                "",
                "ans, _ = answer_question(q)",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — three abstractions",
                "",
                "Read what the abstractor produces. The technique only works if abstractions are good.",
            ],
            code=[
                "for question in [",
                "    'Given selective scan, why is HiPPO still useful?',",
                "    'In Mamba, why does parallel scan matter on modern GPUs?',",
                "    'When would I prefer S4 over Mamba?',",
                "]:",
                "    parent = client.chat(ABSTRACT_PROMPT.format(q=question)).strip()",
                "    print(f'Q: {question}')",
                "    print(f'  parent: {parent}')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — top hits per branch",
                "",
                "Specific query retrieves specific passages. Abstract query retrieves principle passages. Look at both.",
            ],
            code=[
                "q = 'Given that selective scan needs input-dependent transitions, why is HiPPO initialization still useful?'",
                "parent = client.chat(ABSTRACT_PROMPT.format(q=q)).strip()",
                "print('--- Specific branch top-3 ---')",
                "for h in store.search(client.embed([q])[0], top_k=3):",
                "    print(f'  {h.score:.3f}  {h.text[:140]}')",
                "print()",
                "print('--- Abstract branch top-3 ---')",
                "for h in store.search(client.embed([parent])[0], top_k=3):",
                "    print(f'  {h.score:.3f}  {h.text[:140]}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — step-back vs raw on factoid query",
                "",
                "Step-back should not hurt factoid queries even when it doesn't help them. We verify.",
            ],
            code=[
                "q = 'In what year was Mamba published?'",
                "raw_top = store.search(client.embed([q])[0], top_k=1)[0]",
                "sb_hits = step_back_retrieve(q, store, client.chat, client.embed, top_k=1)",
                "print(f'Raw  top-1: {raw_top.text[:120]}')",
                "print(f'SB   top-1: {sb_hits[0].text[:120]}')",
                "print(f'Same chunk? {raw_top.doc_id == sb_hits[0].doc_id}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — recall@5 on the eval slice",
                "",
                "Loose recall proxy. Step-back should match or beat raw on reasoning questions; factoid questions tend to tie.",
            ],
            code=[
                "from cookbook.corpora import load_eval_questions",
                "qs = [q for q in load_eval_questions() if q['corpus'] == 'arxiv-mamba'][:8]",
                "def recall_fn(fn):",
                "    hits = 0",
                "    for q in qs:",
                "        retr = fn(q['question'])",
                "        gold = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "        if any(any(w[:6] in r.text.lower() for w in gold) for r in retr):",
                "            hits += 1",
                "    return hits / max(1, len(qs))",
                "raw = lambda q: store.search(client.embed([q])[0], top_k=5)",
                "sb = lambda q: step_back_retrieve(q, store, client.chat, client.embed, top_k=5)",
                "print(f'raw       recall@5 = {recall_fn(raw):.2f}')",
                "print(f'step-back recall@5 = {recall_fn(sb):.2f}')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a reasoning question.",
    ],
    run_code=[
        "ans, _ = answer_question('Given selective scan, why would I still want a good initialization for the state matrix?')",
        "print('=== Step-back answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla vs step-back.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'Given selective scan, why would I still want a good initialization for the state matrix?'",
        "base = vanilla_pipeline(q, corpus='arxiv-mamba', top_k=5)",
        "ours_a, ours_c = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'step-back', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Five knobs in priority order:",
        "",
        "1. **Abstraction prompt.** The whole technique pivots on this. \"Ask about the underlying principle\" works on textbooks; \"ask about the broader category\" works on encyclopedias. Test different phrasings on a labelled slice.",
        "2. **Abstractor model.** A small fast model is fine. Abstraction is easy and doesn't benefit from frontier-tier reasoning.",
        "3. **Top-k per branch.** We use top-k from each. Tune higher (8 each) if RRF is dropping useful chunks; lower if the fused list is too long.",
        "4. **Fuse or replace.** Cookbook fuses both branches with RRF. Some implementations replace the raw query with the abstracted one — worse on factoid queries, sometimes better on pure reasoning queries.",
        "5. **Compose inside multi-query (Recipe 14).** Step-back is one paraphrase among many. Production stacks often blend HyDE, step-back, and three vanilla paraphrases inside one RRF pool.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Over-abstraction.** \"What is electricity?\" is too general; it retrieves background that doesn't help. Tighten the prompt to keep the parent question on-topic.",
        "- **No principle layer in the corpus.** If your corpus is all facts, the abstracted query retrieves nothing useful. Step-back wastes a call.",
        "- **Factoid contamination.** Adding principle context to a factoid query sometimes confuses the model. Self-RAG (Recipe 24) filters this out.",
        "",
        "Compose with HyDE (Recipe 13) and multi-query (Recipe 14) under one RRF. Compose with sub-question decomposition (Recipe 16) when the question is both abstract and multi-part.",
    ],
)


# =============================================================================
# Sub-Question Decomposition
# =============================================================================
SUB_QUESTION = Recipe(
    path="recipes/03-query-transformation/sub-question-decomposition.ipynb",
    title="Sub-Question Decomposition — Divide and Compose",
    category="query-transformation",
    corpus_filter="sec-10k-pltr",

    theory_problem=[
        "A multi-part question asks several things. \"What concentration risks does Palantir disclose for government customers, and how does its competitive landscape make those risks worse?\" wants both the concentration risk and the competitive analysis. Standard retrieval finds chunks about one or the other; the model has to invent the connection.",
        "Sub-question decomposition explicitly splits the question into independent sub-questions, retrieves and answers each, then composes the answers. Each retrieval is focused; the composition step is just text concatenation plus one final LLM call. Quality improves on multi-part questions; cost is one LLM call per sub-question plus the composer.",
    ],
    theory_origin=[
        "The pattern dates to LlamaIndex's `SubQuestionQueryEngine` in 2023. The technique is older in different forms (multi-hop QA in IR, decomposition prompting), but LlamaIndex packaged it cleanly for RAG. The decomposition-prompting line of research (Khot et al., 2022) provided the conceptual scaffolding — show the LLM what good decomposition looks like via few-shot, then let it generate sub-questions on demand.",
        "By 2025 it had become a standard for production systems whose users ask multi-part questions — analyst tools, research assistants, anywhere queries genuinely have N>1 retrievable parts. Modern implementations parallelise the sub-question retrievals, cap the decomposition depth, and pair the composer with a faithfulness check to avoid the cascading-error failure mode.",
    ],
    theory_landscape=[
        "Sub-question decomposition is the heaviest of the query transformations. Each sub-question is a full RAG call — embedding, retrieval, generation — so the cost multiplies with the number of sub-questions.",
        "",
        "- **HyDE / Multi-query / Step-back (Recipes 13-15).** Transform one query into another query, retrieve once.",
        "- **Sub-question (this recipe).** Transform one query into multiple queries, run them all, compose.",
        "- **Iterative multi-hop (Recipe 26, MULTI_HOP branch).** Transform sequentially based on what came back. Each step's question is informed by the previous step's answer.",
        "",
        "Sub-question is parallel decomposition; multi-hop is sequential. The two compose in production agents — the multi-hop branch can run sub-question decomposition at each hop, or sub-question can run multi-hop for each part.",
    ],
    theory_when_to_use=[
        "Use sub-question decomposition for multi-part questions and comparative analysis. Anywhere the user asks \"what about X and Y\", anywhere a research-style question genuinely needs multiple distinct pieces of evidence.",
        "Skip it for single-part questions. The decomposition step adds latency without value, and the composer call wastes another LLM round-trip.",
        "Skip it when sub-questions don't decompose cleanly. \"What is the meaning of life?\" is not multi-part; it's vague. Decomposing a vague question produces vague sub-questions and useless retrievals.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**Each sub-question retrieves cleanly.** Splitting \"X and Y\" into \"X\" and \"Y\" lets each retrieval focus on one topic, where a combined query would pull noisy chunks for both. Sharper retrieval per sub-question means better partial answers.",
        "",
        "**The composer is doing real work.** Combining partial answers requires reasoning about consistency, contradiction, and emphasis. The composer prompt matters — it has to instruct reconciliation, not just concatenation.",
        "",
        "**Runaway decomposition is the failure mode.** \"What is Palantir?\" can be decomposed into 10 sub-questions if the model is allowed. Cap it. Most production systems set the maximum at 4 sub-questions per query and reject the decomposition (falling back to flat RAG) if the model proposes more.",
    ],

    architecture_mermaid="""
flowchart TB
  Q[Multi-part question] --> D[LLM: decompose<br/>into sub-questions]
  D --> S1[Sub-question 1]
  D --> S2[Sub-question 2]
  D --> S3[Sub-question 3]
  S1 --> R1[RAG]
  S2 --> R2[RAG]
  S3 --> R3[RAG]
  R1 --> P1[Partial 1]
  R2 --> P2[Partial 2]
  R3 --> P3[Partial 3]
  P1 --> C[LLM: compose<br/>final answer]
  P2 --> C
  P3 --> C
""",

    references=[
        Reference(
            title="LlamaIndex SubQuestionQueryEngine",
            url="https://developers.llamaindex.ai/python/framework-api-reference/llama_index/core/query_engine/SubQuestionQueryEngine/",
            kind="docs",
            note="Reference implementation.",
        ),
        Reference(
            title="Decomposition prompting — Khot et al., 2022",
            url="https://arxiv.org/abs/2210.02406",
            kind="paper",
            note="The technique applied generally to LLM reasoning.",
        ),
        Reference(
            title="LangChain MultiVector Retriever",
            url="https://python.langchain.com/docs/how_to/multi_vector/",
            kind="docs",
            note="Related composition pattern.",
        ),
        Reference(
            title="Adaptive-RAG (Recipe 26)",
            url="https://arxiv.org/abs/2403.14403",
            kind="paper",
            note="Routes multi-hop questions to a sub-question-like branch.",
        ),
        Reference(
            title="Self-Ask prompting",
            url="https://arxiv.org/abs/2210.03350",
            kind="paper",
            note="Iterative self-decomposition; cousin pattern.",
        ),
        Reference(
            title="LangGraph multi-step QA tutorial",
            url="https://langchain-ai.github.io/langgraph/tutorials/multi_agent/",
            kind="docs",
            note="How to wrap decomposition in a stateful graph.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Index the SEC 10-K",
                "",
                "10-K filings are loaded with multi-part questions because they cover risk, business model, financials, governance. Standard setup.",
            ],
            code=[
                "from cookbook.corpora import load_sec_10k",
                "from cookbook.chunkers import fixed_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_sec_10k())",
                "chunks = fixed_window(docs, target_tokens=320, overlap_tokens=32)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('subq', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Decompose into sub-questions",
                "",
                "JSON-formatted output. We constrain the number to 2-4 to prevent runaway decomposition.",
            ],
            code=[
                "import json",
                "import re",
                "",
                "DECOMPOSE_PROMPT = (",
                "    'Break the question into 2-4 self-contained sub-questions whose individual answers, '",
                "    'combined, fully answer the original. Each sub-question must be answerable independently.\\n\\n'",
                '    \'Respond as JSON: {{\"sub_questions\": [\"...\", \"...\"]}}\\n\\n\'',
                "    'Question: {q}'",
                ")",
                "",
                "def decompose(question: str) -> list[str]:",
                "    raw = client.chat(DECOMPOSE_PROMPT.format(q=question))",
                "    m = re.search(r'\\{.*\\}', raw, flags=re.DOTALL)",
                "    if not m:",
                "        return [question]",
                "    try:",
                "        data = json.loads(m.group())",
                "        return list(data.get('sub_questions', [question]))",
                "    except json.JSONDecodeError:",
                "        return [question]",
                "",
                "q = 'What concentration risks does Palantir disclose for government customers, and how does its competitive landscape make those risks worse?'",
                "subs = decompose(q)",
                "for s in subs:",
                "    print(f'  - {s}')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 3 — Answer each sub-question",
                "",
                "Standard RAG per sub-question. We collect the partial answers and the contexts they used.",
            ],
            code=[
                "def answer_subquestion(sub: str) -> tuple[str, list[str]]:",
                "    qv = client.embed([sub])[0]",
                "    hits = store.search(qv, top_k=4)",
                "    contexts = [h.text for h in hits]",
                "    ans = client.chat(",
                "        'Use these passages.\\n' + '\\n\\n'.join(contexts) + f'\\nQuestion: {sub}\\nAnswer:'",
                "    )",
                "    return ans, contexts",
                "",
                "for s in subs:",
                "    a, _ = answer_subquestion(s)",
                "    print(f'Q: {s}')",
                "    print(f'   {a[:200]}...')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Compose the final answer",
                "",
                "The composer reads partial answers and produces a coherent response. The composer prompt explicitly asks for reconciliation of any conflicts.",
            ],
            code=[
                "COMPOSE_PROMPT = (",
                "    'Combine the partial answers below into one coherent response to the original question. '",
                "    'Reconcile any conflicts; do not invent facts.\\n\\n'",
                "    'Partial answers:\\n{partials}\\n\\n'",
                "    'Original question: {q}\\nFinal answer:'",
                ")",
                "",
                "partials = []",
                "all_contexts = []",
                "for s in subs:",
                "    a, ctx = answer_subquestion(s)",
                "    partials.append(f'Sub-Q: {s}\\nA: {a}')",
                "    all_contexts.extend(ctx)",
                "",
                "final = client.chat(COMPOSE_PROMPT.format(partials='\\n\\n'.join(partials), q=q))",
                "print(final)",
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
                "    subs = decompose(question)",
                "    partials, all_ctx = [], []",
                "    for s in subs:",
                "        a, ctx = answer_subquestion(s)",
                "        partials.append(f'Sub-Q: {s}\\nA: {a}')",
                "        all_ctx.extend(ctx)",
                "    final = client.chat(COMPOSE_PROMPT.format(partials='\\n\\n'.join(partials), q=question))",
                "    return final, all_ctx",
                "",
                "ans, _ = answer_question('How does AIP commercial revenue relate to government concentration risks?')",
                "print(ans[:400])",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — decompositions across three questions",
                "",
                "Read what the decomposer produces.",
            ],
            code=[
                "for question in [",
                "    'How does Palantir generate revenue and what risks does it disclose?',",
                "    'Compare AIP to Foundry and Gotham.',",
                "    'What governance structures protect founders, and what challenges do they create for investors?',",
                "]:",
                "    print(f'Q: {question}')",
                "    for s in decompose(question):",
                "        print(f'  - {s}')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — single-part query",
                "",
                "If the question is single-part, the decomposer should return one sub-question (or the original).",
            ],
            code=[
                "for question in [",
                "    'What is AIP?',",
                "    'In what state is Palantir incorporated?',",
                "]:",
                "    subs = decompose(question)",
                "    print(f'  Q: {question}')",
                "    print(f'    n_subs: {len(subs)}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost",
                "",
                "Sub-question decomposition is expensive. Count LLM calls per query.",
            ],
            code=[
                "from cookbook import _cache",
                "before = _cache.stats()['entries']",
                "_ = answer_question('How does Palantir generate revenue and what risks does it disclose?')",
                "after = _cache.stats()['entries']",
                "print(f'New cache entries: {after - before}')",
                "print('Rough breakdown for 3 sub-questions:')",
                "print('  1   decomposition LLM call')",
                "print('  3   query embeds (one per sub)')",
                "print('  3   per-sub answer LLM calls')",
                "print('  1   compose LLM call')",
                "print('  = 8 total LLM-equivalent calls')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — read one partial answer",
                "",
                "Make sure individual sub-question answers are coherent before composition. If they're not, composition cannot save them.",
            ],
            code=[
                "q = \"How does Palantir's AIP relate to its government revenue concentration risk?\"",
                "subs = decompose(q)",
                "for s in subs[:2]:",
                "    a, _ = answer_subquestion(s)",
                "    print(f'Sub-Q: {s}')",
                "    print(f'  {a[:300]}')",
                "    print()",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a multi-part question.",
    ],
    run_code=[
        "ans, _ = answer_question('What competitive landscape does Palantir describe, and how does it shape risks tied to government customers?')",
        "print('=== Sub-question composed answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla vs sub-question decomposition on a multi-part question.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What competitive landscape does Palantir describe, and how does it shape risks tied to government customers?'",
        "base = vanilla_pipeline(q, corpus='sec-10k-pltr', top_k=5)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:200]},",
        "    {'pipeline': 'sub-question', 'preview': ours_a[:200]},",
        "])",
    ],

    tuning_md=[
        "Five knobs in priority order:",
        "",
        "1. **Sub-question cap.** 2-4 is the cookbook default. Higher invites runaway decomposition and balloons cost without quality gain.",
        "2. **Decomposer model.** Mid-tier is fine; a frontier model is overkill for decomposition. Save your top-tier budget for the per-sub-question answering.",
        "3. **Per-sub-question `k`.** We use 4. Higher catches more recall but inflates the partial-answer length, which makes composition harder.",
        "4. **Composer prompt.** \"Reconcile conflicts\" matters. Without it the composer often picks the longest partial answer and ignores the rest.",
        "5. **Parallelise.** Sub-question answers are independent. Run them concurrently with asyncio or a thread pool to cut latency by 2-4x.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Bad decomposition.** Sub-questions that overlap heavily produce redundant retrievals. Sub-questions that don't cover the original miss information. Test on a labelled slice.",
        "- **Cascading errors.** If one sub-question's answer is wrong, the composer carries the error forward. Self-RAG (Recipe 24) over each partial answer reduces this.",
        "- **Cost explosion.** Latency and token cost scale with N sub-questions. Cap aggressively.",
        "",
        "Compose with adaptive routing (Recipe 26) — only decompose for genuinely multi-part questions. Compose with CRAG (Recipe 25) on each partial — fallback to web when the corpus misses.",
    ],
)


# =============================================================================
# Semantic Router
# =============================================================================
SEMANTIC_ROUTER = Recipe(
    path="recipes/03-query-transformation/semantic-router.ipynb",
    title="Semantic Router — Classify the Query, Dispatch the Index",
    category="query-transformation",
    corpus_filter=None,

    theory_problem=[
        "A production assistant covers many topics. The user asks one question; the system has many indexes. Sending every query to every index is wasteful and noisy — the top hit from the wrong index can outrank the right hit from the right one.",
        "Semantic routing solves the problem cheaply. Define routes by example: \"Rust ownership\", \"borrow checker\", \"lifetimes\" → `rust_book`. \"Cooper pairs\", \"BCS theory\" → `superconductors`. Embed the examples; at query time, find the route with the highest cosine match. Dispatch the query to that route's index. No LLM call needed.",
    ],
    theory_origin=[
        "The pattern was crystallised by Aurelio Labs' `semantic-router` library in 2024. The underlying technique — cosine-classifier with example phrases per class — predates RAG by years (it's how early intent-classification systems worked in dialogue research) but the library made it trivially composable with LLM agents.",
        "By 2026 every production RAG system that handles mixed query types uses some form of semantic router or LLM-as-classifier upstream. The semantic-router variant tends to win on cost; the LLM-classifier variant wins on flexibility. Most mature systems stack both — semantic router for fast routing on confident queries, LLM classifier as fall-through.",
    ],
    theory_landscape=[
        "Three routing options to know:",
        "",
        "- **Semantic router (this recipe).** Cosine match against example phrases per route. Free at query time after the index is built. Best when routes are well-separated.",
        "- **LLM-as-classifier (Adaptive-RAG, Recipe 26).** One LLM call per query. Slower, smarter, more nuanced on ambiguous queries.",
        "- **Metadata filter (Recipe 20).** Structured filters on chunk metadata. Requires labelled metadata to exist on chunks. Free at query time but inflexible.",
        "",
        "Stack them: semantic router for coarse routing, LLM-as-classifier as fallback when the router score is low. Metadata filters compose underneath both, narrowing the per-route search space further.",
    ],
    theory_when_to_use=[
        "Use a semantic router any time you have multiple knowledge sources. Multi-tenant systems, multi-product knowledge bases, multi-domain agents — anywhere a query genuinely belongs to one of several distinct indexes. The router is the cheapest decision in your stack.",
        "Skip it for single-source systems. There is nothing to route to and the routing cost is wasted.",
        "Skip it when routes blur. If users routinely ask questions that span multiple routes (\"compare X-related to Y-related\"), the router will mis-route or refuse consistently. A higher-level layer like sub-question decomposition (Recipe 16) handles cross-route queries better.",
    ],
    theory_intuition=[
        "Four intuitions:",
        "",
        "**Example phrases define the route.** A route is just a set of canonical queries. The more diverse the examples, the better the route generalises. Add new examples as you observe misrouting in production.",
        "",
        "**Cosine is enough.** No need for a learned classifier; the embedder already encodes semantic similarity. Cosine over example embeddings is a strong baseline that holds up against most fancier approaches in head-to-head tests.",
        "",
        "**Always have a fall-through.** When the top score is below a threshold, fall back to the LLM classifier or refuse. Hard-routing every query is brittle and costs you on out-of-distribution queries that will inevitably show up.",
        "",
        "**Threshold tuning is empirical.** There is no theoretical threshold value. Sweep on a labelled out-of-distribution slice and find where false-positive rate flattens.",
    ],

    architecture_mermaid="""
flowchart TB
  R1[Route 1<br/>example phrases] --> E1[Embed]
  R2[Route 2<br/>example phrases] --> E2[Embed]
  R3[Route 3<br/>example phrases] --> E3[Embed]
  E1 --> S[(Centroid<br/>or examples)]
  E2 --> S
  E3 --> S
  Q[Query] --> QE[Embed]
  QE --> M[Cosine match<br/>vs routes]
  S --> M
  M --> D{Score >=<br/>threshold?}
  D -->|yes| R[Dispatch]
  D -->|no| F[Fallback /<br/>refuse]
""",

    references=[
        Reference(
            title="aurelio-labs/semantic-router",
            url="https://github.com/aurelio-labs/semantic-router",
            kind="repo",
            note="The library that popularised the pattern.",
        ),
        Reference(
            title="Adaptive-RAG (Recipe 26)",
            url="https://arxiv.org/abs/2403.14403",
            kind="paper",
            note="LLM-as-classifier alternative.",
        ),
        Reference(
            title="LlamaIndex RouterQueryEngine",
            url="https://developers.llamaindex.ai/python/framework/module_guides/querying/router/",
            kind="docs",
            note="LlamaIndex's router abstraction.",
        ),
        Reference(
            title="RAGRouter Bench",
            url="https://arxiv.org/abs/2503.10231",
            kind="paper",
            note="Recent benchmark comparing routing approaches.",
        ),
        Reference(
            title="LangChain Multi-Vector + Routing tutorial",
            url="https://python.langchain.com/docs/how_to/routing/",
            kind="docs",
            note="LangChain's routing patterns.",
        ),
        Reference(
            title="MTEB Classification",
            url="https://huggingface.co/spaces/mteb/leaderboard",
            kind="docs",
            note="Embedding benchmark whose classification task correlates with route accuracy.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Define routes by example",
                "",
                "Each route is named and given 4-8 example phrases. The phrases should be diverse — different vocabulary, different aspects of the topic. For this notebook we route across all four cookbook corpora.",
            ],
            code=[
                "ROUTES = {",
                "    'arxiv-mamba': [",
                "        'state-space models', 'selective scan', 'linear attention alternative',",
                "        'long sequence modeling', 'recurrent neural network backbone',",
                "    ],",
                "    'wikipedia-superconductors': [",
                "        'Meissner effect', 'Cooper pairs', 'critical temperature', 'flux pinning',",
                "        'BCS theory', 'YBCO', 'high-temperature superconductor',",
                "    ],",
                "    'sec-10k-pltr': [",
                "        'annual report risk factors', 'government contract revenue', 'AIP platform',",
                "        'Gotham Foundry Apollo', 'cybersecurity disclosures', 'segment reporting',",
                "    ],",
                "    'rust-book': [",
                "        'ownership and borrowing', 'borrow checker', 'lifetimes', 'trait objects',",
                "        'async await', 'Rc and Arc', 'pattern matching',",
                "    ],",
                "}",
                "print(f'{len(ROUTES)} routes defined.')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 2 — Embed the examples",
                "",
                "One embedding per phrase. Store as numpy arrays for fast cosine.",
            ],
            code=[
                "import numpy as np",
                "",
                "route_vectors = {}",
                "for name, examples in ROUTES.items():",
                "    vecs = np.asarray(client.embed(examples), dtype=np.float32)",
                "    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)",
                "    route_vectors[name] = vecs",
                "print('Route vectors ready.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 3 — Score a query against each route",
                "",
                "Embed the query, dot-product with every route's example vectors, take the max within each route. The route with the highest max wins.",
            ],
            code=[
                "def route(question: str) -> tuple[str, float]:",
                "    q = np.asarray(client.embed([question])[0], dtype=np.float32)",
                "    q /= np.linalg.norm(q) + 1e-9",
                "    best, best_score = '?', -1.0",
                "    for name, vecs in route_vectors.items():",
                "        score = float((vecs @ q).max())",
                "        if score > best_score:",
                "            best, best_score = name, score",
                "    return best, best_score",
                "",
                "for q in [",
                "    'What is the asymptotic complexity of attention?',",
                "    'How does a SQUID work?',",
                "    'What is Apollo in Palantir terminology?',",
                "    'When should I prefer Arc over Rc?',",
                "]:",
                "    print(f'  {q[:60]:60s}  -> {route(q)}')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 4 — Dispatch + fall-through",
                "",
                "When the top score is below a threshold (default 0.4), the router can't confidently route. Fall through to a default behaviour — refuse, ask the user, or LLM-classify.",
            ],
            code=[
                "ROUTE_THRESHOLD = 0.4",
                "",
                "def route_or_none(question: str) -> str | None:",
                "    name, score = route(question)",
                "    return name if score >= ROUTE_THRESHOLD else None",
                "",
                "for q in [",
                "    'What is the Meissner effect?',",
                "    'Tell me about cooking pasta.',",
                "    'When should I use async in Rust?',",
                "]:",
                "    dest = route_or_none(q)",
                "    print(f'  {q[:50]:50s}  -> {dest!r}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap as `answer_question`",
                "",
                "For the standard cookbook contract, we route and then call into the appropriate corpus loader. We use the vanilla pipeline as the per-route RAG implementation.",
            ],
            code=[
                "from cookbook.baselines import vanilla_pipeline",
                "",
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    dest = route_or_none(question)",
                "    if dest is None:",
                "        return ('I could not confidently route your question.', [])",
                "    result = vanilla_pipeline(question, corpus=dest, top_k=5)",
                "    return result.answer, result.contexts",
                "",
                "ans, _ = answer_question('What is YBCO?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — score distribution per query",
                "",
                "For one query, see how confidently the router picked. The gap between top and second tells you whether routing is sharp or shaky.",
            ],
            code=[
                "q = 'How does the Meissner effect work?'",
                "qv = np.asarray(client.embed([q])[0], dtype=np.float32)",
                "qv /= np.linalg.norm(qv) + 1e-9",
                "scored = {name: float((vecs @ qv).max()) for name, vecs in route_vectors.items()}",
                "for name, s in sorted(scored.items(), key=lambda x: x[1], reverse=True):",
                "    print(f'  {name:35s}  {s:.3f}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — out-of-distribution queries",
                "",
                "Queries that don't fit any route should score below threshold. Make sure the fall-through fires.",
            ],
            code=[
                "for q in [",
                "    'What is the capital of France?',",
                "    'Tell me a joke.',",
                "    'Translate hello to Mandarin.',",
                "]:",
                "    _, s = route(q)",
                "    print(f'  {q[:40]:40s}  top-score={s:.3f}  routed={\"yes\" if s >= ROUTE_THRESHOLD else \"no\"}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — confusion matrix",
                "",
                "Run a labelled battery and see which queries go where. A confusion matrix highlights routes that overlap.",
            ],
            code=[
                "labelled = [",
                "    ('arxiv-mamba', 'What is selective scan?'),",
                "    ('arxiv-mamba', 'How does Mamba differ from a transformer?'),",
                "    ('wikipedia-superconductors', 'What is Cooper pairing?'),",
                "    ('wikipedia-superconductors', 'When was superconductivity discovered?'),",
                "    ('sec-10k-pltr', 'What are Palantir government revenue risks?'),",
                "    ('sec-10k-pltr', 'What is AIP?'),",
                "    ('rust-book', 'When should I use Arc over Rc?'),",
                "    ('rust-book', 'How does borrow checking work?'),",
                "]",
                "correct = sum(1 for label, q in labelled if route(q)[0] == label)",
                "print(f'Routing accuracy on labelled battery: {correct}/{len(labelled)}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what happens if I add more examples to a route?",
                "",
                "Routes get sharper with more examples. We test by adding 5 more examples to one route and re-scoring.",
            ],
            code=[
                "extra = ['superconducting magnet', 'persistent current', 'flux quantum', 'vortex lattice', 'penetration depth']",
                "extra_v = np.asarray(client.embed(extra), dtype=np.float32)",
                "extra_v /= np.linalg.norm(extra_v, axis=1, keepdims=True).clip(min=1e-9)",
                "augmented = np.vstack([route_vectors['wikipedia-superconductors'], extra_v])",
                "for q in ['What is a penetration depth?', 'How are SQUID magnetometers used?']:",
                "    qv = np.asarray(client.embed([q])[0], dtype=np.float32)",
                "    qv /= np.linalg.norm(qv) + 1e-9",
                "    before = float((route_vectors['wikipedia-superconductors'] @ qv).max())",
                "    after = float((augmented @ qv).max())",
                "    print(f'  {q[:50]:50s}  before={before:.3f}  after={after:.3f}')",
            ],
        ),
    ],

    run_md=[
        "End-to-end query with router.",
    ],
    run_code=[
        "for q in [",
        "    'What is selective scan in state-space models?',",
        "    'How does flux pinning work in superconductors?',",
        "    'What is the Apollo platform from Palantir?',",
        "    'How do lifetimes work in Rust?',",
        "]:",
        "    ans, _ = answer_question(q)",
        "    print(f'Q: {q}')",
        "    print(f'  -> {ans[:200]}')",
        "    print()",
    ],

    comparison_md=[
        "Vanilla (single fixed corpus) vs routed (picks the right corpus). Vanilla only knows one corpus; routed dispatches.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What is the Meissner effect?'",
        "base = vanilla_pipeline(q, corpus='rust-book', top_k=5)   # wrong corpus by design",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla (rust-book only)', 'preview': base.answer[:160]},",
        "    {'pipeline': 'semantic router', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Five knobs:",
        "",
        "1. **Example phrases.** More and more diverse improves routing. Tune by reading the misclassifications and adding examples that cover the gap.",
        "2. **Threshold.** Default 0.4. Lower means more aggressive routing (more false positives); higher means more refusals (more false negatives).",
        "3. **Aggregation across examples.** Max is the cookbook default. Mean is smoother but less responsive to single matching phrases.",
        "4. **Use centroid vectors.** Average each route's example vectors and store the centroid. Faster at query time; slightly less accurate than max-over-examples.",
        "5. **Compose with LLM-classifier fall-through.** When the router score is low, escalate to an LLM-classifier (Recipe 26). Best of both.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Overlapping routes.** Two routes that share vocabulary cause mis-routes. Re-examine route boundaries and add disambiguating examples.",
        "- **Out-of-distribution drift.** New topic categories appear in user queries that don't match any route. Set up a fall-through telemetry channel — log low-score queries and use them to seed new routes.",
        "- **Threshold tuning.** Too high refuses good queries; too low routes bad queries to wrong indexes. Sweep on a labelled slice.",
        "",
        "Compose with Adaptive-RAG (Recipe 26) for LLM-classifier fall-through. Compose with document-summary routing (Recipe 11) for a two-level router. Compose with metadata filters (Recipe 20) when chunks carry structured tags.",
    ],
)


RECIPES = [HYDE, MULTI_QUERY, STEP_BACK, SUB_QUESTION, SEMANTIC_ROUTER]
