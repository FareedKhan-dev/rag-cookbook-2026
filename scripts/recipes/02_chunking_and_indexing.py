"""Category 2 — Chunking & Indexing.

Eight recipes total. Batch 3 ships the two star-draws (contextual-retrieval-anthropic
and late-chunking-jina); Batch 4 adds the remaining six.
"""
from __future__ import annotations

from authoring import CodeStep, Recipe, Reference


# =============================================================================
# Contextual Retrieval (Anthropic 2024)
# =============================================================================
CONTEXTUAL_RETRIEVAL = Recipe(
    path="recipes/02-chunking-and-indexing/contextual-retrieval-anthropic.ipynb",
    title="Contextual Retrieval — Anthropic's Highest-ROI Tweak",
    category="chunking-and-indexing",
    corpus_filter="rust-book",

    theory_problem=[
        "A chunk loses everything its document said about *what it is*. The first paragraph of chapter 17 of the Rust book makes perfect sense in context — \"this chapter\" refers to async/await — but a chunk that starts with \"This chapter introduces the concept of futures and async tasks\" embedded in isolation matches every retrieval question that mentions chapters, futures, or tasks. The cosine vector is dominated by superficial cues, not the actual content.",
        "Anthropic's September 2024 contextual retrieval pattern fixes this with one prepended line: an LLM-written sentence that says where this chunk fits in the document. Chunks now embed with their context attached. Recall jumps 30–50 percent on standard benchmarks; pairing with BM25 and a reranker pushes it further. The implementation is twenty lines of code.",
    ],
    theory_origin=[
        "Anthropic published the technique in September 2024 with a detailed engineering blog and a reference implementation. The headline numbers: 35 percent improvement on retrieval failure rate when contextual headers were added to chunks; 49 percent when combined with contextual BM25; 67 percent when combined with reranking. Those numbers held across nine datasets including code repositories, legal text, and scientific papers.",
        "The technique itself was not novel — researchers had prepended hand-written headers to chunks for years. Anthropic's contribution was twofold: showing the LLM-generated header was as good as a hand-written one, and making it cheap to run at scale by using prompt caching. The 90 percent cache hit rate on the document context cut the per-token cost by an order of magnitude.",
    ],
    theory_landscape=[
        "Three competing approaches to the same problem of \"lost context\":",
        "",
        "- **Contextual retrieval (this recipe)** — prepend an LLM-written header. Cheap, language-agnostic, works on any corpus. Highest ROI of the three.",
        "- **Late chunking (Recipe 6)** — embed the full document first, then pool token embeddings into chunks. The context is baked into the vector instead of into the text. Requires a long-context embedder; saves the LLM-call cost but locks you into compatible embedders.",
        "- **Parent-child retrieval (Recipe 9)** — small chunks for search, big chunks for generation. Solves a related but different problem; pairs well with contextual retrieval.",
        "",
        "In production they stack. The Anthropic post explicitly recommends contextual retrieval + BM25 + reranker as the three-step pipeline. We build the first piece here; Recipe 18 adds BM25, Recipe 22 adds the reranker.",
    ],
    theory_when_to_use=[
        "Use contextual retrieval whenever your corpus is structured (chapters, sections, dated reports, threaded conversations) and chunks are likely to lose their position. Technical docs, legal filings, support tickets, codebases — all good fits.",
        "Skip it when chunks are already self-contained. Wikipedia summary paragraphs, FAQ entries, hand-curated knowledge-base articles often need no extra context. Adding a header buys you nothing and costs an LLM call per chunk.",
        "Skip it also when re-embedding costs are prohibitive. Contextual headers change the embedded text, which means you re-embed every chunk when you switch the header model or prompt. For corpora measured in hundreds of millions of chunks, that is a serious commitment.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**The header carries cheap context the chunk cannot.** Without the header, a chunk's vector represents only its own 384 tokens. With a header that says \"This chunk is from Chapter 17 (async/await), in the section on tokio runtimes, immediately after introducing futures\", the vector now encodes position, topic, and continuity. None of those signals were in the chunk's text.",
        "",
        "**Prompt caching is what makes it tractable.** Without caching, you call the LLM once per chunk with the full document context attached — at, say, 10,000 documents averaging 30 chunks, that is 300k LLM calls, each with the full document. Caching reduces it to one call per chunk with cached context, which is roughly the cost of a basic OCR pipeline.",
        "",
        "**The header does not have to be perfect.** A header that says \"This chunk discusses thread-safety concerns\" is useful even if it slightly mischaracterises the chunk. The embedding is robust to noise; what it needs is more *signal*, and even a noisy header is signal.",
    ],

    architecture_mermaid="""
flowchart LR
  D[Document] --> C[Chunk]
  D --> F[Full document<br/>cached context]
  C --> P[Prompt:<br/>describe where<br/>this chunk fits]
  F --> P
  P --> H[LLM-written<br/>header sentence]
  H --> A[Augmented chunk<br/>= header + chunk]
  A --> E[Embedder]
  E --> S[(Vector store)]
  style F fill:#fff5d0
  style H fill:#e9efff
""",

    references=[
        Reference(
            title="Introducing Contextual Retrieval (Anthropic, Sept 2024)",
            url="https://www.anthropic.com/news/contextual-retrieval",
            kind="blog",
            note="The blog post that defined the pattern.",
        ),
        Reference(
            title="Anthropic Prompt Caching documentation",
            url="https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching",
            kind="docs",
            note="The mechanism that makes contextual retrieval cheap at scale.",
        ),
        Reference(
            title="Contextual retrieval reference cookbook",
            url="https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings",
            kind="repo",
            note="Anthropic's own implementation; useful for cross-checking.",
        ),
        Reference(
            title="Late Chunking — Jina AI, 2024",
            url="https://jina.ai/news/late-chunking-in-long-context-embedding-models/",
            kind="blog",
            note="An alternative answer to the same problem; covered in Recipe 6.",
        ),
        Reference(
            title="OpenAI Prompt Caching",
            url="https://platform.openai.com/docs/guides/prompt-caching",
            kind="docs",
            note="OpenAI's equivalent; works the same way for our purposes.",
        ),
        Reference(
            title="LlamaIndex contextual retrieval pack",
            url="https://docs.llamaindex.ai/en/stable/examples/cookbooks/contextual_retrieval/",
            kind="docs",
            note="Reference implementation in the LlamaIndex ecosystem.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load a small slice of the corpus",
                "",
                "We use four chapters of the Rust book. The cookbook keeps the slice small (around 40 chunks) so the contextual-header generation completes quickly on a free Nebius account. The technique generalises to corpora of any size — only the bill grows.",
            ],
            code=[
                "from cookbook.corpora import load_rust_book",
                "from cookbook.chunkers import fixed_window",
                "",
                "docs = list(load_rust_book())[:4]",
                "chunks = fixed_window(docs, target_tokens=320, overlap_tokens=32)[:40]",
                "print(f'Working with {len(chunks)} chunks across {len(docs)} chapters.')",
                "print()",
                "print('First chunk preview:')",
                "print(chunks[0].text[:240])",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Define the contextual-header prompt",
                "",
                "Anthropic's prompt is short and direct. We follow the same shape: a one-line role, the document title and a chunk-locating sentence, then ask for a single sentence of context. The constraint matters — without \"one sentence only\" the model writes a paragraph.",
            ],
            code=[
                "CONTEXT_PROMPT = (",
                "    'You are creating a one-sentence header that will be prepended to a chunk before embedding, '",
                "    'so a vector search engine can rank it correctly. '",
                "    'Describe where this chunk fits in the document and what it covers. '",
                "    'Output exactly one sentence, no preamble.\\n\\n'",
                "    'Document title: {title}\\n\\n'",
                "    'Chunk:\\n{chunk}\\n\\nHeader:'",
                ")",
                "",
                "sample_header = client.chat(CONTEXT_PROMPT.format(",
                "    title=chunks[0].metadata.get('title', chunks[0].doc_id),",
                "    chunk=chunks[0].text[:1500],",
                "))",
                "print('Generated header:')",
                "print(' ', sample_header.strip())",
            ],
            expected_output_md=[
                "A good header reads like a one-sentence chapter caption. \"This chunk is from the opening of the chapter on async and futures, introducing the runtime concept and motivating why Rust separates the executor from the language.\" Notice how that sentence carries information the chunk's first 30 words cannot.",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Generate headers for every chunk",
                "",
                "One LLM call per chunk. With caching on, repeated runs are free; the first run takes about a second per chunk on Nebius's default model. We store `(header, chunk)` pairs so the next step can embed them with the header prepended.",
            ],
            code=[
                "def contextualise(chunks):",
                "    out = []",
                "    for c in chunks:",
                "        title = c.metadata.get('title', c.doc_id)",
                "        header = client.chat(CONTEXT_PROMPT.format(title=title, chunk=c.text[:1500]))",
                "        out.append((header.strip(), c))",
                "    return out",
                "",
                "contextual = contextualise(chunks)",
                "print(f'Generated {len(contextual)} contextual headers.')",
                "print()",
                "print('Three headers + opening lines of their chunks:')",
                "for header, c in contextual[:3]:",
                "    print(f'  HEADER: {header}')",
                "    print(f'  CHUNK : {c.text[:100]!r}...')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 4 — Embed `header + chunk` and index",
                "",
                "Standard embedding, but the input is the concatenation. Nothing fancy. The whole technique lives in step 3; this step is just plumbing.",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "",
                "augmented_texts = [f'{h}\\n\\n{c.text}' for h, c in contextual]",
                "vectors = client.embed(augmented_texts)",
                "",
                "store = QdrantBackend('contextual', dim=len(vectors[0]))",
                "store.add(",
                "    texts=augmented_texts,",
                "    vectors=vectors,",
                "    metadatas=[c.metadata for _, c in contextual],",
                "    ids=[c.chunk_id for _, c in contextual],",
                ")",
                "print(f'Indexed {len(augmented_texts)} contextualised chunks.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 5 — Compare retrieval on a question that needs context",
                "",
                "Pick a question whose answer chunk would be hard to find without its document position. Run retrieval against both the contextual store and a parallel store built from raw (uncontextualised) chunks. The gap is the technique's value.",
            ],
            code=[
                "raw_vectors = client.embed([c.text for c in chunks])",
                "raw_store = QdrantBackend('raw', dim=len(raw_vectors[0]))",
                "raw_store.add([c.text for c in chunks], raw_vectors, ids=[c.chunk_id for c in chunks])",
                "",
                "q = 'When inside an async block do I need to await a future for it to make progress?'",
                "qv = client.embed([q])[0]",
                "",
                "print('--- raw retrieval (no headers) ---')",
                "for h in raw_store.search(qv, top_k=3):",
                "    print(f'  score={h.score:.3f}  {h.text[:160]}')",
                "",
                "print()",
                "print('--- contextual retrieval ---')",
                "for h in store.search(qv, top_k=3):",
                "    print(f'  score={h.score:.3f}  {h.text[:160]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 6 — Wrap it as `answer_question`",
                "",
                "Standard contract. The augmented chunk text (header + body) gets passed to the LLM; the model is free to use the header for grounding too.",
            ],
            code=[
                "PROMPT = (",
                "    'Use only the passages below to answer the question. '",
                "    'If they do not contain the answer, say so plainly.\\n\\n'",
                "    'Passages:\\n{context}\\n\\nQuestion: {question}\\nAnswer:'",
                ")",
                "",
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    hits = store.search(qv, top_k=k)",
                "    contexts = [h.text for h in hits]",
                "    answer = client.chat(PROMPT.format(context='\\n\\n'.join(contexts), question=question))",
                "    return answer, contexts",
                "",
                "ans, _ = answer_question('How do you mark a function as runnable inside the async runtime, and what changes about its return type?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does a typical header look like?",
                "",
                "Headers are the technique. If they are vague, the technique gives you nothing. If they are specific, you get the recall lift. Read five of them.",
            ],
            code=[
                "for header, c in contextual[10:15]:",
                "    print(f'  Chapter   : {c.metadata.get(\"chapter\", c.doc_id)}')",
                "    print(f'  Header    : {header}')",
                "    print(f'  Chunk start: {c.text[:80]!r}...')",
                "    print()",
            ],
            expected_output_md=[
                "Good headers name the chapter, the sub-topic, and where in the flow this chunk sits. Bad headers say things like \"This chunk discusses Rust\" — too generic to add retrieval signal. If yours look bad, sharpen the CONTEXT_PROMPT.",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — measure recall@5 on the eval slice",
                "",
                "Use the corpus's eval-set slice. Run the loose recall proxy from Recipe 3 on both raw and contextual stores. The contextual store should beat raw by 3–10 points on a corpus that benefits — Rust chapters are a moderate-benefit case.",
            ],
            code=[
                "from cookbook.corpora import load_eval_questions",
                "",
                "qs = [q for q in load_eval_questions() if q['corpus'] == 'rust-book'][:10]",
                "def recall(store_):",
                "    hits = 0",
                "    for q in qs:",
                "        qv = client.embed([q['question']])[0]",
                "        retrieved = store_.search(qv, top_k=5)",
                "        gold_words = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "        if any(any(w[:6] in r.text.lower() for w in gold_words) for r in retrieved):",
                "            hits += 1",
                "    return hits / max(1, len(qs))",
                "",
                "print(f'raw recall@5         = {recall(raw_store):.2f}')",
                "print(f'contextual recall@5  = {recall(store):.2f}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what is the per-chunk cost?",
                "",
                "We measure the size of the contextualisation payload so the cost is visible. Multiply by your corpus size at scale; the prompt cache lowers the input-token bill by 5–10x in production.",
            ],
            code=[
                "import tiktoken",
                "enc = tiktoken.get_encoding('cl100k_base')",
                "header_tokens = enc.encode(contextual[0][0])",
                "chunk_tokens = enc.encode(contextual[0][1].text)",
                "prompt_tokens = enc.encode(CONTEXT_PROMPT.format(title=contextual[0][1].metadata.get('title', ''), chunk=contextual[0][1].text[:1500]))",
                "print(f'Header  tokens : {len(header_tokens)}')",
                "print(f'Chunk   tokens : {len(chunk_tokens)}')",
                "print(f'Prompt  tokens : {len(prompt_tokens)}  (cached after first call on a given document)')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — does the technique change which chunks rank first?",
                "",
                "Compare the top-1 chunk under raw vs contextual retrieval for several questions. The interesting case is when the top-1 changes — that is where contextual retrieval found a better answer than raw.",
            ],
            code=[
                "questions = [",
                "    'How do you define a thread-safe shared counter in Rust?',",
                "    'When should I prefer Rc over Arc?',",
                "    'What does the question-mark operator do for error propagation?',",
                "    'In an async function, what type does the function actually return?',",
                "]",
                "for q in questions:",
                "    qv = client.embed([q])[0]",
                "    raw_top = raw_store.search(qv, top_k=1)[0]",
                "    ctx_top = store.search(qv, top_k=1)[0]",
                "    same = raw_top.doc_id == ctx_top.doc_id",
                "    print(f'  {\"same\" if same else \"DIFFERENT\":10s}  {q[:70]}')",
            ],
        ),
    ],

    run_md=[
        "Representative end-to-end run on a question whose answer chunk depends on its document position.",
    ],
    run_code=[
        "q = 'When inside an async function do you need to await another async call before its work makes progress?'",
        "ans, ctxs = answer_question(q)",
        "print('=== Answer ===')",
        "print(ans)",
        "print()",
        "print('Top context preview:')",
        "print(ctxs[0][:300])",
    ],

    comparison_md=[
        "Vanilla baseline vs contextual retrieval on the same question. The interesting metric here is not the answer text — both pipelines usually answer correctly — but which passages were retrieved, and whether the contextual version pulled in the position-dependent chunk that vanilla missed.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'When inside an async function do you need to await another async call before its work makes progress?'",
        "base = vanilla_pipeline(q, corpus='rust-book', top_k=5)",
        "ours_a, ours_c = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'top_context_start': base.contexts[0][:140]},",
        "    {'pipeline': 'contextual', 'top_context_start': ours_c[0][:140]},",
        "])",
    ],

    tuning_md=[
        "Five knobs:",
        "",
        "1. **The CONTEXT_PROMPT itself.** The biggest lever. A prompt that asks for sub-topic, sub-section, and position usually beats a prompt that asks only for sub-topic. Iterate by hand-reading 10 generated headers and rewriting the prompt to fix the failure mode you see.",
        "2. **Header position.** We prepend the header to the chunk before embedding. Some teams prepend and append; some put the header in metadata only and rely on hybrid search. Anthropic's reference puts header before chunk; we follow.",
        "3. **Header model.** Headers do not need to be written by your best model. A small fast model (Llama-3.3-8B, Qwen-2.5-7B) produces headers of comparable quality at a fraction of the cost. Test on your corpus.",
        "4. **How much document context to give the model.** Anthropic feeds the *full document* alongside the chunk so the header can refer to surrounding sections. For long documents this is expensive without prompt caching; we use the first 1500 chars of the chunk only here for clarity. In production, feed the full document and enable caching.",
        "5. **Re-contextualisation cadence.** When you upgrade the header model or rewrite the prompt, you must re-embed every chunk. Plan for it. Many teams contextualise once, then leave the headers fixed across embedder upgrades.",
    ],

    discussion_md=[
        "Three places contextual retrieval falls down:",
        "",
        "- **Tiny chunks.** If your chunks are already short enough to be self-explanatory, a header adds nothing. We see this in proposition-decomposition pipelines (Recipe 8) where each chunk is a single factual claim.",
        "- **Bad headers.** A model that does not understand your domain writes vague headers. Test on a labelled slice before going to production. If the model writes \"This passage discusses programming language semantics\" five times in a row, the prompt is too generic.",
        "- **Storage of full augmented text.** The augmented text (header + chunk) is what you embed, but you also have to *store* it so retrieval can return the original chunk. This roughly doubles storage at the chunk level; small for most workloads, real at billions.",
        "",
        "Compose it with everything: BM25 hybrid retrieval (Recipe 18), cross-encoder reranking (Recipe 22), Self-RAG filtering (Recipe 24). Anthropic's full pipeline is contextual retrieval + BM25 + reranking; that is what the 49 percent and 67 percent numbers came from.",
    ],
)


# =============================================================================
# Late Chunking (Jina AI 2024)
# =============================================================================
LATE_CHUNKING = Recipe(
    path="recipes/02-chunking-and-indexing/late-chunking-jina.ipynb",
    title="Late Chunking — Embed First, Cut After",
    category="chunking-and-indexing",
    corpus_filter="arxiv-mamba",

    theory_problem=[
        "Classical chunking cuts the document, then embeds each chunk independently. The embedder sees one chunk at a time and has no idea what came before. A chunk that says \"It scales linearly in sequence length\" has lost the \"it\" — the reader knows from the surrounding paragraph that \"it\" is Mamba, but the vector does not.",
        "Late chunking inverts the order. Embed the whole document first with a long-context embedder, then pool the token-level embeddings into chunks. Each chunk's vector now carries information about everything around it. The cost: you need a long-context embedder and slightly more careful pooling code. The benefit: chunks remember their surroundings without an LLM call per chunk.",
    ],
    theory_origin=[
        "Late chunking was named and published by Jina AI in September 2024. The idea — pool token embeddings rather than embedding pre-cut chunks — appears in earlier sentence-transformer work, but Jina's contribution was packaging it as a one-line API in their jina-embeddings-v3 model and showing concrete recall lifts (typically 5–15 points on standard benchmarks).",
        "By 2026 most long-context embedding models (jina-embeddings-v3, Voyage 3 Large, Cohere Embed 4) expose a late-chunking mode. Open-source implementations on top of bge-m3 and other transformer-based embedders work the same way — the model exposes token embeddings, you pool by chunk boundary.",
    ],
    theory_landscape=[
        "Three answers to the lost-context problem:",
        "",
        "- **Contextual retrieval (Recipe 7)** — prepend an LLM-written header to each chunk. Header is text the embedder can read.",
        "- **Late chunking (this recipe)** — let the embedder read the whole document at once and pool by chunk boundary. Header is implicit in the pooled vector.",
        "- **Parent-child retrieval (Recipe 9)** — store small chunks for search and big chunks for generation. Context is recovered at generation time.",
        "",
        "The three compose: late chunking handles the embedding-time context, parent-child handles the generation-time context. Add BM25 (Recipe 18) and reranking (Recipe 22) on top and you have a production-grade retrieval stack.",
    ],
    theory_when_to_use=[
        "Use late chunking when you have a long-context embedder available and documents long enough that classical chunks lose context. Scientific papers, legal contracts, long-form blog posts, structured reports — all good fits.",
        "Skip it when your embedder caps at 512 tokens. Many older sentence-transformers, including some popular MTEB-leaders, do not produce useful token embeddings at long context. Without that, late chunking degrades to classical chunking.",
        "Skip it when document length exceeds your embedder's window. A 1M-token book cannot be late-chunked with an 8K embedder. Split into sections first, then late-chunk inside each section.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**The chunk vector is a pooled token vector.** In classical chunking, you embed text → get a vector. In late chunking, you embed text → get token vectors → pool the ones inside the chunk's span. The pool is usually a mean; the average of token vectors over a span is itself an embedding.",
        "",
        "**Long context is doing the work.** The embedder sees the whole document at once, so token embeddings for chunk #5 are computed with full attention to chunks #1–4. Each token's vector encodes its surroundings. When you pool those vectors for chunk #5, the chunk vector inherits that contextual awareness.",
        "",
        "**No LLM call required.** Unlike contextual retrieval, this is purely an embedder trick. No header generation, no prompt-caching plumbing, no per-chunk LLM cost. The savings show up when scaling.",
    ],

    architecture_mermaid="""
flowchart TB
  D[Long document] --> EM[Long-context<br/>embedder]
  EM --> TT[Per-token<br/>embeddings]
  D --> SP[Sentence/chunk<br/>boundaries]
  TT --> PL[Mean-pool<br/>tokens per chunk]
  SP --> PL
  PL --> V[One chunk vector<br/>per chunk]
  V --> S[(Vector store)]
""",

    references=[
        Reference(
            title="Late Chunking in Long-Context Embedding Models (Jina AI, 2024)",
            url="https://jina.ai/news/late-chunking-in-long-context-embedding-models/",
            kind="blog",
            note="The blog post that defined the technique.",
        ),
        Reference(
            title="Late Chunking — research paper",
            url="https://arxiv.org/abs/2409.04701",
            kind="paper",
            note="The arXiv writeup with benchmark numbers.",
        ),
        Reference(
            title="jina-embeddings-v3 model card",
            url="https://huggingface.co/jinaai/jina-embeddings-v3",
            kind="repo",
            note="The open-weight model with native late-chunking support.",
        ),
        Reference(
            title="Late chunking reference implementation",
            url="https://github.com/jina-ai/late-chunking",
            kind="repo",
            note="The Python reference; we follow its pooling logic.",
        ),
        Reference(
            title="Contextual Retrieval — Anthropic 2024",
            url="https://www.anthropic.com/news/contextual-retrieval",
            kind="blog",
            note="Recipe 7 — the LLM-header alternative.",
        ),
        Reference(
            title="Voyage 3 Large embedding model card",
            url="https://docs.voyageai.com/docs/embeddings",
            kind="docs",
            note="Hosted long-context embedder with late-chunking mode.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load the arXiv Mamba survey, page by page",
                "",
                "We use a small slice — the first few pages — so the long-context embedder fits without GPU help. Late chunking shines on long documents, so even this slice is enough to demonstrate the value.",
            ],
            code=[
                "from cookbook.corpora import load_arxiv_mamba",
                "",
                "docs = list(load_arxiv_mamba())[:6]",
                "print(f'Working with {len(docs)} pages.')",
                "total_chars = sum(len(d.text) for d in docs)",
                "print(f'Total characters: {total_chars:,}')",
            ],
        ),
        CodeStep(
            tag="chunk",
            lead_md=[
                "### Step 2 — Naïve sentence-window chunks for comparison",
                "",
                "Build a baseline classical chunking so we can compare. We use the same chunker we use everywhere; nothing fancy.",
            ],
            code=[
                "from cookbook.chunkers import sentence_window",
                "",
                "naive_chunks = sentence_window(docs, sentences_per_chunk=4, overlap=1)",
                "print(f'Built {len(naive_chunks)} naïve chunks.')",
                "print()",
                "print('First three chunks:')",
                "for c in naive_chunks[:3]:",
                "    print(f'  {c.text[:140]!r}')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 3 — Build the naïve-chunking index",
                "",
                "Embed and index the classical chunks. This is the baseline we compare late chunking against.",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "",
                "naive_vectors = client.embed([c.text for c in naive_chunks])",
                "naive_store = QdrantBackend('naive', dim=len(naive_vectors[0]))",
                "naive_store.add([c.text for c in naive_chunks], naive_vectors, ids=[c.chunk_id for c in naive_chunks])",
                "print(f'Indexed {len(naive_chunks)} naïve chunks.')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 4 — Pseudo-late-chunking via prefixed context",
                "",
                "True late chunking requires token-level access to the embedder, which Nebius's hosted API does not expose. We approximate the effect: for each chunk, embed (whole-document context + chunk) and use that vector. The model attends to the full document while producing what is effectively a context-aware chunk embedding. This is not literally Jina's late chunking — that pools token embeddings — but it produces the same kind of recall lift via the same mechanism (context-aware vectors).",
                "",
                "If you have a local jina-embeddings-v3 install, swap this for the real pooling implementation; the rest of the recipe stays identical.",
            ],
            code=[
                "def pseudo_late_chunk_texts(docs, chunks):",
                "    by_doc = {}",
                "    for d in docs:",
                "        by_doc[d.doc_id] = d.text[:4000]",
                "    augmented = []",
                "    for c in chunks:",
                "        doc_context = by_doc.get(c.doc_id, '')[:2000]",
                "        augmented.append(f'CONTEXT (excerpt from document): {doc_context}\\n\\nCHUNK: {c.text}')",
                "    return augmented",
                "",
                "late_texts = pseudo_late_chunk_texts(docs, naive_chunks)",
                "late_vectors = client.embed(late_texts)",
                "late_store = QdrantBackend('late', dim=len(late_vectors[0]))",
                "late_store.add(",
                "    texts=[c.text for c in naive_chunks],   # store the *original* chunk for display",
                "    vectors=late_vectors,",
                "    ids=[c.chunk_id for c in naive_chunks],",
                ")",
                "print(f'Indexed {len(naive_chunks)} late-chunked vectors.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 5 — Compare on a question that needs context",
                "",
                "Pick a question whose answer chunk uses pronouns or implicit references that lose meaning without the surrounding pages.",
            ],
            code=[
                "q = 'How does it scale to long sequences compared to classical attention?'",
                "qv = client.embed([q])[0]",
                "",
                "print('--- naive top-3 ---')",
                "for h in naive_store.search(qv, top_k=3):",
                "    print(f'  score={h.score:.3f}  {h.text[:160]}')",
                "print()",
                "print('--- late-chunked top-3 ---')",
                "for h in late_store.search(qv, top_k=3):",
                "    print(f'  score={h.score:.3f}  {h.text[:160]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 6 — Wrap as `answer_question`",
                "",
                "The cookbook contract. We use the late-chunked store for the answer.",
            ],
            code=[
                "PROMPT = (",
                "    'Use only the passages below to answer the question. '",
                "    'If they do not contain the answer, say so plainly.\\n\\n'",
                "    'Passages:\\n{context}\\n\\nQuestion: {question}\\nAnswer:'",
                ")",
                "",
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    hits = late_store.search(qv, top_k=k)",
                "    contexts = [h.text for h in hits]",
                "    return client.chat(PROMPT.format(context='\\n\\n'.join(contexts), question=question)), contexts",
                "",
                "ans, _ = answer_question('What complexity advantage motivates state-space models over attention?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — recall@5 on the eval slice",
                "",
                "Run the loose recall proxy on both stores. Late chunking should produce a noticeable lift on questions with implicit references (\"it\", \"the model\", \"this approach\"); classical chunking usually wins on questions with explicit entities (\"selective scan\", \"HiPPO matrix\").",
            ],
            code=[
                "from cookbook.corpora import load_eval_questions",
                "",
                "qs = [q for q in load_eval_questions() if q['corpus'] == 'arxiv-mamba'][:10]",
                "def recall(store_):",
                "    hits = 0",
                "    for q in qs:",
                "        qv = client.embed([q['question']])[0]",
                "        retrieved = store_.search(qv, top_k=5)",
                "        gold_words = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "        if any(any(w[:6] in r.text.lower() for w in gold_words) for r in retrieved):",
                "            hits += 1",
                "    return hits / max(1, len(qs))",
                "",
                "print(f'naive recall@5  = {recall(naive_store):.2f}')",
                "print(f'late  recall@5  = {recall(late_store):.2f}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — when does the top-1 change?",
                "",
                "List questions where naïve and late-chunked retrieval pick different top-1 chunks. Those are the cases where the technique paid for itself.",
            ],
            code=[
                "from cookbook.corpora import load_eval_questions",
                "qs = [q for q in load_eval_questions() if q['corpus'] == 'arxiv-mamba'][:8]",
                "for q in qs:",
                "    qv = client.embed([q['question']])[0]",
                "    naive_top = naive_store.search(qv, top_k=1)[0]",
                "    late_top = late_store.search(qv, top_k=1)[0]",
                "    same = naive_top.doc_id == late_top.doc_id",
                "    label = 'same' if same else 'DIFFERENT'",
                "    print(f'  {label:9s}  {q[\"question\"][:70]}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — embedding distance between naïve vs late vectors",
                "",
                "For the same chunk, the naïve and late-chunked vectors live in slightly different parts of embedding space. The cosine similarity between them tells you how much context the late-chunking step injected.",
            ],
            code=[
                "import numpy as np",
                "",
                "naive_v = np.asarray(naive_vectors[10])",
                "late_v = np.asarray(late_vectors[10])",
                "cos = float((naive_v @ late_v) / (np.linalg.norm(naive_v) * np.linalg.norm(late_v)))",
                "print(f'cosine(naive, late) for chunk 10 = {cos:.4f}')",
                "print('A cosine below ~0.9 means late chunking shifted the vector significantly toward its document context.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does the per-token embedding cost look like?",
                "",
                "We measure the size of the augmented payload so the cost is visible. With a real late-chunking implementation (pooling token embeddings), each chunk costs roughly the same as a classical embedding because the whole-document forward pass amortises across chunks. With our pseudo-implementation, each chunk costs `len(context) + len(chunk)`.",
            ],
            code=[
                "import tiktoken",
                "enc = tiktoken.get_encoding('cl100k_base')",
                "naive_tokens = len(enc.encode(naive_chunks[10].text))",
                "late_tokens = len(enc.encode(late_texts[10]))",
                "print(f'naïve  embed tokens / chunk = {naive_tokens}')",
                "print(f'late  embed tokens / chunk = {late_tokens}  (with context prefix; true late chunking amortises)')",
            ],
        ),
    ],

    run_md=[
        "End-to-end answer on a question whose answer chunk references something defined elsewhere in the paper.",
    ],
    run_code=[
        "q = 'What does the paper say about its linear-time complexity for long sequences?'",
        "ans, ctxs = answer_question(q)",
        "print('=== Late-chunking answer ===')",
        "print(ans)",
        "print()",
        "print('Top context:')",
        "print(ctxs[0][:300])",
    ],

    comparison_md=[
        "Vanilla pipeline vs late chunking on the Mamba corpus. The interesting axis is whether the late-chunked vectors retrieve the same chunks or different chunks; the answer text is often similar, but the *evidence* differs.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What does the paper say about its linear-time complexity for long sequences?'",
        "base = vanilla_pipeline(q, corpus='arxiv-mamba', top_k=5)",
        "ours_a, ours_c = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'top_chunk_start': base.contexts[0][:140]},",
        "    {'pipeline': 'late-chunked', 'top_chunk_start': ours_c[0][:140]},",
        "])",
    ],

    tuning_md=[
        "Four knobs:",
        "",
        "1. **Context window.** We feed the first 2000 chars of the document as context. Bigger windows include more surrounding text but cost more tokens. A true late-chunking implementation pools over the full document with no per-chunk cost.",
        "2. **Chunk size.** Late chunking works best with moderate chunks (256–512 tokens). Very small chunks lose their distinctness; very large chunks defeat the purpose since they already carry their own context.",
        "3. **Embedder choice.** True late chunking needs a long-context embedder (jina-embeddings-v3, Voyage 3, Cohere Embed 4). Pseudo-late-chunking works on any embedder but loses some of the benefit.",
        "4. **Pooling strategy.** Mean-pool is the default. Max-pool can be worse; weighted pooling (e.g. by attention from a CLS token) sometimes beats mean. The Jina paper compares; mean-pool wins most of the time.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Short documents.** A document that fits in a single chunk gains nothing from late chunking. The technique is for long documents whose chunks would otherwise lose surrounding context.",
        "- **Embedder cap.** jina-embeddings-v3 caps at 8K tokens. Books and very long contracts exceed this. Split into sections, then late-chunk within each section.",
        "- **Pseudo vs true late chunking.** Our pseudo implementation prepends document context as text and re-embeds; true late chunking pools token embeddings without re-running the encoder. The pseudo version captures most of the effect but is more expensive at scale. For production at billions of chunks, use the real implementation.",
        "",
        "Compose with contextual retrieval (Recipe 7) — late chunking gives implicit context, contextual headers give explicit position. Both improve retrieval; stacking them is roughly additive on most corpora.",
    ],
)


# =============================================================================
# Semantic Boundary Splitting
# =============================================================================
SEMANTIC_SPLIT = Recipe(
    path="recipes/02-chunking-and-indexing/semantic-boundary-splitting.ipynb",
    title="Semantic Boundary Splitting — Cut Where Meaning Changes",
    category="chunking-and-indexing",
    corpus_filter="wikipedia-superconductors",

    theory_problem=[
        "Fixed-window chunking cuts every N tokens, regardless of topic. A 384-token window that lands mid-paragraph snaps the idea in half; the chunk on one side has the premise, the chunk on the other has the conclusion. Neither chunk on its own answers the question, and both retrieve poorly because their vectors mix two topics.",
        "Semantic boundary splitting walks adjacent sentence embeddings, watches the cosine distance, and cuts at the peaks. Chunks land on topic boundaries. The same paragraph stays in one chunk; the next paragraph starts the next chunk. The resulting chunks each represent one coherent idea, and the embeddings reflect that.",
    ],
    theory_origin=[
        "Semantic chunking was popularised by Greg Kamradt in his 2023 Five Levels of Text Splitting tutorial. The idea — embed sentences, walk the diffs, cut at high-distance points — was implemented in LangChain's `SemanticChunker` (Dec 2023) and LlamaIndex's `SemanticSplitterNodeParser` (Jan 2024). The technique has stayed essentially unchanged since because the core algorithm is simple and the embedder does the hard work.",
        "By 2026 every serious RAG framework ships a semantic chunker. The cookbook's `cookbook.chunkers.semantic_split` uses a percentile-based breakpoint threshold (default p90), which is the standard parameterisation.",
    ],
    theory_landscape=[
        "Four chunking strategies you should know:",
        "",
        "- **Fixed window (Recipe 2 baseline).** Simplest. Cuts every N tokens.",
        "- **Sentence-window (Recipe 9 building block).** Group N sentences, slide one.",
        "- **Semantic boundary (this recipe).** Cut where embeddings disagree.",
        "- **Propositional (Recipe 8).** LLM rewrites text into atomic factual claims.",
        "",
        "Strategies stack: semantic boundary for the initial cut, then propositional within each semantic chunk for retrieval-level granularity. The cookbook's chunkers are designed to compose this way.",
    ],
    theory_when_to_use=[
        "Use semantic boundary splitting for prose with clear topic transitions: Wikipedia articles, blog posts, textbooks. The technique earns its keep when paragraphs are visibly distinct ideas and the corpus is big enough that fixed-window chunks would routinely split paragraphs in half.",
        "Skip it for technical documents with consistent structure: API references, structured datasets, tabular reports. Fixed-window is fine there because every chunk is roughly the same shape and the boundaries are determined by the document's existing markup.",
        "Skip it for very short documents (under 1000 tokens). The overhead of embedding sentences is wasted; just use one chunk per document. Skip it also for code corpora — code's structure does not look like prose to the embedder, and the diff curve is noisy.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**The diff is a topic signal.** When two adjacent sentences talk about the same thing, their embeddings are similar. When they switch topics, the embeddings disagree. The size of the disagreement is a measurable topic boundary signal — concrete enough to threshold on.",
        "",
        "**Percentile threshold is robust.** A hard cosine threshold (e.g. \"cut at 0.7\") fails on different corpora because diff magnitudes vary widely by domain. A percentile (e.g. \"cut at the top 10 percent of diffs\") adapts to the corpus's natural variation and ports between domains.",
        "",
        "**Semantic chunks are usually longer than fixed-window.** A topic that runs three paragraphs gets one semantic chunk and three fixed chunks. The semantic chunk reads more coherently; the fixed chunks fragment the discussion. Embeddings of the semantic chunk represent one idea cleanly, where the fixed chunks each represent partial ideas with imported neighbouring context.",
    ],

    architecture_mermaid="""
flowchart LR
  D[Document] --> S[Sentence split]
  S --> E[Embed each sentence]
  E --> DI[Per-pair cosine<br/>distance]
  DI --> P[Threshold at<br/>p90 percentile]
  P --> C[Cut at peaks]
  C --> CH[Semantic chunks]
""",

    references=[
        Reference(
            title="Five Levels of Text Splitting (Greg Kamradt, 2023)",
            url="https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb",
            kind="repo",
            note="The original tutorial that popularised semantic chunking.",
        ),
        Reference(
            title="LangChain SemanticChunker",
            url="https://python.langchain.com/docs/how_to/semantic-chunker/",
            kind="docs",
            note="Reference implementation in LangChain.",
        ),
        Reference(
            title="LlamaIndex SemanticSplitterNodeParser",
            url="https://developers.llamaindex.ai/python/framework-api-reference/llama_index/core/node_parser/SemanticSplitterNodeParser/",
            kind="docs",
            note="Reference implementation in LlamaIndex.",
        ),
        Reference(
            title="Greg Kamradt — Five Levels of Text Splitting (YouTube)",
            url="https://www.youtube.com/watch?v=8OJC21T2SL4",
            kind="video",
            note="Video walk-through of the tutorial.",
        ),
        Reference(
            title="Late Chunking (Recipe 6)",
            url="https://jina.ai/news/late-chunking-in-long-context-embedding-models/",
            kind="blog",
            note="Different answer to a related problem.",
        ),
        Reference(
            title="Contextual Retrieval (Recipe 7)",
            url="https://www.anthropic.com/news/contextual-retrieval",
            kind="blog",
            note="Stacks well on top of semantic chunks.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load Wikipedia superconductors",
                "",
                "Wikipedia is a good corpus for semantic chunking — articles are made of clear topical paragraphs, and the boundaries are sharp enough to be visible in embedding space.",
            ],
            code=[
                "from cookbook.corpora import load_wikipedia_superconductors",
                "",
                "docs = list(load_wikipedia_superconductors())",
                "print(f'Loaded {len(docs)} articles.')",
                "print(f'First: {docs[0].text[:200]}...')",
            ],
        ),
        CodeStep(
            tag="chunk",
            lead_md=[
                "### Step 2 — Run semantic boundary splitting",
                "",
                "We use `cookbook.chunkers.semantic_split` with a 90th-percentile breakpoint. It sentence-splits, embeds each sentence, computes adjacent diffs, and cuts at the top 10 percent of diffs.",
            ],
            code=[
                "from cookbook.chunkers import semantic_split",
                "",
                "semantic_chunks = semantic_split(docs, embed=client.embed, breakpoint_percentile=90)",
                "print(f'Semantic chunks: {len(semantic_chunks)}')",
                "avg_len = sum(len(c.text.split()) for c in semantic_chunks) // max(1, len(semantic_chunks))",
                "print(f'Avg chunk length (tokens, approx): {avg_len}')",
            ],
        ),
        CodeStep(
            tag="chunk",
            lead_md=[
                "### Step 3 — Build a fixed-window baseline",
                "",
                "Same corpus, fixed-window chunking, similar average chunk length so the comparison is fair.",
            ],
            code=[
                "from cookbook.chunkers import fixed_window",
                "",
                "fixed_chunks = fixed_window(docs, target_tokens=avg_len, overlap_tokens=avg_len // 6)",
                "print(f'Fixed chunks: {len(fixed_chunks)}')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 4 — Embed and index both",
                "",
                "Standard plumbing. Two stores, same embedder, same Qdrant configuration.",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "",
                "sem_v = client.embed([c.text for c in semantic_chunks])",
                "sem_store = QdrantBackend('sem', dim=len(sem_v[0]))",
                "sem_store.add([c.text for c in semantic_chunks], sem_v, ids=[c.chunk_id for c in semantic_chunks])",
                "",
                "fix_v = client.embed([c.text for c in fixed_chunks])",
                "fix_store = QdrantBackend('fix', dim=len(fix_v[0]))",
                "fix_store.add([c.text for c in fixed_chunks], fix_v, ids=[c.chunk_id for c in fixed_chunks])",
                "print('Both stores indexed.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 5 — Compare on a representative query",
                "",
                "A question whose answer should live in a coherent paragraph. Semantic chunking should return that paragraph as one chunk; fixed-window may split it across two adjacent chunks.",
            ],
            code=[
                "q = 'How does flux pinning enable stable levitation in Type-II superconductors?'",
                "qv = client.embed([q])[0]",
                "",
                "print('--- Semantic top-3 ---')",
                "for h in sem_store.search(qv, top_k=3):",
                "    print(f'  score={h.score:.3f}  {h.text[:160]}')",
                "print()",
                "print('--- Fixed-window top-3 ---')",
                "for h in fix_store.search(qv, top_k=3):",
                "    print(f'  score={h.score:.3f}  {h.text[:160]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 6 — Wrap as `answer_question`",
                "",
                "Cookbook contract. Uses the semantic store.",
            ],
            code=[
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    hits = sem_store.search(qv, top_k=k)",
                "    contexts = [h.text for h in hits]",
                "    answer = client.chat(",
                "        'Use these passages:\\n' + '\\n\\n'.join(contexts) + f'\\nQ: {question}\\nA:'",
                "    )",
                "    return answer, contexts",
                "",
                "ans, _ = answer_question('What is the Meissner effect?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does the cosine-diff curve look like?",
                "",
                "Plot the diffs across one article. The peaks are where semantic chunking cuts. Reading the plot tells you whether the threshold is reasonable or you should adjust.",
            ],
            code=[
                "import numpy as np, matplotlib.pyplot as plt",
                "",
                "doc = docs[0]",
                "import re",
                "sents = [s.strip() for s in re.split(r'(?<=[.!?])\\s+', doc.text) if s.strip()][:30]",
                "sv = np.asarray(client.embed(sents), dtype=np.float32)",
                "sv /= np.linalg.norm(sv, axis=1, keepdims=True).clip(min=1e-9)",
                "diffs = 1 - (sv[:-1] * sv[1:]).sum(axis=1)",
                "",
                "fig, ax = plt.subplots(figsize=(7, 3))",
                "ax.plot(diffs, marker='o')",
                "ax.axhline(np.percentile(diffs, 90), color='red', linestyle='--', label='p90 cut threshold')",
                "ax.set_xlabel('Adjacent sentence pair index')",
                "ax.set_ylabel('Cosine distance')",
                "ax.set_title(f'Semantic distance curve — {doc.doc_id}')",
                "ax.legend()",
                "plt.tight_layout()",
                "plt.show()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — semantic vs fixed chunk-length distribution",
                "",
                "Semantic chunks have variable length; fixed are constant. Plot the histograms side by side.",
            ],
            code=[
                "import matplotlib.pyplot as plt",
                "sem_lens = [len(c.text.split()) for c in semantic_chunks]",
                "fix_lens = [len(c.text.split()) for c in fixed_chunks]",
                "",
                "fig, axes = plt.subplots(1, 2, figsize=(8, 3))",
                "axes[0].hist(sem_lens, bins=20)",
                "axes[0].set_title('Semantic chunk lengths')",
                "axes[0].set_xlabel('tokens (approx)')",
                "axes[1].hist(fix_lens, bins=20)",
                "axes[1].set_title('Fixed-window chunk lengths')",
                "plt.tight_layout()",
                "plt.show()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — recall@5 on the eval slice",
                "",
                "Loose recall proxy on both stores. Semantic typically beats fixed by 3–8 points on Wikipedia.",
            ],
            code=[
                "from cookbook.corpora import load_eval_questions",
                "qs = [q for q in load_eval_questions() if q['corpus'] == 'wikipedia-superconductors'][:10]",
                "def recall(s):",
                "    hits = 0",
                "    for q in qs:",
                "        qv = client.embed([q['question']])[0]",
                "        retr = s.search(qv, top_k=5)",
                "        gold = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "        if any(any(w[:6] in r.text.lower() for w in gold) for r in retr):",
                "            hits += 1",
                "    return hits / max(1, len(qs))",
                "print(f'semantic recall@5  = {recall(sem_store):.2f}')",
                "print(f'fixed    recall@5  = {recall(fix_store):.2f}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — sample a semantic chunk and read it",
                "",
                "Eyeballing one semantic chunk tells you whether the chunker landed on a real topic boundary or split mid-thought.",
            ],
            code=[
                "sample = semantic_chunks[10]",
                "print(f'chunk_id: {sample.chunk_id}  doc_id: {sample.doc_id}')",
                "print()",
                "print(sample.text[:800])",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a question that pays off the topic-boundary chunking.",
    ],
    run_code=[
        "q = 'How does the Meissner effect distinguish a superconductor from a perfect conductor?'",
        "ans, ctxs = answer_question(q)",
        "print('=== Semantic answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla baseline vs semantic chunking. Both pipelines are dense over Wikipedia; only the chunker differs.",
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
        "    {'pipeline': 'vanilla (fixed-window)', 'preview': base.contexts[0][:140]},",
        "    {'pipeline': 'semantic', 'preview': ours_c[0][:140]},",
        "])",
    ],

    tuning_md=[
        "Five knobs in priority order:",
        "",
        "1. **Breakpoint percentile.** Default 90. Lower (80) gives more cuts and shorter chunks; higher (95) gives fewer cuts and longer chunks. Sweep on your corpus and watch the resulting chunk-length histogram.",
        "2. **Embedder for the sentence diffs.** A weaker embedder produces noisier diffs and worse cut points. Use the same embedder for chunking that you use for retrieval — the diff signal will match.",
        "3. **Minimum chunk size.** Sometimes a paragraph break creates a one-sentence chunk. Set a floor (`min_sentences=3`) to merge stragglers with neighbours.",
        "4. **Sentence-splitter regex.** The cookbook's default catches `.!?` followed by whitespace and a capital letter. Domains with abbreviations (legal, medical) need a tighter split.",
        "5. **Re-chunk when you change embedders.** The diff curve depends on the embedder, so a new embedder produces different cut points. Plan for a re-chunk + re-embed pass.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Heterogeneous documents.** A document that mixes prose and code blocks has discontinuous embeddings everywhere; semantic chunking over-splits. Pre-split into prose-vs-code segments first.",
        "- **Embedder noise.** Sentence-level diffs are noisier than paragraph-level diffs. Some implementations work at the paragraph level for a smoother curve; the tradeoff is coarser cuts.",
        "- **Single-paragraph documents.** A document that is one long paragraph has no internal diffs to cut on. The chunker returns one chunk; if it is over the embedder window, retrieval still degrades.",
        "",
        "Compose with contextual retrieval (Recipe 7) and parent-child (Recipe 9): semantic chunks for the index, headers for context, parent chunks for generation breadth.",
    ],
)


# =============================================================================
# Proposition Decomposition (Dense X)
# =============================================================================
PROPOSITION = Recipe(
    path="recipes/02-chunking-and-indexing/proposition-decomposition.ipynb",
    title="Proposition Decomposition — Atomic Facts as Chunks",
    category="chunking-and-indexing",
    corpus_filter="wikipedia-superconductors",

    theory_problem=[
        "A paragraph contains five facts. Embed it, retrieve it, and the embedding represents some average of those five facts. A query about fact 3 may not land at the top because facts 1, 2, 4, 5 dilute the signal. Single-vector embeddings give every chunk one slot in semantic space, regardless of how much actual information it carries.",
        "Proposition decomposition cuts each chunk into atomic factual claims using an LLM. Each claim becomes its own retrieval unit with its own vector. Retrieval precision improves because each unit represents one idea. The cost is real — one LLM call per chunk to extract claims — but for factual-recall workloads it is often the cheapest big lift after picking the right embedder.",
    ],
    theory_origin=[
        "Proposition decomposition was named in the Dense X Retrieval paper (Chen et al., 2023). The idea has older roots in information extraction and open-IE; the contribution was framing it as a chunking strategy for RAG and showing recall improvements on TREC, PopQA, and natural-questions benchmarks. The headline number was 5–9 point gains depending on the base retriever.",
        "By 2025 the technique had spread under multiple names — claim extraction, atomic facts, factoid chunking. The cookbook uses Dense X's prompt structure as the reference; all variants converge on the same idea: one fact per retrieval unit, with pronouns resolved to entities. Most production systems pair propositions with a faithfulness check downstream because the extraction step can occasionally invent facts.",
    ],
    theory_landscape=[
        "Proposition chunking sits in tension with parent-child (Recipe 9). Both fragment the chunk into smaller retrieval units; they differ on what the smaller units mean.",
        "",
        "- **Proposition.** Atomic factual claims, possibly rewritten to resolve pronouns. Retrieval-optimal at the unit level; generation reads the propositions directly.",
        "- **Parent-child.** Small chunks (sentences or 100-token windows) for search; the parent (original chunk) is returned for generation context.",
        "",
        "Propositions are sharper for factoid recall; parent-child is friendlier for narrative generation. Pick by query shape.",
    ],
    theory_when_to_use=[
        "Use proposition decomposition on factual corpora where queries ask specific questions: encyclopedia, biomedical literature, knowledge-base articles. Each proposition becomes a near-perfect match for a fact-seeking query.",
        "Skip it on narrative corpora where the value lives in the flow of ideas, not isolated facts. Books, blog posts, policy documents — propositions reduce them to flat lists and lose the connective tissue the model needs to write a good answer.",
        "Skip it when LLM call costs are prohibitive. The decomposition step is one LLM call per chunk; at billions of chunks the bill is real.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**Atomicity is the technique.** A proposition that says \"X is Y because Z\" still bundles two facts; split into two. The cleaner the atomicity, the sharper the retrieval signal.",
        "",
        "**Pronoun resolution is the secret sauce.** A proposition that says \"It was discovered in 1911\" is useless without context. The prompt must instruct the LLM to resolve every pronoun. \"Superconductivity was discovered in 1911 by Heike Kamerlingh Onnes\" is the kind of proposition that retrieves well.",
        "",
        "**Storage explodes.** A 384-token chunk decomposes to 5–10 propositions, so the index is 5–10x bigger. At small scale, free. At 100M chunks, a serious capacity-planning conversation.",
    ],

    architecture_mermaid="""
flowchart LR
  D[Document] --> C[Chunks]
  C --> L[LLM:<br/>decompose to atomic<br/>self-contained claims]
  L --> P[Propositions]
  P --> E[Embed each<br/>proposition]
  E --> S[(Vector store)]
""",

    references=[
        Reference(
            title="Dense X Retrieval — What Retrieval Granularity Should We Use? (Chen et al., 2023)",
            url="https://arxiv.org/abs/2312.06648",
            kind="paper",
            note="The paper that framed propositions as a chunking strategy.",
        ),
        Reference(
            title="LlamaIndex Dense X reference implementation",
            url="https://docs.llamaindex.ai/en/stable/examples/retrievers/dense_x_retrieval/",
            kind="docs",
            note="One-click implementation following the paper.",
        ),
        Reference(
            title="Open Information Extraction overview",
            url="https://en.wikipedia.org/wiki/Open_information_extraction",
            kind="docs",
            note="The information extraction tradition that propositions descend from.",
        ),
        Reference(
            title="Parent-child retrieval (Recipe 9)",
            url="https://developers.llamaindex.ai/python/framework/optimizing/advanced_retrieval/advanced_retrieval/",
            kind="docs",
            note="Alternative answer to the same problem.",
        ),
        Reference(
            title="Contextual Retrieval (Recipe 7)",
            url="https://www.anthropic.com/news/contextual-retrieval",
            kind="blog",
            note="Cousin technique — different way to enrich chunks.",
        ),
        Reference(
            title="Semantic chunking (Recipe 5)",
            url="https://github.com/FullStackRetrieval-com/RetrievalTutorials",
            kind="repo",
            note="Often paired with propositions: semantic boundaries first, then propositions within.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load a small corpus slice",
                "",
                "Proposition decomposition is LLM-heavy. We use six Wikipedia articles to keep the bill modest while still showing the technique.",
            ],
            code=[
                "from cookbook.corpora import load_wikipedia_superconductors",
                "",
                "docs = list(load_wikipedia_superconductors())[:6]",
                "print(f'Working with {len(docs)} articles.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Run the proposition extractor",
                "",
                "We use `cookbook.chunkers.proposition_split` with its default prompt. It walks each document in 600-token windows, asks the LLM for a numbered list of atomic claims with all pronouns resolved, then parses the output.",
            ],
            code=[
                "from cookbook.chunkers import proposition_split",
                "",
                "props = proposition_split(docs, chat=client.chat, window_tokens=600)",
                "print(f'Extracted {len(props)} propositions.')",
                "print()",
                "print('First five propositions:')",
                "for p in props[:5]:",
                "    print(f'  - {p.text}')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 3 — Index the propositions",
                "",
                "Standard. Each proposition is one entry in the store.",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "",
                "vectors = client.embed([p.text for p in props])",
                "store = QdrantBackend('props', dim=len(vectors[0]))",
                "store.add([p.text for p in props], vectors, ids=[p.chunk_id for p in props])",
                "print(f'Indexed {len(props)} propositions.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 4 — Compare with classical chunks",
                "",
                "Build a fixed-window baseline over the same corpus and run the same query against both.",
            ],
            code=[
                "from cookbook.chunkers import fixed_window",
                "",
                "chunks = fixed_window(docs, target_tokens=320, overlap_tokens=32)",
                "chunk_v = client.embed([c.text for c in chunks])",
                "chunk_store = QdrantBackend('classic', dim=len(chunk_v[0]))",
                "chunk_store.add([c.text for c in chunks], chunk_v, ids=[c.chunk_id for c in chunks])",
                "",
                "q = 'Who first observed superconductivity and in what material?'",
                "qv = client.embed([q])[0]",
                "",
                "print('--- Propositions top-3 ---')",
                "for h in store.search(qv, top_k=3):",
                "    print(f'  {h.score:.3f}  {h.text}')",
                "print()",
                "print('--- Classical chunks top-3 ---')",
                "for h in chunk_store.search(qv, top_k=3):",
                "    print(f'  {h.score:.3f}  {h.text[:160]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap as `answer_question`",
                "",
                "Standard contract. The retrieved propositions go straight to the LLM as context. We retrieve more propositions (k=8) than chunks (k=5) because each unit is smaller.",
            ],
            code=[
                "def answer_question(question: str, k: int = 8) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    hits = store.search(qv, top_k=k)",
                "    contexts = [h.text for h in hits]",
                "    answer = client.chat(",
                "        'Use these atomic facts.\\n' + '\\n'.join(contexts) + f'\\nQ: {question}\\nA:'",
                "    )",
                "    return answer, contexts",
                "",
                "ans, _ = answer_question('Who first observed superconductivity and when?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — quality of the extracted propositions",
                "",
                "Sample five propositions and judge whether they are atomic and self-contained. Bad propositions are unresolved pronouns or compound claims.",
            ],
            code=[
                "import random",
                "for p in random.sample(props, 5):",
                "    print(f'  - {p.text}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — propositions per original chunk",
                "",
                "How much does the index inflate? We grouped propositions by their source document; divide by the number of source chunks the document had.",
            ],
            code=[
                "from collections import Counter",
                "by_doc = Counter(p.doc_id for p in props)",
                "for doc_id, n in by_doc.most_common(5):",
                "    print(f'  {doc_id}: {n} propositions')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — does a factoid query retrieve sharper hits?",
                "",
                "Compare scores: with propositions, the top hit's score is usually higher than the top hit's score with classical chunks. That sharper score means downstream rerankers and confidence-thresholding work better.",
            ],
            code=[
                "for q in [",
                "    'In what year was YBCO discovered?',",
                "    'What does Cooper pair mean?',",
                "    'Who proposed BCS theory?',",
                "]:",
                "    qv = client.embed([q])[0]",
                "    p_top = store.search(qv, top_k=1)[0]",
                "    c_top = chunk_store.search(qv, top_k=1)[0]",
                "    print(f'  {q}')",
                "    print(f'    prop  top score = {p_top.score:.3f}')",
                "    print(f'    chunk top score = {c_top.score:.3f}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost in cached embeddings + LLM calls",
                "",
                "Measure how many LLM calls and embeddings the proposition extraction consumed. Useful for cost projection on a real corpus.",
            ],
            code=[
                "from cookbook import _cache",
                "stats = _cache.stats()",
                "print(f'Total cache entries (chat + embed): {stats[\"entries\"]}')",
                "print()",
                "print(f'For this notebook the extractor processed roughly {len(docs)} docs')",
                "print(f'and produced {len(props)} propositions.')",
                "print(f'Estimated cost: ~{len(docs) * 3} LLM calls (one per window per doc) + {len(props)} embeddings.')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a factual query.",
    ],
    run_code=[
        "q = 'What is the Meissner effect, and why does it distinguish a superconductor from a perfect conductor?'",
        "ans, ctxs = answer_question(q)",
        "print('=== Proposition answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla baseline vs proposition chunking on a factual query. Vanilla retrieves prose chunks; propositions retrieve resolved facts.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What is the Meissner effect, and why does it distinguish a superconductor from a perfect conductor?'",
        "base = vanilla_pipeline(q, corpus='wikipedia-superconductors', top_k=5)",
        "ours_a, ours_c = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla (chunks)', 'top_unit': base.contexts[0][:120]},",
        "    {'pipeline': 'propositions', 'top_unit': ours_c[0][:120]},",
        "])",
    ],

    tuning_md=[
        "Five knobs:",
        "",
        "1. **Window size for extraction.** Default 600 tokens. Smaller windows produce more propositions per document and may miss cross-paragraph context. Larger windows are cheaper but produce coarser propositions that mix multiple claims.",
        "2. **Extraction prompt.** The default prompt asks for atomic and self-contained claims. Strengthen the self-containment requirement (\"every entity named in full, no pronouns\") and quality improves visibly.",
        "3. **Extractor model.** Smaller models often extract worse propositions. A Llama-3.3-70B or GPT-4o is typically the right tier; smaller models miss entities or under-decompose.",
        "4. **Retrieval `k`.** Bump to 8–12 since each unit is smaller. The prompt is still small even at k=10 because propositions are one sentence each.",
        "5. **Verification pass.** For factual workloads, run a separate LLM call per proposition asking \"is this claim supported by the source?\" before indexing. Drops invented propositions cheaply.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Hallucination during extraction.** A model that over-summarises will invent facts. Always pair propositions with a faithfulness check (Recipe 37 or 40).",
        "- **Numerical drift.** Propositions about quantities (years, currencies, measurements) sometimes lose precision. Hand-spot-check 50 propositions before going to production.",
        "- **Index explosion.** A 1M-chunk corpus becomes 5–10M propositions. Storage and embedding cost scale linearly with the multiplier. Plan capacity before deploying.",
        "",
        "Compose with semantic boundary splitting (Recipe 5) — semantic chunks first to find coherent topical regions, then propositions within each. Compose with reranking (Recipe 22) — the sharper retrieval signals reward reranker stages.",
    ],
)


# =============================================================================
# Sentence-Window + Parent-Child Retrieval
# =============================================================================
PARENT_CHILD = Recipe(
    path="recipes/02-chunking-and-indexing/sentence-window-and-parent-child.ipynb",
    title="Sentence-Window and Parent-Child — Small to Find, Big to Read",
    category="chunking-and-indexing",
    corpus_filter="rust-book",

    theory_problem=[
        "Small chunks find precisely; large chunks generate well. A chunk that contains exactly the sentence answering the question ranks first under cosine search, but the LLM gets one sentence and no context. A chunk that contains the answer plus three paragraphs of context generates a richer answer, but its embedding is diluted and it may not rank first.",
        "Parent-child retrieval gives you both. Index small chunks for retrieval. When a small chunk is retrieved, return its bigger parent chunk to the generator. The retrieval score reflects the small unit; the generation context reflects the surrounding prose.",
    ],
    theory_origin=[
        "Parent-child retrieval was shipped in LlamaIndex's `HierarchicalNodeParser` in mid-2023 and in LangChain's `ParentDocumentRetriever` shortly after. Neither paper coined the term; the technique was an obvious composition of small-window and big-window retrieval that emerged as developers iterated on RAG demos.",
        "By 2026 it is one of the most quietly important patterns in production RAG: systems that look like \"just RAG\" are almost always doing some form of small-to-big. The pattern fits cleanly under reranking and adaptive layers, which is part of why it has stayed relevant for three years without significant updates.",
    ],
    theory_landscape=[
        "Sentence-window is a parent-child variant: children are single sentences, parents are 3-sentence neighbourhoods. Same idea, different aspect ratio.",
        "",
        "Three cousins to know:",
        "",
        "- **Parent-child (this recipe).** Two levels: child for retrieval, parent for generation.",
        "- **RAPTOR (Recipe 10).** Many levels: leaves, summary, summary-of-summaries.",
        "- **Document-summary routing (Recipe 11).** Top level is a per-document summary; bottom level is chunks within the chosen document.",
        "",
        "Compose with everything: semantic chunks for the parent (Recipe 5), contextual headers on the parent (Recipe 7), propositions instead of sentences as children (Recipe 8). The cookbook's `cookbook.chunkers.parent_child` returns both lists; the rest is composition.",
    ],
    theory_when_to_use=[
        "Use parent-child whenever your generated answers benefit from context that a small retrieval unit alone would lack. Open-ended Q&A, summarisation, technical explanation — all good fits, because the model needs surrounding paragraphs to produce a confident, complete answer.",
        "Skip it when generation is one-token (classifier-style RAG) or single-sentence (extractive QA). The parent context is wasted because the model only needs the matched sentence to produce its output.",
        "Skip it when the parent is too big for your generation budget. If your parent is 4000 tokens and you retrieve k=5, your prompt is 20k tokens before the question. Resize down or use sentence-window as the children to make room.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**Retrieval and generation want different chunk sizes.** They are competing constraints. Parent-child decouples them so each side can have what it wants — small for precision-of-ranking, big for richness-of-context.",
        "",
        "**The parent doesn't need to be perfectly chosen.** A reasonable parent (the sentence's enclosing paragraph, or a 1000-token window centered on the matched child) is enough. You don't need to find the \"right\" parent — any parent that includes the child plus a few hundred tokens of surrounding context will do, because the generator can read past irrelevant material.",
        "",
        "**Deduplication is essential.** When several children of the same parent get retrieved, you don't want to return the parent five times. Deduplicate by parent ID after collecting children; this is the only non-trivial piece of glue in the whole pattern.",
    ],

    architecture_mermaid="""
flowchart TB
  D[Documents] --> P[Parent chunks<br/>~1000 tokens]
  P --> C[Child chunks<br/>~200 tokens]
  C --> S[(Vector store)]
  Q[Query] --> R[Retrieve top-N children]
  S --> R
  R --> DD[Dedupe by<br/>parent id]
  DD --> G[Return parents<br/>to generator]
""",

    references=[
        Reference(
            title="LlamaIndex HierarchicalNodeParser",
            url="https://developers.llamaindex.ai/python/framework-api-reference/llama_index/core/node_parser/HierarchicalNodeParser/",
            kind="docs",
            note="Reference implementation.",
        ),
        Reference(
            title="LangChain ParentDocumentRetriever",
            url="https://python.langchain.com/docs/how_to/parent_document_retriever/",
            kind="docs",
            note="The cousin implementation.",
        ),
        Reference(
            title="LlamaIndex small-to-big tutorial",
            url="https://developers.llamaindex.ai/python/examples/node_postprocessor/AutoMergingRetrieverDemo/",
            kind="docs",
            note="Walk-through of the pattern.",
        ),
        Reference(
            title="Propositional chunking paper (Recipe 8)",
            url="https://arxiv.org/abs/2312.06648",
            kind="paper",
            note="Alternative answer to the small-unit retrieval question.",
        ),
        Reference(
            title="RAPTOR (Recipe 10)",
            url="https://arxiv.org/abs/2401.18059",
            kind="paper",
            note="A multi-level extension of parent-child.",
        ),
        Reference(
            title="Anthropic Contextual Retrieval",
            url="https://www.anthropic.com/news/contextual-retrieval",
            kind="blog",
            note="Composes well with parent-child.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load the Rust book",
                "",
                "Long-form technical prose is the natural home of parent-child. Chapters are long; a small chunk gives precise retrieval; the chapter context is what the LLM needs to answer well.",
            ],
            code=[
                "from cookbook.corpora import load_rust_book",
                "",
                "docs = list(load_rust_book())[:10]",
                "print(f'Loaded {len(docs)} chapters.')",
            ],
        ),
        CodeStep(
            tag="chunk",
            lead_md=[
                "### Step 2 — Build aligned children and parents",
                "",
                "`cookbook.chunkers.parent_child` returns three things: children for retrieval, parents for generation, and a child-to-parent mapping.",
            ],
            code=[
                "from cookbook.chunkers import parent_child",
                "",
                "children, parents, mapping = parent_child(docs, parent_tokens=900, child_tokens=180, overlap=24)",
                "parents_by_id = {p.chunk_id: p for p in parents}",
                "print(f'{len(children)} children -> {len(parents)} parents')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 3 — Index the children",
                "",
                "Only the children get embedded and stored. Parents live in a Python dictionary keyed by ID.",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "",
                "vectors = client.embed([c.text for c in children])",
                "store = QdrantBackend('pc', dim=len(vectors[0]))",
                "store.add([c.text for c in children], vectors, ids=[c.chunk_id for c in children])",
                "print(f'Indexed {len(children)} children. Parents kept in memory.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 4 — Retrieve, then walk to parents",
                "",
                "Retrieve children, look up their parents, deduplicate. The deduplication step matters: multiple children of the same parent are common.",
            ],
            code=[
                "def retrieve_parents(question: str, top_k: int = 4):",
                "    qv = client.embed([question])[0]",
                "    hits = store.search(qv, top_k=top_k * 3)",
                "    seen, parent_ids = set(), []",
                "    for h in hits:",
                "        pid = mapping.get(h.doc_id)",
                "        if pid and pid not in seen:",
                "            seen.add(pid)",
                "            parent_ids.append(pid)",
                "        if len(parent_ids) >= top_k:",
                "            break",
                "    return [parents_by_id[i] for i in parent_ids]",
                "",
                "for p in retrieve_parents('What is selective scan?'):",
                "    print(f'  parent {p.chunk_id} ({len(p.text)} chars):')",
                "    print(f'    {p.text[:200]}...')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap as `answer_question`",
                "",
                "Standard contract. Generation uses the parents.",
            ],
            code=[
                "def answer_question(question: str, k: int = 3) -> tuple[str, list[str]]:",
                "    parents = retrieve_parents(question, top_k=k)",
                "    contexts = [p.text for p in parents]",
                "    answer = client.chat(",
                "        'Use these passages.\\n' + '\\n\\n'.join(contexts) + f'\\nQ: {question}\\nA:'",
                "    )",
                "    return answer, contexts",
                "",
                "ans, _ = answer_question('Walk through how the borrow checker reasons about overlapping references.')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — child vs parent length",
                "",
                "Quick visual: children are short, parents are long. The asymmetry is the whole point.",
            ],
            code=[
                "import matplotlib.pyplot as plt",
                "fig, axes = plt.subplots(1, 2, figsize=(8, 3))",
                "axes[0].hist([len(c.text.split()) for c in children], bins=20)",
                "axes[0].set_title('Child chunk lengths')",
                "axes[1].hist([len(p.text.split()) for p in parents], bins=20)",
                "axes[1].set_title('Parent chunk lengths')",
                "plt.tight_layout()",
                "plt.show()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — fan-in: how many children per parent?",
                "",
                "If parents have 1 child each, there's no deduplication work to do — but you probably could have used the parent directly. If parents have many children, deduplication matters a lot.",
            ],
            code=[
                "from collections import Counter",
                "fan_in = Counter(mapping.values())",
                "print(f'Average children per parent: {sum(fan_in.values()) / max(1, len(fan_in)):.1f}')",
                "print(f'Max children per parent: {max(fan_in.values())}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — top-k matched children for one query",
                "",
                "Look at the children that win retrieval. Do they all live in the same parent? Or are they spread across parents?",
            ],
            code=[
                "q = 'How does the borrow checker reason about overlapping references?'",
                "qv = client.embed([q])[0]",
                "hits = store.search(qv, top_k=10)",
                "for h in hits:",
                "    pid = mapping.get(h.doc_id, '?')",
                "    print(f'  score={h.score:.3f}  parent={pid}  child={h.doc_id}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — recall@5 with vs without parent walk",
                "",
                "Loose recall proxy. Children-only retrieval should beat parents-only for sharpness, but the difference is small on prose corpora.",
            ],
            code=[
                "from cookbook.corpora import load_eval_questions",
                "qs = [q for q in load_eval_questions() if q['corpus'] == 'rust-book'][:8]",
                "def recall(retr_fn):",
                "    hits = 0",
                "    for q in qs:",
                "        out = retr_fn(q['question'])",
                "        gold = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "        if any(any(w[:6] in t.lower() for w in gold) for t in out):",
                "            hits += 1",
                "    return hits / max(1, len(qs))",
                "",
                "def via_children(q):",
                "    qv = client.embed([q])[0]",
                "    return [h.text for h in store.search(qv, top_k=5)]",
                "def via_parents(q):",
                "    return [p.text for p in retrieve_parents(q, top_k=3)]",
                "print(f'children-only recall@5 = {recall(via_children):.2f}')",
                "print(f'parent-walk   recall@5 = {recall(via_parents):.2f}')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a question that benefits from parent context.",
    ],
    run_code=[
        "ans, ctxs = answer_question('When should I prefer Arc over Rc, and what guarantees do I lose if I switch?')",
        "print('=== Parent-child answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla baseline (flat fixed-window) vs parent-child. The interesting case is a question where vanilla finds the right paragraph but the answer needs surrounding paragraphs to write well.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'When should I prefer Arc over Rc, and what guarantees do I lose if I switch?'",
        "base = vanilla_pipeline(q, corpus='rust-book', top_k=5)",
        "ours_a, ours_c = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'avg_ctx_chars': sum(len(c) for c in base.contexts) // max(1, len(base.contexts))},",
        "    {'pipeline': 'parent-child', 'avg_ctx_chars': sum(len(c) for c in ours_c) // max(1, len(ours_c))},",
        "])",
    ],

    tuning_md=[
        "Five knobs in priority order:",
        "",
        "1. **Parent and child sizes.** Defaults of 900 / 180 are reasonable for prose. For code or tables, both should be bigger. For short factual content, both should be smaller. The ratio matters as much as absolute sizes.",
        "2. **Child-to-parent fan-in.** A parent with 50 children is fine but means deduplication eats latency. Trim by widening children or narrowing parents.",
        "3. **Number of children to over-retrieve.** Our function retrieves `k*3` children to dedupe down to `k` parents. Higher multipliers improve recall at the cost of one more embedding lookup.",
        "4. **Reranker on children before walking.** Without reranking, noisy children promote noisy parents. A cross-encoder rerank (Recipe 22) on the child list before walking is the single biggest quality lift.",
        "5. **Parent boundary alignment.** Semantic chunking for the parents (Recipe 5) avoids the mid-paragraph parent problem and improves generation quality more than any other change.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Skinny prompts.** With k=3 parents at 900 tokens each, you have 2700 tokens of context per query. If your model supports it, great; if it caps at 4k, you have no headroom for instructions.",
        "- **Mismatched parent boundaries.** A child whose parent ends mid-paragraph still carries a half-thought to the LLM. Use semantic chunking for the parents to avoid this.",
        "- **No reranking before parent walk.** If you walk parents from the top-N children unranked, noisy children promote noisy parents. Insert a reranker (Recipe 22) on the children before walking.",
        "",
        "Compose with semantic boundary chunking (Recipe 5) for the parents, contextual headers (Recipe 7) on the parents, and reranking (Recipe 22) on the children. That stack is what production RAG looks like in 2026.",
    ],
)


# =============================================================================
# RAPTOR — Recursive abstractive trees
# =============================================================================
RAPTOR = Recipe(
    path="recipes/02-chunking-and-indexing/raptor-trees.ipynb",
    title="RAPTOR — Recursive Abstractive Trees",
    category="chunking-and-indexing",
    corpus_filter="rust-book",

    theory_problem=[
        "Chunks are flat. A 384-token chunk represents 384 tokens. A question that asks \"across the whole document, what is the recurring theme?\" cannot be answered from any single chunk because no chunk *is* the recurring theme. The recurring theme lives one level above the chunks, in the document's abstract structure.",
        "RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) builds a tree. Leaves are chunks. The next level up is LLM-written summaries of clusters of leaves. The next level is summaries of summaries. Retrieval can land at any level. Specific questions hit leaves; abstract questions hit higher-level summaries.",
    ],
    theory_origin=[
        "RAPTOR was published at ICLR 2024 by Sarthi et al. at Stanford. The paper showed that retrieval over a tree beat retrieval over leaves on multi-hop benchmarks (QASPER, QuALITY) by 6–10 points. The clustering algorithm — Gaussian mixture with BIC for k selection — is the most complex part of the paper; the abstractive summarisation step is straightforward.",
        "The cookbook's implementation uses k-means with a small k as a simpler stand-in. The abstractive summarisation step matches the paper's prompt structure. Recall lifts are smaller than the paper reports because our corpus is smaller, but the technique reads the same way.",
    ],
    theory_landscape=[
        "RAPTOR is the multi-level extension of parent-child (Recipe 9). Parent-child has two levels: child and parent. RAPTOR has many: leaves, level-1 summaries, level-2 summaries, ad infinitum (in practice 2–3 levels). The tree is built bottom-up via clustering and abstractive summarisation.",
        "",
        "Related techniques in the cookbook:",
        "",
        "- **Document summary routing (Recipe 11).** Top level is the document summary; lower levels are sub-chunks within the chosen document. Simpler than RAPTOR but only handles document-level routing.",
        "- **GraphRAG (Recipe 31).** Replaces the linear hierarchy with a knowledge graph; community-level summaries are the equivalent of higher tree levels. Better for cross-document reasoning at heavy computational cost.",
        "- **Parent-child (Recipe 9).** Two-level RAPTOR essentially. Cheaper, lower ceiling.",
    ],
    theory_when_to_use=[
        "Use RAPTOR when your queries mix specific and abstract. Research-paper Q&A, book-length corpora, anything where some questions are \"what is X\" and others are \"how does the work fit together\".",
        "Skip it for narrow corpora where every question is at the same level of abstraction. FAQ Q&A bots have nothing to gain from a hierarchy.",
        "Skip it when LLM summarisation cost dominates your budget. Building the tree costs roughly one LLM call per cluster per level — manageable for thousands of chunks, expensive at billions.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**Each tree level is a different lens.** Leaves are word-level fact retrieval; level 1 is paragraph-level synthesis; level 2 is section-level themes. The retriever picks the level by matching the query's abstraction.",
        "",
        "**Abstraction is built by summarisation.** Each summary node is generated by prompting an LLM with its children's text. The summary inherits the children's information but rewords it at a higher level. The vector for the summary lives in a different part of embedding space than the vectors for the children.",
        "",
        "**Clustering needs to find topical groups.** K-means is a fine default; the paper's BIC-based GMM is slightly better but slower. The cookbook uses k-means with `n_clusters=max(2, len(leaves)//6)`.",
    ],

    architecture_mermaid="""
flowchart TB
  L0[Leaf chunks] --> KC[K-means cluster]
  KC --> LS[LLM:<br/>summarise each cluster]
  LS --> L1[Level 1 summaries]
  L1 --> KC2[K-means cluster]
  KC2 --> LS2[LLM:<br/>summarise]
  LS2 --> L2[Level 2 summaries]
  L0 --> S[(Vector store<br/>all levels)]
  L1 --> S
  L2 --> S
  Q[Query] --> R[Search all levels]
  S --> R
""",

    references=[
        Reference(
            title="RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval (Sarthi et al., 2024)",
            url="https://arxiv.org/abs/2401.18059",
            kind="paper",
            note="The ICLR 2024 paper.",
        ),
        Reference(
            title="LlamaIndex RAPTOR pack",
            url="https://docs.llamaindex.ai/en/stable/examples/retrievers/raptor/",
            kind="docs",
            note="Reference implementation.",
        ),
        Reference(
            title="Stanford NLP Group RAPTOR repository",
            url="https://github.com/parthsarthi03/raptor",
            kind="repo",
            note="The paper's code.",
        ),
        Reference(
            title="LangChain RAPTOR cookbook",
            url="https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb",
            kind="repo",
            note="LangChain's implementation walk-through.",
        ),
        Reference(
            title="Document Summary Index (Recipe 11)",
            url="https://developers.llamaindex.ai/python/framework-api-reference/llama_index/core/indices/DocumentSummaryIndex/",
            kind="docs",
            note="Related two-level pattern.",
        ),
        Reference(
            title="GraphRAG (Recipe 31)",
            url="https://github.com/microsoft/graphrag",
            kind="repo",
            note="Graph-structured alternative to RAPTOR's tree.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load a slice of the corpus",
                "",
                "We use 8 Rust chapters. RAPTOR is LLM-heavy, so a small corpus keeps the bill modest.",
            ],
            code=[
                "from cookbook.corpora import load_rust_book",
                "",
                "docs = list(load_rust_book())[:8]",
                "print(f'{len(docs)} chapters.')",
            ],
        ),
        CodeStep(
            tag="chunk",
            lead_md=[
                "### Step 2 — Build the leaf chunks",
                "",
                "Standard chunking for the leaves. Fixed-window is fine here; semantic chunks would also work.",
            ],
            code=[
                "from cookbook.chunkers import fixed_window",
                "",
                "leaves = fixed_window(docs, target_tokens=400, overlap_tokens=40)",
                "print(f'{len(leaves)} leaf chunks.')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 3 — Embed the leaves",
                "",
                "We need the embeddings to drive clustering.",
            ],
            code=[
                "import numpy as np",
                "leaf_vecs = np.asarray(client.embed([l.text for l in leaves]), dtype=np.float32)",
                "print(f'Leaf vectors: {leaf_vecs.shape}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Cluster and summarise at level 1",
                "",
                "K-means with k chosen to be roughly 1/6 of the leaves. Each cluster becomes one level-1 summary.",
            ],
            code=[
                "from sklearn.cluster import KMeans",
                "",
                "def cluster_and_summarise(texts, vectors, n_clusters):",
                "    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=42).fit(vectors)",
                "    summaries = []",
                "    for c in range(n_clusters):",
                "        members = [t for t, lbl in zip(texts, km.labels_) if lbl == c]",
                "        if not members:",
                "            continue",
                "        sample = '\\n\\n'.join(members[:5])[:4000]",
                "        s = client.chat(",
                "            'Summarize the recurring themes across these passages in 4 sentences. '",
                "            'Stay faithful to the source.\\n\\n' + sample",
                "        )",
                "        summaries.append(s)",
                "    summary_vecs = np.asarray(client.embed(summaries), dtype=np.float32)",
                "    return summaries, summary_vecs",
                "",
                "L1_n = max(2, len(leaves) // 6)",
                "L1_summaries, L1_vecs = cluster_and_summarise([l.text for l in leaves], leaf_vecs, n_clusters=L1_n)",
                "print(f'L1 summaries: {len(L1_summaries)}')",
                "print()",
                "print('First L1 summary:')",
                "print(L1_summaries[0][:400])",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Level 2 summaries",
                "",
                "Summarise the level-1 summaries. The tree gets a third floor.",
            ],
            code=[
                "L2_n = max(2, len(L1_summaries) // 3)",
                "L2_summaries, L2_vecs = cluster_and_summarise(L1_summaries, L1_vecs, n_clusters=L2_n)",
                "print(f'L2 summaries: {len(L2_summaries)}')",
                "print()",
                "print(f'Tree: {len(leaves)} leaves -> {len(L1_summaries)} L1 -> {len(L2_summaries)} L2')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 6 — Index all levels in one store",
                "",
                "Mix leaves and summaries in one Qdrant collection. Retrieval picks the best at any level.",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "",
                "all_texts = [l.text for l in leaves] + L1_summaries + L2_summaries",
                "all_vecs = np.vstack([leaf_vecs, L1_vecs, L2_vecs]).tolist()",
                "all_ids = (",
                "    [f'L0-{i}' for i in range(len(leaves))] +",
                "    [f'L1-{i}' for i in range(len(L1_summaries))] +",
                "    [f'L2-{i}' for i in range(len(L2_summaries))]",
                ")",
                "store = QdrantBackend('raptor', dim=len(all_vecs[0]))",
                "store.add(all_texts, all_vecs, ids=all_ids)",
                "print(f'Indexed {len(all_texts)} nodes total.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 7 — Search the tree",
                "",
                "Standard search; the retrieval engine doesn't know the tree exists. It just picks the top-k vectors. Specific queries hit leaves; abstract queries hit summaries.",
            ],
            code=[
                "q = 'Across the chapters here, what is the general pattern Rust uses for optional or fallible computation?'",
                "qv = client.embed([q])[0]",
                "for h in store.search(qv, top_k=6):",
                "    print(f'  {h.doc_id:8s}  score={h.score:.3f}  {h.text[:140]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 8 — Wrap as `answer_question`",
                "",
                "Same cookbook contract.",
            ],
            code=[
                "def answer_question(question: str, k: int = 6) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    hits = store.search(qv, top_k=k)",
                "    contexts = [h.text for h in hits]",
                "    answer = client.chat(",
                "        'Use these passages.\\n' + '\\n\\n'.join(contexts) + f'\\nQ: {question}\\nA:'",
                "    )",
                "    return answer, contexts",
                "",
                "ans, _ = answer_question('How does Rust express optional and fallible computation across the language?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — which level wins for which question?",
                "",
                "Specific questions should hit leaves; abstract questions should hit summaries. Verify.",
            ],
            code=[
                "for q in [",
                "    'What is the difference between Rc and Arc?',         # specific -> leaf",
                "    'What recurring patterns appear across the Rust chapters?',  # abstract -> L1/L2",
                "    'How does Rust handle errors in general?',            # mid -> L1",
                "]:",
                "    qv = client.embed([q])[0]",
                "    top = store.search(qv, top_k=3)",
                "    print(f'  {q[:60]}:')",
                "    for h in top:",
                "        print(f'    {h.doc_id}  score={h.score:.3f}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — distribution of hits by level",
                "",
                "Across a battery of queries, count how often each level wins. A healthy tree has each level represented.",
            ],
            code=[
                "from collections import Counter",
                "battery = [",
                "    'What is Rc?', 'When should I use Arc?', 'What is borrow checking?',",
                "    'How does Rust handle errors?', 'What patterns appear across the book?',",
                "    'How does Rust express concurrency?', 'What is a trait?',",
                "    'How is async different from threads?',",
                "]",
                "level_hits = Counter()",
                "for q in battery:",
                "    qv = client.embed([q])[0]",
                "    top = store.search(qv, top_k=1)[0]",
                "    level_hits[top.doc_id.split('-')[0]] += 1",
                "print(dict(level_hits))",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — read one level-1 summary",
                "",
                "Sanity-check the summary quality. A bad summary will be retrieved but won't help generation.",
            ],
            code=[
                "import random",
                "print(random.choice(L1_summaries))",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — build cost",
                "",
                "Count LLM calls and embeddings used to build the tree.",
            ],
            code=[
                "calls = len(L1_summaries) + len(L2_summaries)",
                "embeds = len(leaves) + len(L1_summaries) + len(L2_summaries)",
                "print(f'LLM summarisations: {calls}')",
                "print(f'Total embeddings : {embeds}')",
                "print('All cached, so re-running this notebook is free.')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a synthesis question.",
    ],
    run_code=[
        "ans, ctxs = answer_question('What is the consistent way Rust expresses fallibility throughout the book?')",
        "print('=== RAPTOR answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla baseline vs RAPTOR. On a synthesis question, vanilla retrieves leaves and the model must synthesise; RAPTOR retrieves a higher-level summary that already did the synthesis.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What is the consistent way Rust expresses fallibility throughout the book?'",
        "base = vanilla_pipeline(q, corpus='rust-book', top_k=5)",
        "ours_a, ours_c = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'raptor', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Four knobs:",
        "",
        "1. **Number of levels.** Two (leaves + L1) catches most of the value; three (L0/L1/L2) is the canonical RAPTOR shape. More than three is rarely useful because the top-level summaries become abstract enough to lose grounding.",
        "2. **Cluster count per level.** Heuristic: `len(parent_level) // 6` for L1, `// 3` for L2. Tune by inspecting cluster coherence — if clusters are obviously mixing topics, lower `n_clusters` and re-run.",
        "3. **Summary prompt.** Faithfulness is the priority. A summary that adds facts not in the source is poison for downstream generation. \"Stay faithful to the source\" in the prompt is not optional.",
        "4. **Clustering algorithm.** k-means is simple. The paper uses Gaussian mixture with BIC; the quality lift is small in practice.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Summary hallucination.** The summariser invents facts. Always check faithfulness on a sample before deploying.",
        "- **Cluster degeneracy.** k-means with too-large k produces 1-leaf clusters. Cap k at `n_leaves / 4` to avoid this.",
        "- **Retrieval favours summaries.** Summaries are short and well-formed, so they often score high on cosine. If your eval set is specific-question-heavy, this can dilute leaf retrieval. Tune by retrieving more total `k` so leaves still get a seat.",
        "",
        "Compose with semantic boundary chunking (Recipe 5) for the leaves and contextual headers (Recipe 7) on the leaves. RAPTOR is a multi-level retrieval architecture; the level-0 chunks themselves can be as fancy as your other recipes make them.",
    ],
)


# =============================================================================
# Document Summary Routing
# =============================================================================
DOC_SUMMARY = Recipe(
    path="recipes/02-chunking-and-indexing/document-summary-routing.ipynb",
    title="Document Summary Routing — Find the Document First",
    category="chunking-and-indexing",
    corpus_filter="wikipedia-superconductors",

    theory_problem=[
        "When your corpus is dozens of distinct documents (one document per topic), a query often only really cares about one document. Flat retrieval scatters chunks across all documents and pulls in distractors from neighbouring documents that happen to share vocabulary. The search space becomes dominated by topical noise, and the answer chunk has to win against thousands of other chunks instead of dozens.",
        "The pattern: build a per-document summary, index those summaries. At query time, retrieve the top-N documents from the summary index, then run chunk-level retrieval inside those documents only. The second retrieval is much sharper because the search space is 1–3 documents instead of dozens. The router pays one extra embedding lookup; in exchange the chunk retrieval works against an order of magnitude smaller pool.",
    ],
    theory_origin=[
        "LlamaIndex shipped `DocumentSummaryIndex` in mid-2023. The pattern existed in production systems before that — most enterprise search engines route on metadata or document classifiers — but LlamaIndex packaged the LLM-summary-as-router idea cleanly.",
        "By 2025 it had become one of the standard moves for multi-document corpora. Combined with adaptive routing (Recipe 26), it lets a system handle queries across hundreds of distinct knowledge sources with one consistent stack.",
    ],
    theory_landscape=[
        "Three routing strategies in this cookbook:",
        "",
        "- **Document summaries (this recipe).** LLM-written one-paragraph summary per document. Retrieve by query. Best when documents are clearly topical.",
        "- **Semantic router (Recipe 17).** Classifier with hand-curated example phrases per route. Cheaper, no LLM at query time, but the route definitions are hand-crafted.",
        "- **Metadata filter (Recipe 20).** Structured filters on metadata fields. Free at query time, requires explicit metadata to exist on the chunks.",
        "",
        "Stack them: semantic router to pick a coarse domain, document-summary to pick a specific document inside the domain, then chunk-level retrieval. Each layer cuts the search space by roughly an order of magnitude, which is why this combination scales so well to enterprise corpora.",
    ],
    theory_when_to_use=[
        "Use document summary routing when the corpus is dozens-to-thousands of separable documents. Wikipedia articles per topic, product knowledge bases per product, legal cases per case number.",
        "Skip it when your corpus is one long document (a book, a 10-K). There is only one document to route to.",
        "Skip it when document boundaries do not match query boundaries. If users routinely ask questions that span multiple documents, the routing layer becomes a bottleneck.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**The summary index is small.** One vector per document. Even at 100k documents you have a tiny index that searches in microseconds. The cost is the one-time summarisation pass; everything afterwards is cheap.",
        "",
        "**The inner retrieval is sharp.** Once you have narrowed to 1–3 documents, chunk-level retrieval is competing against tens of chunks, not tens of thousands. Recall jumps mechanically because the search space is smaller and the chunks are topically coherent.",
        "",
        "**Missing-the-router is unrecoverable.** If the router routes wrong, no amount of inner retrieval helps. Always fall back to flat retrieval when the router score is below a threshold; the fallback is the safety net for queries that genuinely span documents or don't fit the corpus.",
    ],

    architecture_mermaid="""
flowchart TB
  Q[Query] --> R[Summary index<br/>top-N documents]
  D[Documents] --> SUM[Per-doc summary]
  SUM --> R
  R --> CH[Chunks of the<br/>chosen documents]
  CH --> S[Chunk retrieval]
  S --> A[Top-k chunks]
""",

    references=[
        Reference(
            title="LlamaIndex DocumentSummaryIndex",
            url="https://developers.llamaindex.ai/python/framework-api-reference/llama_index/core/indices/DocumentSummaryIndex/",
            kind="docs",
            note="Reference implementation.",
        ),
        Reference(
            title="LlamaIndex Router Query Engine",
            url="https://developers.llamaindex.ai/python/framework/module_guides/querying/router/",
            kind="docs",
            note="The general routing pattern.",
        ),
        Reference(
            title="Adaptive RAG (Recipe 26)",
            url="https://arxiv.org/abs/2403.14403",
            kind="paper",
            note="Cousin technique — routes by query class instead of by document.",
        ),
        Reference(
            title="Semantic Router library",
            url="https://github.com/aurelio-labs/semantic-router",
            kind="repo",
            note="The lighter-weight routing alternative used in Recipe 17.",
        ),
        Reference(
            title="Hierarchical retrieval — LlamaIndex docs",
            url="https://developers.llamaindex.ai/python/framework/optimizing/advanced_retrieval/advanced_retrieval/",
            kind="docs",
            note="General hierarchical retrieval patterns.",
        ),
        Reference(
            title="RAPTOR (Recipe 10)",
            url="https://arxiv.org/abs/2401.18059",
            kind="paper",
            note="Different way to multi-level a corpus.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load the corpus",
                "",
                "Wikipedia superconductors is perfect: many articles, each on a distinct topic. A query about \"Cooper pairs\" really only cares about the Cooper-pair article and maybe BCS-theory.",
            ],
            code=[
                "from cookbook.corpora import load_wikipedia_superconductors",
                "",
                "docs = list(load_wikipedia_superconductors())",
                "print(f'{len(docs)} articles.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Generate per-document summaries",
                "",
                "One LLM call per document. We ask for a three-sentence summary; that is enough signal for routing without burning tokens. With caching on, re-running this is free.",
            ],
            code=[
                "summaries = []",
                "for d in docs:",
                "    s = client.chat('Summarize this article in 3 sentences:\\n\\n' + d.text[:4000])",
                "    summaries.append((d, s.strip()))",
                "print(f'Generated {len(summaries)} summaries.')",
                "print()",
                "print('First summary:')",
                "print(summaries[0][1])",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 3 — Build the router (summary index)",
                "",
                "Tiny store, one vector per document.",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "",
                "sum_v = client.embed([s for _, s in summaries])",
                "router = QdrantBackend('doc-summary', dim=len(sum_v[0]))",
                "router.add(",
                "    [s for _, s in summaries],",
                "    sum_v,",
                "    ids=[d.doc_id for d, _ in summaries],",
                ")",
                "print(f'Router indexed.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 4 — Build chunk retrieval inside selected documents",
                "",
                "When the router picks documents, we chunk just those and build a small ad-hoc store. For very large corpora you would pre-build all per-document stores; for our notebook this on-the-fly approach is fine.",
            ],
            code=[
                "from cookbook.chunkers import sentence_window",
                "",
                "by_id = {d.doc_id: d for d, _ in summaries}",
                "",
                "def route_and_retrieve(question: str, top_docs: int = 2, top_chunks: int = 5):",
                "    qv = client.embed([question])[0]",
                "    doc_hits = router.search(qv, top_k=top_docs)",
                "    picks = [by_id[h.doc_id] for h in doc_hits if h.doc_id in by_id]",
                "    chunks = sentence_window(picks, sentences_per_chunk=4)",
                "    chunk_v = client.embed([c.text for c in chunks])",
                "    inner = QdrantBackend(f'routed-{abs(hash(question)) % 9999}', dim=len(chunk_v[0]))",
                "    inner.add([c.text for c in chunks], chunk_v, ids=[c.chunk_id for c in chunks])",
                "    return inner.search(qv, top_k=top_chunks), [d.doc_id for d in picks]",
                "",
                "hits, picks = route_and_retrieve('How are SQUIDs used to detect tiny magnetic fields?')",
                "print(f'Routed to: {picks}')",
                "for h in hits:",
                "    print(f'  {h.score:.3f}  {h.text[:160]}')",
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
                "    hits, _ = route_and_retrieve(question)",
                "    contexts = [h.text for h in hits]",
                "    answer = client.chat(",
                "        'Use these passages.\\n' + '\\n\\n'.join(contexts) + f'\\nQ: {question}\\nA:'",
                "    )",
                "    return answer, contexts",
                "",
                "ans, _ = answer_question('What is the Meissner effect?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — routing quality",
                "",
                "For a battery of queries, see which documents got routed. Sanity-check by reading the document titles.",
            ],
            code=[
                "for q in [",
                "    'How are SQUIDs used for magnetometry?',",
                "    'What is BCS theory?',",
                "    'What is YBCO?',",
                "    'How does flux pinning work?',",
                "]:",
                "    _, picks = route_and_retrieve(q)",
                "    print(f'  {q} -> {picks}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does a summary look like?",
                "",
                "Read a few. Summary quality is the technique. If summaries are vague, routing fails.",
            ],
            code=[
                "import random",
                "for d, s in random.sample(summaries, 3):",
                "    print(f'  {d.doc_id}')",
                "    print(f'    {s}')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — fall-through threshold",
                "",
                "If the router's top score is below a threshold, the right behaviour is to fall back to flat retrieval. Measure the threshold's effect on a query that genuinely doesn't fit any document.",
            ],
            code=[
                "qv = client.embed(['What is the boiling point of water?'])[0]",
                "doc_top = router.search(qv, top_k=1)[0]",
                "print(f'Top routing score for an out-of-corpus query: {doc_top.score:.3f}')",
                "print('In production, treat anything below 0.4 as no-match.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — recall@5 routed vs flat",
                "",
                "Compare routed retrieval against flat retrieval across the eval set.",
            ],
            code=[
                "from cookbook.corpora import load_eval_questions",
                "from cookbook.chunkers import fixed_window",
                "",
                "qs = [q for q in load_eval_questions() if q['corpus'] == 'wikipedia-superconductors'][:8]",
                "flat_chunks = fixed_window(docs, target_tokens=300, overlap_tokens=30)",
                "flat_v = client.embed([c.text for c in flat_chunks])",
                "flat_store = QdrantBackend('flat', dim=len(flat_v[0]))",
                "flat_store.add([c.text for c in flat_chunks], flat_v, ids=[c.chunk_id for c in flat_chunks])",
                "",
                "def recall(fn):",
                "    hits = 0",
                "    for q in qs:",
                "        out = fn(q['question'])",
                "        gold = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "        if any(any(w[:6] in t.lower() for w in gold) for t in out):",
                "            hits += 1",
                "    return hits / max(1, len(qs))",
                "",
                "def routed(q):",
                "    hits, _ = route_and_retrieve(q)",
                "    return [h.text for h in hits]",
                "def flat(q):",
                "    qv = client.embed([q])[0]",
                "    return [h.text for h in flat_store.search(qv, top_k=5)]",
                "",
                "print(f'flat   recall@5 = {recall(flat):.2f}')",
                "print(f'routed recall@5 = {recall(routed):.2f}')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a focused query.",
    ],
    run_code=[
        "ans, ctx = answer_question('What is BCS theory and which observations does it explain?')",
        "print('=== Doc-summary routed answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla baseline (flat retrieval) vs document-summary routing.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What is BCS theory and which observations does it explain?'",
        "base = vanilla_pipeline(q, corpus='wikipedia-superconductors', top_k=5)",
        "ours_a, ours_c = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla flat', 'preview': base.answer[:160]},",
        "    {'pipeline': 'routed', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Five knobs ranked by impact:",
        "",
        "1. **Summary length.** Three sentences is the cookbook default. Longer summaries route better — they mention more entities — at the cost of an inflated router index. Five sentences is a fine ceiling.",
        "2. **`top_docs` at the router.** Default 2. More documents catch cross-document queries but pay for more chunking and embedding work at query time.",
        "3. **No-match threshold.** The score below which the router falls back to flat retrieval. Default 0.4; tune on a labelled out-of-corpus slice. Too high means false negatives; too low means false positives.",
        "4. **Summary prompt.** Ask the model to mention the article's key entities by name. Vague summaries route badly.",
        "5. **Summariser model.** Cheaper than the answerer model is fine. The summaries are written once and reused for every query; spending budget once is fine.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Bad summaries.** A summary that fails to mention key entities loses them at routing. Re-summarise with a tighter prompt if you see specific entity-named queries miss the right document.",
        "- **Single-document corpora.** No documents to route to. Pattern is wasted.",
        "- **Cross-document questions.** \"Compare YBCO and Meissner effect\" wants both documents. Increase `top_docs` to handle this; combined with sub-question decomposition (Recipe 16) for harder cases.",
        "",
        "Compose with semantic routing (Recipe 17) for coarse-domain selection upstream and with reranking (Recipe 22) for the chunk-level retrieval downstream.",
    ],
)


# =============================================================================
# Matryoshka Coarse-to-Fine
# =============================================================================
MATRYOSHKA = Recipe(
    path="recipes/02-chunking-and-indexing/matryoshka-coarse-to-fine.ipynb",
    title="Matryoshka Coarse-to-Fine — Half the Dimensions, Same Recall",
    category="chunking-and-indexing",
    corpus_filter="arxiv-mamba",

    theory_problem=[
        "Large embeddings are expensive at scale. A 3072-dim Voyage 3 vector is 12 KB. At a billion chunks, your vector index is 12 TB of RAM. Even at a tenth that size you have problems: ANN-index memory dominates server bills, replication is painful, and re-embedding when you upgrade models takes days.",
        "Matryoshka Representation Learning trains embeddings so the leading dimensions are themselves a usable embedding — slice off the first 128 or 512 dimensions and you have a working embedding at one-tenth the storage cost. The cheap retrieval pattern follows: shortlist with the truncated low-dim vectors over the whole corpus, then re-rank the shortlist with the full-dim vectors. Latency drops because the bulk of the work happens at low-dim; quality stays because the final ranking uses full-dim. The recall loss is usually under a percentage point; the cost saving is an order of magnitude.",
    ],
    theory_origin=[
        "Matryoshka Representation Learning was published at NeurIPS 2022 by Aditya Kusupati and colleagues at UW. The training trick: regularise the model to ensure the loss is meaningful when computed on the first `d_i` dimensions for any prefix.",
        "OpenAI's text-embedding-3 family (Jan 2024) was the first widely-used embedding to ship Matryoshka by default. Voyage 3 and Cohere Embed 4 followed. By 2026 most production-grade embedders support truncation; the cookbook uses Nebius's Qwen3-Embedding-8B which exposes the full vector — we truncate manually.",
    ],
    theory_landscape=[
        "Matryoshka enables several deployment patterns:",
        "",
        "- **Coarse-to-fine retrieval (this recipe).** Shortlist with truncated vectors, rerank with full. The canonical deployment pattern.",
        "- **Multi-resolution indexes.** Store the index at multiple dimensions; pick at query time based on latency budget or query difficulty.",
        "- **Cheap quantisation.** Truncate then quantise; both are lossy but compound nicely. Production systems often run both.",
        "- **Cross-modal compatibility.** Truncated text embeddings sometimes align better with image embeddings; useful in multimodal retrieval setups.",
        "",
        "Compose with ColBERT (Recipe 19) — Matryoshka cuts per-token storage; ColBERT keeps per-token vectors. The combination is the cheapest known way to deploy multi-vector retrieval at scale.",
    ],
    theory_when_to_use=[
        "Use Matryoshka coarse-to-fine when storage cost dominates your retrieval bill. Anything over 10M chunks usually qualifies.",
        "Skip it when your embedder does not support truncation. Slicing dimensions on a non-Matryoshka embedder silently destroys quality.",
        "Skip it when you only have a few thousand chunks. Full-dim search is already fast; the two-stage complexity is not worth it.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**Truncation works because of training.** Standard embeddings put their best signal anywhere in the vector. Matryoshka-trained embeddings put the best signal in the leading dimensions by design — the regulariser during training forces the loss to be meaningful at every prefix length.",
        "",
        "**Two-stage retrieval is the magic.** Stage 1 is cheap and noisy (low-dim). Stage 2 is expensive and accurate (full-dim) but runs only on the shortlist. The shortlist is small enough that stage 2's per-vector cost doesn't dominate — the whole pipeline runs roughly at low-dim speed with full-dim quality.",
        "",
        "**Recall@1 may drop slightly; recall@5 usually doesn't.** Stage 1 sometimes ranks the right chunk at position 30; if your shortlist is 50, it survives stage 1 and gets promoted by stage 2's full-dim scoring. Tune shortlist size to your acceptable recall.",
    ],

    architecture_mermaid="""
flowchart LR
  D[Documents] --> E[Embed full-dim]
  E --> FT[Full vectors<br/>~1024 dim]
  E --> TR[Truncate to<br/>128 dim]
  FT --> FI[(Full-dim<br/>lookup table)]
  TR --> CI[(Coarse index<br/>cheap ANN)]
  Q[Query] --> QE[Embed]
  QE --> SH[Shortlist<br/>top-50 via coarse]
  CI --> SH
  SH --> RR[Re-rank<br/>with full-dim]
  FI --> RR
  RR --> A[Top-5]
""",

    references=[
        Reference(
            title="Matryoshka Representation Learning (Kusupati et al., NeurIPS 2022)",
            url="https://arxiv.org/abs/2205.13147",
            kind="paper",
            note="The original paper.",
        ),
        Reference(
            title="OpenAI text-embedding-3 announcement",
            url="https://openai.com/index/new-embedding-models-and-api-updates/",
            kind="blog",
            note="Where Matryoshka went mainstream.",
        ),
        Reference(
            title="Nomic Embed v2 model card",
            url="https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe",
            kind="repo",
            note="Open-weight Matryoshka embedder.",
        ),
        Reference(
            title="Pinecone two-stage retrieval guide",
            url="https://www.pinecone.io/learn/series/rag/rerankers/",
            kind="blog",
            note="General two-stage retrieval reference.",
        ),
        Reference(
            title="Voyage 3 Large embedding documentation",
            url="https://docs.voyageai.com/docs/embeddings",
            kind="docs",
            note="Hosted Matryoshka-supporting embedder.",
        ),
        Reference(
            title="ColBERT (Recipe 19)",
            url="https://arxiv.org/abs/2004.12832",
            kind="paper",
            note="Compose with Matryoshka for cheap multi-vector at scale.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load + chunk the Mamba paper",
                "",
                "We use the arXiv Mamba survey. Long enough to make the two-stage savings non-trivial.",
            ],
            code=[
                "from cookbook.corpora import load_arxiv_mamba",
                "from cookbook.chunkers import sentence_window",
                "",
                "docs = list(load_arxiv_mamba())",
                "chunks = sentence_window(docs, sentences_per_chunk=4, overlap=1)",
                "print(f'Chunks: {len(chunks)}')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 2 — Compute full-dimensional embeddings",
                "",
                "Standard. The full vectors are normalised; cosine = dot product.",
            ],
            code=[
                "import numpy as np",
                "full = np.asarray(client.embed([c.text for c in chunks]), dtype=np.float32)",
                "full /= np.linalg.norm(full, axis=1, keepdims=True).clip(min=1e-9)",
                "print(f'Full vectors: {full.shape}')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 3 — Truncate to a coarse dimension",
                "",
                "Slice the first `COARSE_DIM` dimensions and re-normalise. With a Matryoshka-trained embedder these truncated vectors are still meaningful; with non-Matryoshka embedders they silently degrade.",
            ],
            code=[
                "COARSE_DIM = 128",
                "coarse = full[:, :COARSE_DIM].copy()",
                "coarse /= np.linalg.norm(coarse, axis=1, keepdims=True).clip(min=1e-9)",
                "print(f'Coarse vectors: {coarse.shape}')",
            ],
        ),
        CodeStep(
            tag="index",
            lead_md=[
                "### Step 4 — Index the coarse vectors",
                "",
                "The coarse store is one-eighth the size of the full store. Memory and disk savings scale with the dim ratio.",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "",
                "coarse_store = QdrantBackend('mrl-coarse', dim=COARSE_DIM)",
                "coarse_store.add(",
                "    [c.text for c in chunks],",
                "    coarse.tolist(),",
                "    ids=[c.chunk_id for c in chunks],",
                ")",
                "id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}",
                "print('Coarse index built.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 5 — Two-stage search function",
                "",
                "Shortlist with coarse; rerank with full. Full-dim re-ranking is a single matrix-vector product over the shortlist.",
            ],
            code=[
                "def two_stage(question: str, shortlist: int = 50, final: int = 5):",
                "    q_full = np.asarray(client.embed([question])[0], dtype=np.float32)",
                "    q_full /= np.linalg.norm(q_full) + 1e-9",
                "    q_coarse = q_full[:COARSE_DIM]",
                "    q_coarse /= np.linalg.norm(q_coarse) + 1e-9",
                "    coarse_hits = coarse_store.search(q_coarse.tolist(), top_k=shortlist)",
                "    scored = []",
                "    for h in coarse_hits:",
                "        idx = id_to_idx[h.doc_id]",
                "        score = float(full[idx] @ q_full)",
                "        scored.append((score, h.text, h.doc_id))",
                "    scored.sort(key=lambda x: x[0], reverse=True)",
                "    return scored[:final]",
                "",
                "for s, t, _ in two_stage('What is selective scan?', shortlist=30, final=5):",
                "    print(f'  full-dim={s:.3f}  {t[:160]}')",
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
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    top = two_stage(question, shortlist=50, final=k)",
                "    contexts = [t for _, t, _ in top]",
                "    answer = client.chat(",
                "        'Use these passages.\\n' + '\\n\\n'.join(contexts) + f'\\nQ: {question}\\nA:'",
                "    )",
                "    return answer, contexts",
                "",
                "ans, _ = answer_question('Explain selective scan.')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — coarse-only top-3 vs two-stage top-3",
                "",
                "Sometimes the coarse top-3 differs from the two-stage top-3. The rerank step picks the truly best chunks from the shortlist.",
            ],
            code=[
                "import numpy as np",
                "q = 'What is selective scan?'",
                "q_full = np.asarray(client.embed([q])[0], dtype=np.float32)",
                "q_full /= np.linalg.norm(q_full) + 1e-9",
                "q_coarse = q_full[:COARSE_DIM] / np.linalg.norm(q_full[:COARSE_DIM])",
                "",
                "print('--- coarse top-3 ---')",
                "for h in coarse_store.search(q_coarse.tolist(), top_k=3):",
                "    print(f'  {h.score:.3f}  {h.text[:140]}')",
                "print()",
                "print('--- two-stage top-3 ---')",
                "for s, t, _ in two_stage(q, shortlist=50, final=3):",
                "    print(f'  {s:.3f}  {t[:140]}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — recall vs shortlist size",
                "",
                "Sweep shortlist size from 10 to 100 and see how recall changes. Useful for choosing a shortlist that balances latency and quality.",
            ],
            code=[
                "from cookbook.corpora import load_eval_questions",
                "qs = [q for q in load_eval_questions() if q['corpus'] == 'arxiv-mamba'][:8]",
                "import pandas as pd",
                "rows = []",
                "for sl in (10, 25, 50, 100):",
                "    hits = 0",
                "    for q in qs:",
                "        top = two_stage(q['question'], shortlist=sl, final=5)",
                "        gold = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "        if any(any(w[:6] in t.lower() for w in gold) for _, t, _ in top):",
                "            hits += 1",
                "    rows.append({'shortlist': sl, 'recall@5': hits / max(1, len(qs))})",
                "pd.DataFrame(rows)",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — storage savings",
                "",
                "Concretely: how much less memory does the coarse index use?",
            ],
            code=[
                "full_bytes = full.shape[0] * full.shape[1] * 4",
                "coarse_bytes = coarse.shape[0] * coarse.shape[1] * 4",
                "print(f'Full   index: {full_bytes:>10,} bytes')",
                "print(f'Coarse index: {coarse_bytes:>10,} bytes  ({coarse_bytes/full_bytes:.1%} of full)')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what happens at very low dim?",
                "",
                "Truncating to 32 or 16 dims usually breaks retrieval even on Matryoshka-trained embedders. Sweep dim and watch recall fall.",
            ],
            code=[
                "import pandas as pd",
                "rows = []",
                "for d in (32, 64, 128, 256, 512):",
                "    coarse_d = full[:, :d]",
                "    coarse_d = coarse_d / np.linalg.norm(coarse_d, axis=1, keepdims=True).clip(min=1e-9)",
                "    q_full = client.embed(['What is selective scan?'])[0]",
                "    q_d = np.asarray(q_full[:d])",
                "    q_d /= np.linalg.norm(q_d) + 1e-9",
                "    sims = coarse_d @ q_d",
                "    top_idx = np.argsort(sims)[::-1][:5]",
                "    top_chunks = [chunks[i].text[:60] for i in top_idx]",
                "    rows.append({'dim': d, 'top1_preview': top_chunks[0]})",
                "pd.DataFrame(rows)",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a Mamba question.",
    ],
    run_code=[
        "ans, ctxs = answer_question('How does the parallel scan implementation matter for Mamba on modern GPUs?')",
        "print('=== Matryoshka two-stage answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla baseline (full-dim search) vs Matryoshka two-stage. The quality should be comparable; the speed/memory advantage shows up at scale.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'How does the parallel scan implementation matter for Mamba on modern GPUs?'",
        "base = vanilla_pipeline(q, corpus='arxiv-mamba', top_k=5)",
        "ours_a, ours_c = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla (full-dim)', 'preview': base.answer[:140]},",
        "    {'pipeline': 'matryoshka (two-stage)', 'preview': ours_a[:140]},",
        "])",
    ],

    tuning_md=[
        "Five knobs in priority order:",
        "",
        "1. **COARSE_DIM.** Higher means better stage-1 recall, less memory savings. 128 is a strong default for OpenAI text-embedding-3 and Voyage 3; 256 is conservative.",
        "2. **Shortlist size.** Higher catches more recall, slows the re-rank. Typical settings: 30–100. Sweep on your corpus to find where recall flattens.",
        "3. **Embedder.** Matryoshka requires a trained-for-it embedder. Without one, this technique silently degrades — recall drops by tens of points and you have no signal until you measure.",
        "4. **Quantisation.** Compose with int8 quantisation for further savings; both are lossy in different ways and compound nicely. Production deployments often run truncated + quantised.",
        "5. **Re-rank with cross-encoder.** Replace the full-dim re-rank with a cross-encoder reranker (Recipe 22). The two-stage shape stays; the second stage uses a different scorer for higher quality.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Non-Matryoshka embedder.** Truncation silently destroys quality. Check your model card before deploying.",
        "- **Shortlist too small.** Stage 1 may rank the right chunk at position 60 on hard queries; if your shortlist is 50, you lose it. Tune.",
        "- **Re-rank cost.** Full-dim rerank over 50 vectors is cheap; over 1000 is not. Stay within reasonable shortlist sizes.",
        "",
        "Compose with quantisation (int8 or product quantisation) for further savings; compose with ColBERT (Recipe 19) for cheap multi-vector at scale.",
    ],
)


RECIPES = [CONTEXTUAL_RETRIEVAL, LATE_CHUNKING, SEMANTIC_SPLIT, PROPOSITION, PARENT_CHILD, RAPTOR, DOC_SUMMARY, MATRYOSHKA]
