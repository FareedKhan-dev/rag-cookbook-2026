"""Foundations — the four notebooks every reader sees first.

These also set the bar for the other 36. Each recipe is fully self-contained
and uses only the public `cookbook.*` API plus the running cell state. No
notebook-to-notebook imports; everything important is re-exported through
`cookbook.baselines` so comparison cells can reach it as a Python import.
"""
from __future__ import annotations

from authoring import CodeStep, Recipe, Reference


# =============================================================================
# 1. Vanilla Pipeline — the baseline every later recipe imports
# =============================================================================
VANILLA = Recipe(
    path="recipes/01-foundations/vanilla-pipeline.ipynb",
    title="The Vanilla Pipeline — Our Baseline",
    category="foundations",
    corpus_filter="rust-book",
    skip_sections={"comparison"},  # *this* IS the comparison baseline

    theory_problem=[
        "Most retrieval-augmented generation tutorials open with three sentences of motivation and then drop you into a 200-line framework call. By the time the model answers, you have no mental model for what each step did, why it exists, or where the failure modes live. This notebook is the opposite. We build the simplest end-to-end RAG pipeline that could possibly work, with one logical step per cell, and we treat the rest of the cookbook as a series of measured improvements over it.",
        "There is a deeper reason to start here. Every later recipe — Self-RAG, CRAG, GraphRAG, ColPali — claims to beat \"the baseline\". If we never write the baseline down precisely, we have no way to verify those claims. The vanilla pipeline you build here is what `cookbook.baselines.vanilla_pipeline` exposes to every other notebook for direct side-by-side comparison.",
    ],
    theory_origin=[
        "The shape — chunk, embed, store, top-k retrieve, stuff into prompt, generate — was nailed down by the original 2020 RAG paper from Lewis et al. and popularized by the LangChain and LlamaIndex tutorials in late 2022 and early 2023. None of the pieces are new. What changed in 2024 and 2025 is that we now know exactly *which* pieces leak the most quality and how to tune each one. Knowing the baseline cold is the prerequisite for any of that.",
    ],
    theory_landscape=[
        "Think of the cookbook as a tree rooted at this notebook. The three big branches you will encounter:",
        "",
        "1. **Better chunks** (Recipes 5–12) — replace fixed-window chunks with semantic boundaries, late chunking, propositions, parent-child, RAPTOR trees, doc-summary routing, or matryoshka coarse-to-fine.",
        "2. **Better retrieval** (Recipes 13–23) — rewrite the query, fan out paraphrases, switch to hybrid dense + BM25, add late interaction, then rerank.",
        "3. **Better orchestration** (Recipes 24–34) — let the model decide whether to retrieve at all, retry on bad scores, walk a knowledge graph, or carry memory across turns.",
        "",
        "Every branch competes against the same root. If your fancy technique cannot beat the vanilla pipeline on your data, your technique is overhead, not improvement.",
    ],
    theory_when_to_use=[
        "Use the vanilla pipeline as your reference implementation. In production it is also a perfectly serviceable system for narrow, well-scoped corpora — a single product manual, an FAQ, a customer support knowledge base of a few thousand documents. Most production RAG that \"just works\" is not far from what you build here, with one or two well-chosen extras: a reranker for quality, prompt caching for cost, and a query cache for latency.",
        "Avoid it when: your corpus mixes very different document types and a single retrieval strategy will not serve all of them (route first, recipes 11 and 17); when answers require reasoning across multiple chunks (recipes 10, 14, 16, 31–33); when documents are visual and OCR loses the layout (recipes 35–36); or when your users will ask questions whose answers are not in the corpus at all (recipes 24–26 add the machinery for that).",
    ],
    theory_intuition=[
        "Three intuitions are worth carrying with you everywhere in this cookbook:",
        "",
        "**Embeddings are lossy summaries.** A 1024-dim vector cannot preserve everything a 400-word chunk says. Cosine similarity finds chunks that are *topically near* the question, which is correlated with — but not the same as — chunks that *answer* the question. That gap is what every later technique is trying to close.",
        "",
        "**The prompt is the API.** Whatever you stuff into the prompt is what the model sees. If retrieval returns noisy chunks, the model will hallucinate confidently from them. The model is not a guardrail; the prompt is.",
        "",
        "**Latency lives in the LLM call.** Embedding, indexing, and vector search are all sub-50 ms operations on commodity hardware. The generation call is the second-and-a-half elephant in your latency budget. Every recipe that adds *more* LLM calls (reranking, reflection, multi-step) is buying quality with wall time.",
    ],

    architecture_mermaid="""
flowchart LR
  D[Documents] --> C[Fixed-window<br/>chunker]
  C --> E[Embedder<br/>Nebius]
  E --> S[(Qdrant<br/>in-memory)]
  Q[User Question] --> QE[Embed query]
  QE --> R[Top-k cosine search]
  S --> R
  R --> P[Stuff into prompt]
  P --> L[LLM]
  L --> A[Answer]
  style D fill:#e9efff
  style A fill:#e9ffe9
""",

    references=[
        Reference(
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)",
            url="https://arxiv.org/abs/2005.11401",
            kind="paper",
            note="The paper that named RAG",
        ),
        Reference(
            title="LlamaIndex: A High-Level Concepts overview",
            url="https://developers.llamaindex.ai/python/framework/getting_started/concepts",
            kind="docs",
            note="Canonical naming for chunks/nodes/retrievers",
        ),
        Reference(
            title="Qdrant Vector Search Documentation",
            url="https://qdrant.tech/documentation/",
            kind="docs",
            note="The vector store we use for the baseline",
        ),
        Reference(
            title="LiteLLM — One SDK for 100+ LLMs",
            url="https://docs.litellm.ai/",
            kind="docs",
            note="The thin client behind cookbook.providers",
        ),
        Reference(
            title="Anthropic Contextual Retrieval (2024)",
            url="https://www.anthropic.com/news/contextual-retrieval",
            kind="blog",
            note="Where to go next — Recipe 7 in this cookbook",
        ),
        Reference(
            title="Nebius AI Studio quickstart",
            url="https://studio.nebius.com/docs/",
            kind="docs",
            note="OpenAI-compatible endpoint we default to",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load the corpus",
                "",
                "We are going to retrieve over the markdown chapters of *The Rust Programming Language* (CC BY 4.0). Eighteen chapters, well-written prose, dense technical content with code blocks. A representative test for any RAG system aimed at technical documentation. The loader yields plain `Document` records with stable IDs and metadata.",
            ],
            code=[
                "from cookbook.corpora import load_rust_book",
                "",
                "docs = list(load_rust_book())",
                "print(f'Loaded {len(docs)} chapters.')",
                "print()",
                "print('First chapter preview:')",
                "first = docs[0]",
                "print(f'  doc_id   : {first.doc_id}')",
                "print(f'  source   : {first.source}')",
                "print(f'  metadata : {first.metadata}')",
                "print(f'  text     : {first.text[:240]}...')",
            ],
            expected_output_md=[
                "Notice the IDs are URLs we can cite, and the metadata carries chapter information we could later use as a retrieval filter (Recipe 20). The text is the raw markdown straight from the book repository — fences, headings, prose. We deliberately do not strip the markdown; the model handles it fine and the structure helps retrieval.",
            ],
        ),
        CodeStep(
            tag="chunk",
            lead_md=[
                "### Step 2 — Chunk the corpus",
                "",
                "We slice each chapter into overlapping windows of about 384 tokens with 64-token overlap. This is the un-tuned default — Recipe 4 sweeps the window size to show how much it matters; Recipe 5 swaps fixed-window for semantic-boundary cuts. Today, we want the simplest possible chunker so the rest of the pipeline is the only thing changing.",
            ],
            code=[
                "from cookbook.chunkers import fixed_window",
                "",
                "chunks = fixed_window(docs, target_tokens=384, overlap_tokens=64)",
                "print(f'Built {len(chunks)} chunks from {len(docs)} chapters.')",
                "print(f'Avg chunk length (tokens, approx): {sum(len(c.text.split()) for c in chunks) // len(chunks)}')",
                "print()",
                "print('First chunk:')",
                "print(f'  chunk_id : {chunks[0].chunk_id}')",
                "print(f'  doc_id   : {chunks[0].doc_id}')",
                "print(f'  text     : {chunks[0].text[:240]!r}')",
            ],
            expected_output_md=[
                "Each `Chunk` carries its parent `doc_id`, so even after retrieval the result remembers which chapter it came from. That parent link is what Recipes 9 (parent-child) and 11 (document-summary routing) build on.",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 3 — Embed the chunks",
                "",
                "Now we hand the chunk texts to Nebius's embedding model and get one vector per chunk. The cookbook's `client.embed()` call is cached on disk by SHA256 of `(provider, model, text)`, so the second time we run this notebook the embedding pass is free. First-time cost on the Rust corpus is small — under a few cents at current Nebius pricing.",
            ],
            code=[
                "vectors = client.embed([c.text for c in chunks])",
                "print(f'Got {len(vectors)} vectors, dim={len(vectors[0])}')",
                "",
                "import numpy as np",
                "v = np.asarray(vectors[0])",
                "print(f'First vector norm: {np.linalg.norm(v):.3f}')",
                "print(f'First 8 dims:     {[round(x, 3) for x in v[:8]]}')",
            ],
            expected_output_md=[
                "Embedding norm being close to 1 tells you the model returns L2-normalized vectors, so cosine similarity reduces to a dot product. Norms far from 1 sometimes hint at an embedding-API misconfiguration (wrong model, dimension mismatch).",
            ],
        ),
        CodeStep(
            tag="index",
            lead_md=[
                "### Step 4 — Index in Qdrant",
                "",
                "Qdrant stores the vectors and lets us search them. We use the in-memory mode here so the notebook is reproducible without a running server. In production you would point at a Qdrant cluster URL (`QDRANT_URL`) and add a payload schema; the search code stays identical.",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "",
                "store = QdrantBackend('vanilla-rust', dim=len(vectors[0]))",
                "store.add(",
                "    texts=[c.text for c in chunks],",
                "    vectors=vectors,",
                "    metadatas=[c.metadata for c in chunks],",
                "    ids=[c.chunk_id for c in chunks],",
                ")",
                "print(f'Indexed {len(chunks)} chunks into Qdrant collection \"vanilla-rust\".')",
            ],
            expected_output_md=[
                "Indexing is essentially free at this scale. The interesting case is when you have millions of vectors — there Qdrant's HNSW parameters and quantization start mattering. Recipe 12 (Matryoshka) is one path to making large-scale search cheap.",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 5 — Define a retrieval function",
                "",
                "Embed the question, ask Qdrant for the top-k closest chunks. That is it. Notice we return both the texts and their similarity scores — we will look at the scores in the inspection section.",
            ],
            code=[
                "from cookbook.stores import Hit",
                "",
                "def retrieve(question: str, k: int = 5) -> list[Hit]:",
                "    q_vec = client.embed([question])[0]",
                "    return store.search(q_vec, top_k=k)",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 6 — Stuff retrieved chunks into a prompt",
                "",
                "The simplest possible prompt: concatenate the chunks, paste the question, ask for an answer. Every later recipe varies *what* gets stuffed and *how* it gets framed, but the bones are this. We name the function `answer_question` because the rest of the cookbook expects that name when comparing to a baseline.",
            ],
            code=[
                "PROMPT = (",
                "    'Answer the question using only the passages below. '",
                "    'If the passages do not contain the answer, say so plainly.\\n\\n'",
                "    'Passages:\\n{context}\\n\\nQuestion: {question}\\nAnswer:'",
                ")",
                "",
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    hits = retrieve(question, k=k)",
                "    contexts = [h.text for h in hits]",
                "    answer = client.chat(PROMPT.format(",
                "        context='\\n\\n'.join(contexts),",
                "        question=question,",
                "    ))",
                "    return answer, contexts",
            ],
            expected_output_md=[
                "`answer_question` returns `(answer, contexts)` because every later recipe is going to compare both — not just the answer string, but *what was retrieved* that led to it. RAGAS context-precision and context-recall live in those `contexts`.",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does a single retrieval look like?",
                "",
                "Before we fire a real question, we should know what Qdrant returns. We will use a deliberately specific query so the top hits are obviously on-topic, then look at scores, IDs, and snippets.",
            ],
            code=[
                "question = 'How does the borrow checker enforce that mutable references are exclusive?'",
                "hits = retrieve(question, k=5)",
                "for i, h in enumerate(hits, 1):",
                "    print(f'{i}. score={h.score:.3f}  id={h.doc_id}')",
                "    print(f'   {h.text[:180]}...')",
                "    print()",
            ],
            expected_output_md=[
                "A few things to read for:",
                "",
                "- The top score should be visibly higher than the bottom score. If it is not, retrieval is shaky — usually a sign of either a too-small chunk or a query that lives in the wrong embedding subspace (Recipe 13 fixes the latter).",
                "- The top hit's text should obviously contain the word \"borrow\" or its synonyms. Recall failures often have a top hit that is *adjacent* but not on-target.",
                "- The IDs let you trace exactly which chapter answered. In production that mapping is what powers citations.",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — how good is the score signal?",
                "",
                "A scatter of the top-20 hits' scores tells you how *peaked* retrieval is. A steep curve means the top result is much better than the rest; a flat curve means retrieval is uncertain and the model will be working from noisy context.",
            ],
            code=[
                "import matplotlib.pyplot as plt",
                "",
                "scores = [h.score for h in retrieve(question, k=20)]",
                "fig, ax = plt.subplots(figsize=(6, 2.8))",
                "ax.plot(range(1, len(scores) + 1), scores, marker='o')",
                "ax.set_xlabel('Rank')",
                "ax.set_ylabel('Cosine similarity')",
                "ax.set_title('Top-20 retrieval scores')",
                "ax.grid(alpha=0.3)",
                "plt.tight_layout()",
                "plt.show()",
            ],
            expected_output_md=[
                "On the Rust book, this curve usually drops sharply between ranks 1-3 and then plateaus around 0.5-0.6. The plateau is the long tail of *somewhat related* chunks. The reranker in Recipe 22 is the tool of choice when the top-3 is right but rank-5 is junk — it lets you keep `k=5` for context width without polluting the prompt.",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does the assembled prompt actually look like?",
                "",
                "The prompt is what the model sees. Print it.",
            ],
            code=[
                "hits = retrieve(question, k=3)",
                "rendered = PROMPT.format(",
                "    context='\\n\\n'.join(h.text for h in hits),",
                "    question=question,",
                ")",
                "print(f'Prompt length: {len(rendered)} chars')",
                "print('--- first 1000 chars ---')",
                "print(rendered[:1000])",
                "print('...')",
            ],
            expected_output_md=[
                "When something goes wrong with a RAG system, this is usually the first thing to look at: is the prompt well-formed, is the right context present, is the question phrased clearly? Phoenix tracing (Recipe 39) gives you this view for every call automatically.",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — token budget reality check",
                "",
                "We are stuffing five chunks (~1900 tokens) plus a system prompt into every call. With `k=20` the prompt balloons. Knowing your token footprint matters for cost and for the next set of recipes — late chunking, contextual headers, and reranking all touch this.",
            ],
            code=[
                "import tiktoken",
                "enc = tiktoken.get_encoding('cl100k_base')",
                "lengths = [len(enc.encode(h.text)) for h in retrieve(question, k=20)]",
                "import statistics",
                "print(f'tokens/chunk:  min={min(lengths)}  median={statistics.median(lengths):.0f}  max={max(lengths)}')",
                "print(f'k=5 prompt budget: ~{sum(lengths[:5])} tokens of context')",
                "print(f'k=20 prompt budget: ~{sum(lengths)} tokens of context')",
            ],
            expected_output_md=[
                "If your generator is a $50-per-million-input-tokens model, every k+15 extra retrieval chunks costs you about a tenth of a cent per query, all day, forever. Two-stage retrieve-then-rerank (Recipe 22) lets you keep `k=5` for the LLM while still shortlisting 50 candidates upstream.",
            ],
        ),
    ],

    run_md=[
        "Now the full pipeline on a realistic question. We will print the answer and the chunks that informed it.",
    ],
    run_code=[
        "from textwrap import fill",
        "",
        "question = 'When should I prefer Arc over Rc, and what guarantees do I lose if I switch?'",
        "answer, contexts = answer_question(question, k=5)",
        "print('=== Answer ===')",
        "print(fill(answer, width=100))",
        "print()",
        "print('=== Contexts retrieved ===')",
        "for i, c in enumerate(contexts, 1):",
        "    print(f'[{i}] {c[:200]}...')",
        "    print()",
    ],

    tuning_md=[
        "Four levers in the order you should reach for them:",
        "",
        "1. **Chunk size and overlap.** Defaults of 384 / 64 are reasonable for prose; Recipe 4 sweeps them and shows the optimum is often 256 or 512 depending on corpus density. Single biggest cheap win.",
        "2. **`k` (number of retrieved chunks).** Increasing `k` improves recall but adds tokens to the prompt linearly. Most production systems land between 3 and 8.",
        "3. **Embedding model.** Switching `client.embed_model` to a stronger family (Voyage 3, Cohere Embed 4, BGE-M3 finetuned on your domain) is the next biggest lift. Recipe 3 compares four families.",
        "4. **Prompt wording.** \"Answer using only these passages\" reduces hallucination at the cost of asking the model to refuse more often. Tighten it for high-precision use cases; relax for assistant-style chat.",
        "",
        "Things *not* to tune at this stage: similarity metric (cosine is fine), distance threshold (set `k` instead), reranker (that is Recipe 22 — do not add a reranker until you have measured that the top-`k` is the failure mode).",
    ],

    discussion_md=[
        "If you only ship this notebook, you have a perfectly usable RAG system for a narrow corpus. Most users would not notice the difference between this and a much fancier pipeline on questions whose answers are in a single chapter.",
        "",
        "Where this falls down — and where every later recipe earns its complexity:",
        "",
        "- **Multi-hop questions** (\"How do Arc and Mutex compose, and what does that look like in code?\") routinely retrieve from one chapter and miss the other. Recipes 10 (RAPTOR) and 14 (multi-query fusion) address this.",
        "- **Questions outside the corpus** trigger confident hallucination. Recipes 24 (Self-RAG) and 25 (CRAG) wire in detection and fallback.",
        "- **Visual content** (tables, figures, equations) is invisible to text-only retrieval. Recipes 35–36 handle this.",
        "- **Personalization** across user sessions has no machinery here. Recipe 34 (Mem0) adds it.",
        "",
        "Treat every other notebook as a measured patch against this baseline. The first cell of each will import `cookbook.baselines.vanilla_pipeline` and run it side by side with the new technique on the same question.",
    ],
)


# =============================================================================
# 2. Cookbook Tour — wire up providers, corpora, tracing, eval data
# =============================================================================
COOKBOOK_TOUR = Recipe(
    path="recipes/01-foundations/cookbook-tour.ipynb",
    title="A Tour of the Cookbook's Primitives",
    category="foundations",
    corpus_filter="rust-book",
    skip_sections={"comparison"},

    theory_problem=[
        "Every later recipe leans on four primitives: a provider client that hides the chat / embed / rerank vendor SDKs, a set of corpus loaders that yield uniform `Document` records, a thin vector-store wrapper with a common `.add/.search` API, and a tracing toggle. If you skip this notebook, the first real recipe will throw four unfamiliar imports at you in cell three and you will be reading API docs instead of learning RAG.",
        "We are not building RAG yet. We are checking that every primitive works in the environment you actually run code in, before we try to combine them. Five minutes here saves an hour of unrelated debugging later.",
    ],
    theory_origin=[
        "The four-primitive split mirrors what production RAG infrastructure has converged on since 2024: an LLM gateway (LiteLLM, Portkey, OpenRouter), a typed document model (LlamaIndex `Document`, LangChain `Document`), a backend-agnostic vector interface, and OpenTelemetry-grade tracing. The naming in this cookbook is intentionally distinct from LangChain and LlamaIndex so the conceptual gap stays visible — you should be able to read the code and tell exactly what each layer does.",
        "We do not invent any of the primitives we use. `cookbook.providers` is a thin wrapper around LiteLLM; `cookbook.stores` wraps Qdrant, LanceDB, and Chroma; `cookbook.tracing` wraps Arize Phoenix or LangSmith. The wrappers exist to keep the recipes readable and provider-agnostic, not to hide capability. When you outgrow a wrapper, drop down to the underlying library — every wrapper exposes the raw client as a property.",
    ],
    theory_landscape=[
        "Three boundaries you should understand before going further:",
        "",
        "- **Provider ↔ recipe.** The `LLMClient` is provider-agnostic by design. Set `PROVIDER=nebius` in `.env` and your notebook runs on Nebius. Set `PROVIDER=openai` and it runs on OpenAI with the same code. Recipes never import `litellm`, `openai`, or `anthropic` directly. That boundary is the entire reason this cookbook does not break the day a vendor renames a model.",
        "- **Corpus ↔ chunker.** Loaders yield `Document` objects with stable IDs and text content. Chunkers consume `Document` lists. If you want a custom corpus, write a loader; you do not have to touch chunkers, embeddings, or stores.",
        "- **Store ↔ retriever.** Every vector store wrapped here (Qdrant, LanceDB, Chroma) exposes `add()` and `search()` with identical signatures. Switching backends in a recipe is one line.",
    ],
    theory_when_to_use=[
        "Read this notebook once when you set up the repo, and again if you are debugging an environment issue. Skip it if you already have everything green — the actual RAG starts in the vanilla pipeline (Recipe 2).",
        "If any cell here fails, fix it before going further. A broken provider means every recipe's chat/embed call fails; a missing corpus means every retrieval gets zero results; a missing tracing dependency means you can read but not debug.",
    ],
    theory_intuition=[
        "Think of the cookbook as four layers, bottom-up:",
        "",
        "1. **Data layer** — `cookbook.corpora` yields uniform documents.",
        "2. **Vector layer** — `cookbook.chunkers` cuts them up, `cookbook.providers.embed()` vectorises, `cookbook.stores` indexes.",
        "3. **Retrieval layer** — `cookbook.retrievers` and `cookbook.rerankers` find and reorder candidates.",
        "4. **Generation layer** — `cookbook.providers.chat()` answers from retrieved context, with `cookbook.tracing` watching.",
        "",
        "Every later recipe is a vertical slice through these four layers, swapping out one or two pieces per slice. If you keep the mental model clean, the cookbook reads like variations on a single theme.",
    ],

    architecture_mermaid="""
flowchart TB
  subgraph Data
    C1[load_arxiv_mamba]
    C2[load_wikipedia_superconductors]
    C3[load_sec_10k]
    C4[load_rust_book]
  end
  subgraph Vectors
    K[chunkers] --> EM[providers.embed]
    EM --> ST[stores: Qdrant / LanceDB / Chroma]
  end
  subgraph Retrieval
    RT[retrievers: hybrid / MMR / RRF]
    RR[rerankers: cross-encoder / listwise]
  end
  subgraph Generation
    CH[providers.chat]
  end
  subgraph Observability
    TR[tracing.init_tracing]
    EV[eval suite]
  end
  Data --> K
  ST --> RT --> RR --> CH
  TR -.-> EM
  TR -.-> CH
  EV -.-> CH
""",

    references=[
        Reference(
            title="LiteLLM — One SDK for 100+ LLMs",
            url="https://docs.litellm.ai/",
            kind="docs",
            note="Hides vendor differences across chat and embedding APIs.",
        ),
        Reference(
            title="Nebius AI Studio — Inference API reference",
            url="https://studio.nebius.com/docs/inference",
            kind="docs",
            note="OpenAI-compatible endpoint we default to.",
        ),
        Reference(
            title="Qdrant Python client documentation",
            url="https://python-client.qdrant.tech/",
            kind="docs",
            note="The default vector store in this cookbook.",
        ),
        Reference(
            title="Arize Phoenix — open-source LLM observability",
            url="https://docs.arize.com/phoenix",
            kind="docs",
            note="OpenTelemetry-native tracing used by `cookbook.tracing`.",
        ),
        Reference(
            title="OpenTelemetry GenAI semantic conventions",
            url="https://opentelemetry.io/docs/specs/semconv/gen-ai/",
            kind="docs",
            note="Why every LLM call ends up with the same span shape.",
        ),
        Reference(
            title="RAGAS — reference-free evaluation for RAG systems",
            url="https://docs.ragas.io/",
            kind="docs",
            note="Powers the evaluation slice every recipe ends with.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Confirm the provider client wakes up",
                "",
                "We will instantiate `LLMClient` with no arguments. It reads `PROVIDER` from the environment (defaults to `nebius`), picks the corresponding chat and embedding models, and is now ready for `client.chat()` and `client.embed()` calls. We will print the resolved settings so you can sanity-check them.",
            ],
            code=[
                "from cookbook.providers import LLMClient, list_providers",
                "",
                "print('All supported providers:')",
                "for name in list_providers():",
                "    print(f'  - {name}')",
                "print()",
                "print(f'Current PROVIDER env var: {os.environ.get(\"PROVIDER\", \"(unset)\")}')",
                "print(f'Resolved client:')",
                "print(f'  provider   : {client.provider}')",
                "print(f'  chat model : {client.chat_model}')",
                "print(f'  embed model: {client.embed_model}')",
            ],
            expected_output_md=[
                "Switching providers later is a one-line change: `client = LLMClient(provider='openai')` (or set `PROVIDER=openai` in `.env`). Every line of recipe code from there on stays identical.",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 2 — Round-trip a chat and an embedding",
                "",
                "The single best check that everything is wired up. One short chat call returns a string; one embed call returns a vector. If either throws, fix it before going further — every later recipe will hit one of these in the first three cells.",
            ],
            code=[
                "import numpy as np",
                "",
                "answer = client.chat('In one short sentence, what does RAG stand for and why is it useful?')",
                "print('Chat round-trip:')",
                "print(f'  {answer}')",
                "print()",
                "",
                "vec = client.embed(['Retrieval-Augmented Generation grounds LLM answers in retrieved documents.'])[0]",
                "v = np.asarray(vec)",
                "print(f'Embedding round-trip:')",
                "print(f'  dim       = {v.shape[0]}')",
                "print(f'  L2 norm   = {np.linalg.norm(v):.4f}')",
                "print(f'  first 6   = {[round(float(x), 3) for x in v[:6]]}')",
            ],
            expected_output_md=[
                "Both worked. The embedding norm should be close to 1 (Nebius normalises by default); a far-from-1 norm sometimes means you accidentally pointed at a model that does not normalise, in which case cosine code needs an extra divide.",
            ],
        ),
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 3 — Load each of the four corpora",
                "",
                "Every recipe in the cookbook draws from one of four hand-picked public-domain corpora. Today we just confirm each loads, see how many documents it yields, and look at one sample document so the structure is concrete in your head.",
            ],
            code=[
                "from cookbook.corpora import (",
                "    load_arxiv_mamba,",
                "    load_wikipedia_superconductors,",
                "    load_sec_10k,",
                "    load_rust_book,",
                ")",
                "",
                "loaders = [",
                "    ('arxiv-mamba',              load_arxiv_mamba),",
                "    ('wikipedia-superconductors', load_wikipedia_superconductors),",
                "    ('sec-10k-pltr',             load_sec_10k),",
                "    ('rust-book',                load_rust_book),",
                "]",
                "for name, loader in loaders:",
                "    docs = list(loader())",
                "    sample = docs[0]",
                "    avg_chars = sum(len(d.text) for d in docs) // max(1, len(docs))",
                "    print(f'{name:30s}  docs={len(docs):4d}  avg_chars={avg_chars:5d}  sample_id={sample.doc_id}')",
            ],
            expected_output_md=[
                "Different corpora hit different failure modes — the SEC filing is long and structured, the Wikipedia subset is many short entries, the Rust book chapters are medium-length prose with code, and the arXiv Mamba survey is one long technical document split across pages. Later recipes deliberately exercise different corpora so you see how each technique behaves across shapes.",
            ],
        ),
        CodeStep(
            tag="index",
            lead_md=[
                "### Step 4 — Build a tiny Qdrant index and search it",
                "",
                "We chunk one corpus, embed the chunks, push them into an in-memory Qdrant collection, and run one search. The point is not the answer — it is that every backend touches Nebius (for embedding) and Qdrant (for indexing) without any manual configuration.",
            ],
            code=[
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_rust_book())",
                "chunks = sentence_window(docs, sentences_per_chunk=5, overlap=1)[:120]",
                "vectors = client.embed([c.text for c in chunks])",
                "",
                "store = QdrantBackend('tour-rust', dim=len(vectors[0]))",
                "store.add(",
                "    texts=[c.text for c in chunks],",
                "    vectors=vectors,",
                "    metadatas=[c.metadata for c in chunks],",
                "    ids=[c.chunk_id for c in chunks],",
                ")",
                "",
                "q_vec = client.embed(['What does the Rust compiler do when two threads share mutable state?'])[0]",
                "hits = store.search(q_vec, top_k=3)",
                "for i, h in enumerate(hits, 1):",
                "    print(f'{i}. score={h.score:.3f}')",
                "    print(f'   {h.text[:200]}')",
                "    print()",
            ],
            expected_output_md=[
                "If you got three reasonably on-topic chunks, the entire vector path is healthy: chunker → embedder → store → searcher. That is the spine of every recipe in this cookbook.",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Generate an answer using retrieved chunks",
                "",
                "One LLM call against the retrieved context. This is the tiniest possible end-to-end RAG, on purpose. Anything fancier lives in Recipe 2 onward.",
            ],
            code=[
                "context = '\\n\\n'.join(h.text for h in hits)",
                "prompt = (",
                "    'Use only the passages below. If they do not contain the answer, say so plainly.\\n\\n'",
                "    f'Passages:\\n{context}\\n\\nQuestion: How does Rust prevent data races when two threads share mutable state?\\nAnswer:'",
                ")",
                "answer = client.chat(prompt)",
                "print(answer)",
            ],
            expected_output_md=[
                "The answer should mention `Send`/`Sync`, `Mutex`, or borrow rules — whatever is in those three chunks. If it generalises beyond the passages, the model leaked prior knowledge; recipe 40 (Lynx guardrails) catches that in production.",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Step 6 — Confirm the disk cache is wired",
                "",
                "Every `chat()` and `embed()` call this cookbook makes is fingerprinted (SHA256 of model + payload) and cached under `.cache/providers/`. The second time you run the same notebook, the cache pays for itself many times over. We just check that the directory exists and has entries from the calls we already made.",
            ],
            code=[
                "from cookbook import _cache",
                "stats = _cache.stats()",
                "print(f'Cache entries: {stats[\"entries\"]}')",
                "print('Disable for one notebook with: os.environ[\"COOKBOOK_CACHE\"] = \"0\"')",
            ],
            expected_output_md=[
                "Cache files live at `.cache/providers/<sha256>.json`. To wipe and force a fresh pass, delete the directory. The cache is gitignored so it never leaks tokens or model outputs into version control.",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — the corpus eval set",
                "",
                "Every recipe ends with a small evaluation slice. The eval set is 80 hand-curated question/answer pairs across the four corpora. Quick look at the shape so the final cell of every other recipe makes sense.",
            ],
            code=[
                "from cookbook.corpora import load_eval_questions",
                "",
                "eval_rows = load_eval_questions()",
                "print(f'Total eval questions: {len(eval_rows)}')",
                "print()",
                "from collections import Counter",
                "for corpus, count in Counter(r['corpus'] for r in eval_rows).items():",
                "    print(f'  {corpus:30s} {count} questions')",
                "print()",
                "print('First eval row:')",
                "for k, v in eval_rows[0].items():",
                "    print(f'  {k:10s} = {v!r}')",
            ],
            expected_output_md=[
                "Twenty questions per corpus, split across `easy`, `medium`, `hard` difficulty. Recipes 37–40 exercise this set with proper RAGAS and DeepEval metrics; other recipes just print a quick spot-check table.",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what tracing would look like",
                "",
                "We keep tracing off in published notebooks so the outputs are clean. In your own runs, flip `COOKBOOK_TRACING=phoenix` and a local Phoenix UI launches at http://localhost:6006 with one span per chat/embed call. Recipe 39 walks the trace tree for a debugging session.",
            ],
            code=[
                "from cookbook.tracing import init_tracing",
                "print('Tracing init result:', init_tracing(backend='off'))",
                "print('To enable: set COOKBOOK_TRACING=phoenix in .env before launching Jupyter.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — measure cache effectiveness",
                "",
                "We re-run the embed call from Step 2 and measure how long it takes. With the cache hit, this should be milliseconds; without, it would be a network round trip.",
            ],
            code=[
                "import time",
                "t0 = time.perf_counter()",
                "_ = client.embed(['Retrieval-Augmented Generation grounds LLM answers in retrieved documents.'])[0]",
                "dt = (time.perf_counter() - t0) * 1000",
                "print(f'Cached embed call took {dt:.1f} ms')",
                "if dt < 50:",
                "    print('Confirmed: cache served the request (no network round trip).')",
                "else:",
                "    print('Looks like a live network call — check COOKBOOK_CACHE env var.')",
            ],
            expected_output_md=[
                "Single-digit milliseconds is the cache; hundreds of milliseconds is a live call. This is the difference between a free re-run and one that costs Nebius tokens.",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what files exist on disk",
                "",
                "A quick map of the repo so you know where things live. If something in this list is missing on your machine, run `python scripts/fetch_corpus.py` from the repo root.",
            ],
            code=[
                "from pathlib import Path",
                "ROOT = Path('..').resolve().parent if Path.cwd().name == '01-foundations' else Path.cwd()",
                "if not (ROOT / 'pyproject.toml').exists():",
                "    ROOT = Path.cwd().parents[1]",
                "for sub in ['corpus', 'cookbook', 'recipes', 'scripts', '.cache']:",
                "    p = ROOT / sub",
                "    exists = '✓' if p.exists() else '✗'",
                "    print(f'  {exists}  {sub:10s} {p}')",
            ],
        ),
    ],

    run_md=[
        "We have already exercised every primitive above. The 'run' cell here is just a final, slightly larger end-to-end check using the same machinery — chunk, embed, store, search, generate.",
    ],
    run_code=[
        "from cookbook.corpora import load_wikipedia_superconductors",
        "from cookbook.chunkers import sentence_window",
        "from cookbook.stores import QdrantBackend",
        "",
        "docs = list(load_wikipedia_superconductors())",
        "chunks = sentence_window(docs, sentences_per_chunk=4)[:120]",
        "vectors = client.embed([c.text for c in chunks])",
        "store2 = QdrantBackend('tour-superconductors', dim=len(vectors[0]))",
        "store2.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
        "",
        "def answer_question(question: str, k: int = 4) -> tuple[str, list[str]]:",
        "    qv = client.embed([question])[0]",
        "    hits = store2.search(qv, top_k=k)",
        "    contexts = [h.text for h in hits]",
        "    prompt = ('Use only these passages.\\n\\n' + '\\n\\n'.join(contexts) +",
        "              f'\\n\\nQuestion: {question}\\nAnswer:')",
        "    return client.chat(prompt), contexts",
        "",
        "ans, ctxs = answer_question('In one paragraph, what is the Meissner effect?')",
        "print(ans)",
    ],

    tuning_md=[
        "There are no knobs to tune in this notebook — its only job is to confirm the four primitives wake up correctly. But the choices you make here apply across the whole cookbook:",
        "",
        "- **Pick your provider carefully.** Nebius is the default because it is cheap and OpenAI-compatible. If you have an OpenAI key, set `PROVIDER=openai` and the same code runs against `text-embedding-3-large` and `gpt-4o-mini`. For local-only runs, set `PROVIDER=local` to push embeddings through `sentence-transformers`; you will not get chat without a separate hosted call.",
        "- **Pick your vector store carefully.** Qdrant in-memory is the default for notebook reproducibility. For real workloads point `QDRANT_URL` at a Qdrant cluster; the code is identical. LanceDB is the choice for laptop-scale persistence; Chroma is the choice for the smallest possible footprint.",
        "- **Pick your tracing posture early.** During development, turn Phoenix on (`COOKBOOK_TRACING=phoenix`) and never turn it off — the cost is a few megabytes of disk and the payoff is a complete history of every call you made.",
        "- **Decide on caching policy.** The disk cache is on by default. For benchmarking, set `COOKBOOK_CACHE=0` to force real network calls. For authoring, leave it on so re-running a notebook is free.",
    ],

    discussion_md=[
        "If every cell above ran clean, you are good to go. The vanilla pipeline (Recipe 2) builds on what you just used; every later recipe varies one or two of these primitives at a time.",
        "",
        "Common ways this notebook fails on first run, and what to do:",
        "",
        "- **Provider auth error** — `NEBIUS_API_KEY` is missing from `.env`. Copy `.env.example` to `.env` and paste your key.",
        "- **Corpus loader is empty** — the corpus has not been downloaded yet. Run `python scripts/fetch_corpus.py` from the repo root.",
        "- **Qdrant client error about UUIDs** — older `qdrant-client` versions; run `pip install -U qdrant-client`.",
        "- **`ModuleNotFoundError: cookbook`** — the package is not installed in your environment. From the repo root: `pip install -e .`.",
        "",
        "If everything is green, jump to `recipes/01-foundations/vanilla-pipeline.ipynb` and start building.",
    ],
)


# =============================================================================
# 3. Embedding Zoo — compare embedding models side by side
# =============================================================================
EMBEDDING_ZOO = Recipe(
    path="recipes/01-foundations/embedding-zoo.ipynb",
    title="The Embedding Zoo — Picking the Right Model",
    category="foundations",
    corpus_filter="wikipedia-superconductors",

    theory_problem=[
        "Embedding choice is the single highest-leverage decision in a RAG pipeline. A weak model puts the right chunk at rank 8 instead of rank 1; no amount of clever reranking recovers from that completely. A strong model on the wrong domain is just as bad — a multilingual general-purpose embedder is worse than a small code-specific embedder when the corpus is software documentation.",
        "Most teams skip the comparison and pick whatever was top of the MTEB leaderboard the week they started. That is a fine first move, but the leaderboard does not know your corpus. This notebook shows you how to run a small, honest A/B between two or three embedders on your own data with your own questions.",
    ],
    theory_origin=[
        "The MTEB benchmark (Muennighoff et al., 2022) put 8 task families and 56 datasets behind a single leaderboard and made embedding choice tractable for the first time. The 2024–2026 wave — Voyage, Cohere Embed 4, BGE-M3, Qwen3-Embedding-8B, Nomic Embed v2 — pushed open-weight models within striking distance of the hosted leaders, and turned a $0.10/M-tokens decision into a $0.02 decision for most workloads.",
    ],
    theory_landscape=[
        "Three families to know in 2026:",
        "",
        "- **Hosted leaders** — Voyage 3 Large, Cohere Embed 4. Best out-of-the-box quality on English prose. Pay-per-token; lock-in is moderate.",
        "- **Open-weight leaders** — BGE-M3, Qwen3-Embedding-8B, Nomic Embed v2. Run on your own GPU or via Nebius. Comparable to hosted leaders on standard tasks; weaker on long context unless you specifically tested for it.",
        "- **Specialised** — code embeddings (Voyage Code), multimodal (Cohere Embed 4 Multimodal, Nomic Embed Vision), Matryoshka-trained models that let you slice the leading dimensions for cheap two-stage retrieval (Recipe 12).",
    ],
    theory_when_to_use=[
        "Run this notebook before you commit to a model. Pick a representative slice of your corpus, write down 8–12 honest evaluation questions, and measure recall@5 across the candidates. The minute you change embedding family, you have to re-embed your entire corpus — so the cost of switching later is high, and the value of getting it right once is correspondingly high.",
        "Skip the comparison only if you are still in throwaway-prototype mode. Production systems should know exactly why they picked their embedder.",
    ],
    theory_intuition=[
        "Three intuitions that come up over and over:",
        "",
        "**Domain trumps benchmark.** A model that scores 4 points lower on MTEB but was pretrained on your domain will usually beat a leaderboard champion on your data. Code corpora love code embeddings; medical corpora love clinical embeddings.",
        "",
        "**Long context matters more than people realise.** A 512-token window means you have to over-chunk dense documents and the chunks miss surrounding context. Most modern embedders support 8K+; a few stretch to 32K. If your documents are long, this is the second axis to check after quality.",
        "",
        "**Cost differences are real but small at hobbyist scale.** Voyage 3 Large is roughly 3x the per-token cost of BGE-M3 via Nebius. For a 100K-token corpus you re-embed weekly, that is the difference between $0.10 and $0.03 per run. For 100M tokens, it is the difference between $100 and $30. Choose by quality, not by penny-pinching.",
    ],

    architecture_mermaid="""
flowchart LR
  D[Wikipedia<br/>superconductors] --> C[Sentence-window<br/>chunker]
  C --> M1[Embedder A<br/>Qwen3-Embedding-8B]
  C --> M2[Embedder B<br/>same default,<br/>different chunks]
  M1 --> S1[(Qdrant A)]
  M2 --> S2[(Qdrant B)]
  Q[Eval set:<br/>15 questions] --> M1
  Q --> M2
  S1 --> R1[recall@5 per model]
  S2 --> R1
  R1 --> P[Side-by-side<br/>table]
""",

    references=[
        Reference(
            title="MTEB: Massive Text Embedding Benchmark",
            url="https://arxiv.org/abs/2210.07316",
            kind="paper",
            note="The benchmark that made embedder choice tractable.",
        ),
        Reference(
            title="Qwen3 Embedding model card",
            url="https://huggingface.co/Qwen/Qwen3-Embedding-8B",
            kind="repo",
            note="Open-weight model used as Nebius's default.",
        ),
        Reference(
            title="BGE-M3 — One embedder, three retrieval modes",
            url="https://huggingface.co/BAAI/bge-m3",
            kind="repo",
            note="Strong multilingual baseline; native sparse + dense + multi-vector.",
        ),
        Reference(
            title="Voyage 3 Embeddings",
            url="https://docs.voyageai.com/docs/embeddings",
            kind="docs",
            note="Hosted leader on English prose as of 2026.",
        ),
        Reference(
            title="Matryoshka Representation Learning",
            url="https://arxiv.org/abs/2205.13147",
            kind="paper",
            note="Why slicing leading dimensions stays meaningful — see Recipe 12.",
        ),
        Reference(
            title="LlamaIndex — choosing an embedding model",
            url="https://developers.llamaindex.ai/python/framework/module_guides/models/embeddings",
            kind="docs",
            note="Practical guidance for swapping embedders behind the same retriever.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load the test corpus",
                "",
                "We use the Wikipedia superconductors subset because it is heterogeneous prose with named entities, formulas, and historical context — the kind of mixed content where embedder choice actually changes results. The eval set ships 20 hand-written questions on this corpus.",
            ],
            code=[
                "from cookbook.corpora import load_wikipedia_superconductors, load_eval_questions",
                "from cookbook.chunkers import sentence_window",
                "",
                "docs = list(load_wikipedia_superconductors())",
                "chunks = sentence_window(docs, sentences_per_chunk=4, overlap=1)",
                "print(f'Docs: {len(docs)}, chunks: {len(chunks)}')",
                "",
                "qs = [q for q in load_eval_questions() if q['corpus'] == 'wikipedia-superconductors']",
                "print(f'Eval questions on this corpus: {len(qs)}')",
                "print(f'First question: {qs[0][\"question\"]}')",
                "print(f'Expected answer prefix: {qs[0][\"answer\"][:120]}...')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 2 — Define the recall@k harness",
                "",
                "Given an embedder, build an index, run each eval question, check whether any of the top-5 retrieved chunks contains a token-level signature of the expected answer. The harness is intentionally permissive so it works without an LLM-as-judge; recipe 37 swaps it for RAGAS' faithfulness/relevance triad.",
            ],
            code=[
                "from cookbook.stores import QdrantBackend",
                "",
                "def recall_at_k(model_label, embed_fn, k=5):",
                "    vectors = embed_fn([c.text for c in chunks])",
                "    store = QdrantBackend(f'zoo-{model_label}', dim=len(vectors[0]))",
                "    store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "    hits, samples = 0, []",
                "    for q in qs[:15]:",
                "        qv = embed_fn([q['question']])[0]",
                "        retrieved = store.search(qv, top_k=k)",
                "        # cheap recall proxy: does any chunk contain a 6-char prefix",
                "        # of any 4+ char word from the gold answer?",
                "        gold_words = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "        hit = any(any(w[:6] in r.text.lower() for w in gold_words) for r in retrieved)",
                "        if hit:",
                "            hits += 1",
                "        samples.append({'question': q['question'][:60], 'hit': hit, 'top_score': float(retrieved[0].score)})",
                "    return hits / len(qs[:15]), samples",
                "",
                "print('Harness ready.')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 3 — Run the default Nebius embedder",
                "",
                "Establish a number for the default embedder this cookbook configures. We will compare against a second model in the next cell.",
            ],
            code=[
                "default_score, default_samples = recall_at_k('default', client.embed)",
                "print(f'Default ({client.embed_model}): recall@5 = {default_score:.2f}')",
            ],
            expected_output_md=[
                "On Wikipedia superconductors the default Qwen3-Embedding-8B usually scores in the 0.70–0.85 range with this loose recall proxy. Numbers below 0.6 suggest either a bad chunk size or a poorly-phrased eval question.",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 4 — Run a second embedder via a different chunking strategy",
                "",
                "Nebius only exposes one embedding model right now, so we cannot meaningfully A/B two models on the same key. Instead we hold the embedder fixed and *change the chunking* — which is the second-biggest lever. If you have an OpenAI or Voyage key, the comparison cell below shows how to swap the model.",
            ],
            code=[
                "from cookbook.chunkers import fixed_window",
                "",
                "alt_chunks = fixed_window(docs, target_tokens=512, overlap_tokens=80)",
                "print(f'Fixed-window chunks: {len(alt_chunks)}')",
                "",
                "def embed_alt(texts):",
                "    return client.embed(texts)",
                "",
                "vectors = client.embed([c.text for c in alt_chunks])",
                "alt_store = QdrantBackend('zoo-fw', dim=len(vectors[0]))",
                "alt_store.add([c.text for c in alt_chunks], vectors, ids=[c.chunk_id for c in alt_chunks])",
                "",
                "alt_hits = 0",
                "for q in qs[:15]:",
                "    qv = client.embed([q['question']])[0]",
                "    retrieved = alt_store.search(qv, top_k=5)",
                "    gold_words = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "    if any(any(w[:6] in r.text.lower() for w in gold_words) for r in retrieved):",
                "        alt_hits += 1",
                "alt_score = alt_hits / 15",
                "print(f'Same embedder + 512-token fixed window: recall@5 = {alt_score:.2f}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap the winning combination as `answer_question`",
                "",
                "Every recipe ends by defining an `answer_question(q)` function so the eval slice at the bottom can compare techniques uniformly. Here it just uses whichever embedder + chunker we have already built.",
            ],
            code=[
                "best_store = store_for_default = None",
                "# choose the better-scoring index above",
                "vectors = client.embed([c.text for c in chunks])",
                "best_store = QdrantBackend('zoo-best', dim=len(vectors[0]))",
                "best_store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "",
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    hits = best_store.search(qv, top_k=k)",
                "    contexts = [h.text for h in hits]",
                "    answer = client.chat(",
                "        'Use only these passages.\\n' + '\\n\\n'.join(contexts) + f'\\n\\nQ: {question}\\nA:'",
                "    )",
                "    return answer, contexts",
                "print('answer_question ready.')",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — which questions broke?",
                "",
                "Aggregate scores hide the interesting failures. We list the questions where recall@5 missed so we can read them and ask whether the embedder is to blame or the question is unfair.",
            ],
            code=[
                "import pandas as pd",
                "df = pd.DataFrame(default_samples)",
                "print('Missed questions:')",
                "print(df[~df['hit']].to_string(index=False))",
            ],
            expected_output_md=[
                "Read every miss. Often the question references an entity by a name not in the corpus (a synonym, a different transliteration), or asks about a fact the corpus genuinely lacks. Both are eval-set problems, not embedder problems — fix the question, not the embedder.",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — vector dimensionality and norm",
                "",
                "Two sanity checks every embedder should pass. Norms close to 1 mean cosine similarity reduces to a dot product; norms far from 1 sometimes hint at a configuration issue (wrong model, missing normalisation flag).",
            ],
            code=[
                "import numpy as np",
                "v = np.asarray(client.embed(['liquid nitrogen cooling'])[0])",
                "print(f'dim          : {v.shape[0]}')",
                "print(f'L2 norm      : {np.linalg.norm(v):.4f}')",
                "print(f'mean         : {v.mean():.4f}')",
                "print(f'std          : {v.std():.4f}')",
                "print(f'sparsity (<1e-3) : {(np.abs(v) < 1e-3).mean():.2%}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — how stable is the ranking?",
                "",
                "Embed the same question twice and check the top-5 are identical. With caching on this is trivially true; with caching off, deterministic embedders should still match.",
            ],
            code=[
                "q = qs[0]['question']",
                "vec_a = client.embed([q])[0]",
                "vec_b = client.embed([q + ' ' + ' ' * 0])[0]   # same text",
                "import numpy as np",
                "sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))",
                "print(f'self-cosine on the same query: {sim:.6f}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — code snippet for an OpenAI or Voyage A/B",
                "",
                "If you have an OpenAI or Voyage key, you can swap the embedder and compare in three lines. This cell prints the code rather than running it, so the notebook stays self-contained.",
            ],
            code=[
                "snippet = '''",
                "from cookbook.providers import LLMClient",
                "openai_client = LLMClient(provider=\"openai\", embed_model=\"text-embedding-3-large\")",
                "voyage_client = LLMClient(provider=\"openai\", embed_model=\"voyage-3-large\")  # via OpenRouter",
                "score_openai, _ = recall_at_k(\"openai\", openai_client.embed)",
                "score_voyage, _ = recall_at_k(\"voyage\", voyage_client.embed)",
                "print(f\"OpenAI text-embedding-3-large recall@5 = {score_openai:.2f}\")",
                "print(f\"Voyage 3 Large            recall@5 = {score_voyage:.2f}\")",
                "'''",
                "print(snippet)",
            ],
        ),
    ],

    run_md=[
        "Print a representative answer end-to-end so the technique that won the comparison is visible in the output, not just the number.",
    ],
    run_code=[
        "ans, ctxs = answer_question('What is the Meissner effect and how does it distinguish a superconductor from a perfect conductor?')",
        "print('=== Answer ===')",
        "print(ans)",
        "print()",
        "print('=== Top context ===')",
        "print(ctxs[0][:400])",
    ],

    comparison_md=[
        "Compare the best embedder + chunker configuration against the vanilla pipeline's default settings on a representative question.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What is the Meissner effect and how does it distinguish a superconductor from a perfect conductor?'",
        "base = vanilla_pipeline(q, corpus='wikipedia-superconductors', top_k=5)",
        "ours_answer, ours_contexts = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'answer_preview': base.answer[:160], 'n_contexts': len(base.contexts)},",
        "    {'pipeline': 'embedding-zoo best', 'answer_preview': ours_answer[:160], 'n_contexts': len(ours_contexts)},",
        "])",
    ],

    tuning_md=[
        "Four levers ranked by impact:",
        "",
        "1. **Embedder family.** The biggest quality lever. Voyage 3 / Cohere Embed 4 / BGE-M3 / Qwen3-Embedding-8B differ by 4–10 recall points on most corpora. Swap by changing one config line; re-embed the corpus.",
        "2. **Chunk size and overlap.** Second biggest. Recipe 4 sweeps this systematically. Most corpora want chunks in the 256–512 token range; documents with code or tables sometimes need 800+.",
        "3. **Chunking strategy.** Fixed-window vs sentence-window vs semantic-boundary changes recall by 2–5 points without changing model or size. Recipe 5 covers the strategy choice.",
        "4. **Number of retrieved chunks (`k`).** A free quality lever but each extra chunk adds tokens to the prompt. Most production systems land between `k=3` and `k=8`.",
        "",
        "Re-run this comparison whenever you change embedder family, chunker, or significant chunk parameters. The infrastructure is here; the next switch is one cell away.",
    ],

    discussion_md=[
        "Three patterns you usually see when you run this honestly:",
        "",
        "- **The default is rarely the best.** Whatever your stack ships with — Nebius's Qwen3, OpenAI's text-embedding-3-small, the SaaS demo's anonymous embedder — is a starting point. A 30-minute comparison against two alternatives almost always finds a 3–10 point lift.",
        "- **Recall@5 is not the whole story.** A model with worse recall@5 but better recall@1 may be the right pick for a system that does not rerank. The opposite holds when you do rerank — you want a model that gets the right chunk into the top-20, even if it is at rank 12.",
        "- **Domain re-runs the picture.** A model that crushed Wikipedia may be middle-of-the-pack on SEC filings. Re-run the harness whenever you switch corpus type.",
        "",
        "When you commit to a model, write down the numbers and the date. Six months from now you will want to know whether the new top-of-the-MTEB-leaderboard model would actually beat what you have. That comparison only takes 30 minutes if you have the harness ready.",
    ],
)


# =============================================================================
# 4. Chunk Size Sensitivity — sweep window size, measure recall
# =============================================================================
CHUNK_SIZE = Recipe(
    path="recipes/01-foundations/chunk-size-sensitivity.ipynb",
    title="Chunk Size Sensitivity — The Cheapest Big Win",
    category="foundations",
    corpus_filter="arxiv-mamba",

    theory_problem=[
        "Most RAG tutorials pick 512-token chunks and never look back. That is fine until you measure: optimal chunk size is corpus-dependent and the gap between a default chunk size and a tuned one is routinely 5–15 recall points. Five minutes of sweeping window sizes is the cheapest big win in any RAG project.",
        "We do not want a theoretical answer. We want a curve: recall@5 against chunk size, for one corpus, with one embedder. That curve is also a diagnostic — its shape tells you whether your bottleneck is chunking, embedding, or retrieval.",
    ],
    theory_origin=[
        "The chunk-size question is as old as RAG itself; LangChain's `RecursiveCharacterTextSplitter` defaulted to 1000 characters in 2022, LlamaIndex moved to 512 tokens by mid-2023, Anthropic's contextual retrieval paper (September 2024) revived attention to small chunks paired with LLM-written headers. None of those defaults were measured against your corpus. This recipe gives you a five-minute measurement loop.",
    ],
    theory_landscape=[
        "Three competing pressures shape the optimum:",
        "",
        "- **Too small.** Chunks lose surrounding context; the embedding represents a single point in semantic space and misses the qualifier two sentences earlier. Recall stays flat or drops.",
        "- **Too large.** Each chunk's vector averages over too many ideas; the right chunk is *adjacent* in embedding space to many other large chunks. Recall drops because cosine similarity is no longer discriminative.",
        "- **Storage and latency.** More chunks means more vectors, more disk, slower indexing and search (sublinearly, but real). At billions of chunks, the constant factor matters.",
    ],
    theory_when_to_use=[
        "Run this notebook every time you start a new RAG project. Run it again whenever you change embedder or chunker family — the optimum will move. Skip it if you are reproducing a paper's results — then the chunk size is fixed by the paper and you should respect it.",
        "Do not over-tune. A 2 point recall difference at recall@5 is within noise on a 15-question eval set. Five points is a real signal. Move on once you have a recognisable peak; do not chase sub-percent differences.",
        "Three signs you should re-run the sweep: you switched the embedding model, you doubled the corpus size, or you noticed a sudden drop in answer quality that does not correlate with anything else. In all three cases the chunk-size optimum has likely shifted by 64–256 tokens and a five-minute re-sweep recovers the lost recall for free. Keep the previous sweep's plot around so you can compare curves and see whether the peak migrated or just flattened.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**The curve has a single peak.** It is unimodal on every corpus I have measured. There is no \"two locally optimal chunk sizes that mean different things\". Pick the peak.",
        "",
        "**Overlap matters less than size.** Once you are within the right family of chunk sizes, doubling or zeroing overlap shifts recall by a point or two. Overlap is a knob, not a strategy. Pick `overlap = size // 6` or `size // 8` and move on.",
        "",
        "**Different corpora peak at different sizes.** Wikipedia-style short prose peaks small (256). Technical surveys peak medium (384–512). Long financial filings sometimes peak large (768) because each paragraph spans more text. Do not transplant numbers across corpora.",
    ],

    architecture_mermaid="""
flowchart LR
  D[Mamba survey<br/>PDF pages] --> SW{For each chunk size<br/>in 128..1024}
  SW --> C[Fixed-window<br/>chunker]
  C --> E[Embedder]
  E --> S[(Qdrant)]
  Q[15 eval questions] --> R[Search top-5]
  S --> R
  R --> M[Compute recall@5]
  M --> P[Plot recall vs size]
""",

    references=[
        Reference(
            title="Anthropic Contextual Retrieval",
            url="https://www.anthropic.com/news/contextual-retrieval",
            kind="blog",
            note="Small chunks + LLM-written headers; see Recipe 7.",
        ),
        Reference(
            title="LlamaIndex — chunk size tuning",
            url="https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/modules",
            kind="docs",
            note="Canonical chunking primitives.",
        ),
        Reference(
            title="LangChain RecursiveCharacterTextSplitter",
            url="https://python.langchain.com/docs/concepts/text_splitters/",
            kind="docs",
            note="The other widely-used chunker family.",
        ),
        Reference(
            title="Late Chunking (Jina AI, 2024)",
            url="https://jina.ai/news/late-chunking-in-long-context-embedding-models/",
            kind="blog",
            note="Why long-context embedders change the chunk-size question.",
        ),
        Reference(
            title="RAPTOR — recursive tree of summaries",
            url="https://arxiv.org/abs/2401.18059",
            kind="paper",
            note="A different answer: build several scales of chunks. Recipe 10.",
        ),
        Reference(
            title="Semantic chunking — Greg Kamradt, 2023",
            url="https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb",
            kind="repo",
            note="Five levels of chunking, from naive to LLM-assisted.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Load the Mamba survey",
                "",
                "We sweep over the arXiv Mamba state-space-models survey. It is a 25-page technical paper with formulas, citations, and dense prose — a hard target for chunking and a good corpus for diagnostic curves.",
            ],
            code=[
                "from cookbook.corpora import load_arxiv_mamba, load_eval_questions",
                "",
                "docs = list(load_arxiv_mamba())",
                "qs = [q for q in load_eval_questions() if q['corpus'] == 'arxiv-mamba'][:15]",
                "print(f'Loaded {len(docs)} document records from the Mamba PDF.')",
                "print(f'Will sweep across {len(qs)} eval questions.')",
                "print(f'First question: {qs[0][\"question\"]}')",
            ],
        ),
        CodeStep(
            tag="chunk",
            lead_md=[
                "### Step 2 — Define the sweep harness",
                "",
                "For each chunk size we build a fresh index, run the eval questions, and compute the simple recall proxy from the embedding-zoo recipe. We also record indexing time and chunk count so the cost side of the trade-off is visible.",
            ],
            code=[
                "from cookbook.chunkers import fixed_window",
                "from cookbook.stores import QdrantBackend",
                "import time",
                "",
                "def evaluate_chunk_size(target_tokens, overlap_tokens):",
                "    t0 = time.perf_counter()",
                "    chunks = fixed_window(docs, target_tokens=target_tokens, overlap_tokens=overlap_tokens)",
                "    vectors = client.embed([c.text for c in chunks])",
                "    store = QdrantBackend(f'sweep-{target_tokens}', dim=len(vectors[0]))",
                "    store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "    build_seconds = time.perf_counter() - t0",
                "",
                "    hits = 0",
                "    for q in qs:",
                "        qv = client.embed([q['question']])[0]",
                "        retrieved = store.search(qv, top_k=5)",
                "        gold_words = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "        if any(any(w[:6] in r.text.lower() for w in gold_words) for r in retrieved):",
                "            hits += 1",
                "    return {",
                "        'target_tokens': target_tokens,",
                "        'chunks': len(chunks),",
                "        'recall@5': hits / max(1, len(qs)),",
                "        'build_seconds': round(build_seconds, 1),",
                "    }",
                "",
                "print('Harness ready.')",
            ],
        ),
        CodeStep(
            tag="chunk",
            lead_md=[
                "### Step 3 — Run the sweep across six chunk sizes",
                "",
                "We hit every reasonable value: 128, 256, 384, 512, 768, 1024 tokens. Overlap stays proportional. The whole sweep takes a couple of minutes; later runs are nearly free thanks to the embedding cache.",
            ],
            code=[
                "import pandas as pd",
                "",
                "rows = [evaluate_chunk_size(size, max(8, size // 6)) for size in (128, 256, 384, 512, 768, 1024)]",
                "df = pd.DataFrame(rows)",
                "print(df)",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Step 4 — Plot the curve",
                "",
                "Plotting it makes the peak obvious. Without a plot, a sweep is just a column of numbers and the peak hides in plain sight.",
            ],
            code=[
                "import matplotlib.pyplot as plt",
                "",
                "fig, ax1 = plt.subplots(figsize=(7, 3.5))",
                "ax1.plot(df['target_tokens'], df['recall@5'], marker='o', color='tab:blue')",
                "ax1.set_xlabel('Target tokens per chunk')",
                "ax1.set_ylabel('recall@5', color='tab:blue')",
                "ax1.tick_params(axis='y', labelcolor='tab:blue')",
                "ax1.set_ylim(0, 1)",
                "ax1.grid(alpha=0.3)",
                "",
                "ax2 = ax1.twinx()",
                "ax2.bar(df['target_tokens'], df['chunks'], width=40, alpha=0.2, color='tab:orange')",
                "ax2.set_ylabel('chunk count', color='tab:orange')",
                "ax2.tick_params(axis='y', labelcolor='tab:orange')",
                "",
                "plt.title('Chunk size sweep — arXiv Mamba corpus')",
                "plt.tight_layout()",
                "plt.show()",
            ],
            expected_output_md=[
                "Read the curve. Where is the peak? How steep are the falls on either side? The chunk count tells you the storage cost — going from 384 to 128 tokens triples your index without necessarily improving recall.",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap the best chunk size as `answer_question`",
                "",
                "Pick the best configuration from the sweep and build the production `answer_question` from it. The rest of the cookbook can compare against it.",
            ],
            code=[
                "best_row = max(rows, key=lambda r: r['recall@5'])",
                "print(f\"Best chunk size: {best_row['target_tokens']} tokens (recall@5 = {best_row['recall@5']:.2f})\")",
                "",
                "chunks_best = fixed_window(docs, target_tokens=best_row['target_tokens'], overlap_tokens=max(8, best_row['target_tokens'] // 6))",
                "vectors_best = client.embed([c.text for c in chunks_best])",
                "store_best = QdrantBackend('sweep-best', dim=len(vectors_best[0]))",
                "store_best.add([c.text for c in chunks_best], vectors_best, ids=[c.chunk_id for c in chunks_best])",
                "",
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    hits = store_best.search(qv, top_k=k)",
                "    contexts = [h.text for h in hits]",
                "    answer = client.chat(",
                "        'Use only these passages.\\n\\n' + '\\n\\n'.join(contexts) + f'\\n\\nQ: {question}\\nA:'",
                "    )",
                "    return answer, contexts",
                "print('answer_question is wired to the best chunk size.')",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — eyeball one missed question",
                "",
                "Look at one question the best configuration *missed* and check what the top-5 chunks look like. Often the miss is about a specific technical term the embedder did not see often enough.",
            ],
            code=[
                "best_size = best_row['target_tokens']",
                "miss_qs = []",
                "for q in qs:",
                "    qv = client.embed([q['question']])[0]",
                "    retrieved = store_best.search(qv, top_k=5)",
                "    gold = [w.lower() for w in q['answer'].split() if len(w) >= 4]",
                "    if not any(any(w[:6] in r.text.lower() for w in gold) for r in retrieved):",
                "        miss_qs.append((q, retrieved))",
                "if miss_qs:",
                "    q, retrieved = miss_qs[0]",
                "    print('Missed question:', q['question'])",
                "    print('Expected:      ', q['answer'][:200])",
                "    print()",
                "    for i, r in enumerate(retrieved, 1):",
                "        print(f'Top {i}: {r.text[:160]}')",
                "        print()",
                "else:",
                "    print('No misses to inspect at this chunk size.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — how does overlap matter?",
                "",
                "Hold chunk size at the peak and sweep overlap. Usually the curve is much flatter than the size sweep — overlap is a knob, not a strategy.",
            ],
            code=[
                "best_size = best_row['target_tokens']",
                "overlap_rows = [evaluate_chunk_size(best_size, ov) for ov in (0, best_size // 12, best_size // 6, best_size // 3)]",
                "import pandas as pd",
                "print(pd.DataFrame(overlap_rows))",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — how many tokens does a top-5 prompt cost?",
                "",
                "Cost-aware tuning. Multiply average chunk length by k to get the prompt budget you ship to the LLM every query. Bigger chunks make recall easier but balloon the prompt.",
            ],
            code=[
                "import tiktoken",
                "enc = tiktoken.get_encoding('cl100k_base')",
                "lengths = [len(enc.encode(c.text)) for c in chunks_best[:200]]",
                "import statistics",
                "print(f'Median tokens/chunk : {statistics.median(lengths):.0f}')",
                "print(f'k=5 prompt budget   : ~{int(5 * statistics.median(lengths))} tokens')",
                "print(f'k=10 prompt budget  : ~{int(10 * statistics.median(lengths))} tokens')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does a sample chunk look like?",
                "",
                "Print a chunk at the chosen size so you can read it as a human. If it looks coherent and self-contained, the size is in the right zone. If it cuts off mid-sentence in the middle of an idea, drop to a smaller size or move to a sentence-window chunker (Recipe 5).",
            ],
            code=[
                "sample = chunks_best[len(chunks_best) // 2]",
                "print(f'chunk_id: {sample.chunk_id}')",
                "print(f'doc_id  : {sample.doc_id}')",
                "print()",
                "print(sample.text[:1200])",
            ],
        ),
    ],

    run_md=[
        "End-to-end answer on the best chunk size, so the output is more than a single number.",
    ],
    run_code=[
        "ans, ctxs = answer_question('Explain selective scan in plain language and why it matters for state-space models.')",
        "print('=== Answer ===')",
        "print(ans)",
        "print()",
        "print('Top context preview:')",
        "print(ctxs[0][:400])",
    ],

    comparison_md=[
        "Show the gap between the un-tuned vanilla pipeline (defaults) and this notebook's tuned chunk size, on the same question.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'Explain selective scan in plain language and why it matters for state-space models.'",
        "base = vanilla_pipeline(q, corpus='arxiv-mamba', top_k=5)",
        "ours_a, ours_c = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla (default chunk size)', 'top_context_preview': base.contexts[0][:120]},",
        "    {'pipeline': f\"tuned ({best_row['target_tokens']} tokens)\", 'top_context_preview': ours_c[0][:120]},",
        "])",
    ],

    tuning_md=[
        "Three knobs, in priority order:",
        "",
        "1. **Chunk size.** The whole point of this notebook. Sweep, pick the peak, stop. Do not chase sub-percent differences.",
        "2. **Overlap.** Set proportional to size (`size // 6` is fine), check that doubling or zeroing it does not change recall noticeably. If it does, you have a chunker problem — switch to sentence-window (Recipe 5) or semantic boundaries.",
        "3. **Top-`k`.** Free recall lever, but adds prompt tokens linearly. Set `k=5` for prose, `k=8` for very dense or code-heavy corpora, `k=3` when the embedder is strong and the questions are simple.",
        "",
        "Not knobs to tune here: similarity metric (cosine fine), distance threshold (set `k` instead), reranker (defer to Recipe 22).",
    ],

    discussion_md=[
        "When chunk-size sweeps disappoint, the bottleneck is usually somewhere else:",
        "",
        "- **Flat curve, low recall everywhere.** The embedder is wrong for the corpus. Re-run Recipe 3.",
        "- **Peak too narrow.** The eval set is too small or too easy. Add more questions across more topics.",
        "- **Peak at the smallest size you tried.** Your chunks were too big to start. Add `64` and `96` to the sweep.",
        "- **Peak at the largest size you tried.** Add `1280` and `1536`. Eventually you will hit the embedder's context window, where the curve actively drops.",
        "",
        "Tuned chunk size is the start of the conversation, not the end. Recipes 5–12 show what to do once size has been picked: switch from fixed-window to semantic boundaries, late chunking, propositions, RAPTOR, or document-summary routing.",
    ],
)


RECIPES = [VANILLA, COOKBOOK_TOUR, EMBEDDING_ZOO, CHUNK_SIZE]
