"""Category 6 — Adaptive & Agentic RAG.

This file holds the seven adaptive/agentic recipes. Batch 2 ships the first
three (the canonical reference shapes that set the depth bar for the rest of
the cookbook). The other four (speculative, langgraph, mcp, dspy) are added
in Batch 7.
"""
from __future__ import annotations

from authoring import CodeStep, Recipe, Reference


# =============================================================================
# Self-Reflective Retrieval (Self-RAG)
# =============================================================================
SELF_REFLECTIVE = Recipe(
    path="recipes/06-adaptive-and-agentic/self-reflective-retrieval.ipynb",
    title="Self-Reflective Retrieval — Letting the Model Decide",
    category="adaptive-and-agentic",
    corpus_filter="rust-book",

    theory_problem=[
        "Vanilla RAG retrieves on every query, regardless of whether retrieval helps. For trivia the model already knows, retrieval injects noise. For questions outside the corpus, retrieval drags the model toward confidently-wrong neighbours. For multi-hop questions, a single retrieval pass is structurally insufficient. The fix is to let the model decide — should I retrieve, and is what I got any good?",
        "Self-RAG (Asai et al., ICLR 2024) trained a model that emits reflection tokens during generation: a `[Retrieve]` decision before retrieving, a `[IsRel]` decision after each retrieved passage, a `[IsSup]` decision over whether the answer is supported. The published recipes use the trained model; this notebook reproduces the *behaviour* with prompt-engineered reflection so it works on any base model — Llama, Qwen, GPT-4o, Claude — without specialised weights.",
    ],
    theory_origin=[
        "Self-RAG was published at ICLR 2024 by Akari Asai and colleagues at the University of Washington. The paper showed that explicit reflection tokens, trained via instruction-tuning and a custom critic model, beat both vanilla RAG and chain-of-thought RAG on TriviaQA, PubHealth, and ARC-Challenge. The headline number was a 4–7 point gain on PubHealth grounding, but the more important contribution was the *framing*: retrieval is a decision, not a default.",
        "By mid-2025 the prompt-engineered variant — ask the base model the same yes/no questions — was the dominant production pattern, because it works on any model and the quality gap to the fine-tuned variant is small. We use the prompt variant here so the recipe runs on any provider.",
    ],
    theory_landscape=[
        "Three cousins in this category, each making a different decision explicit:",
        "",
        "- **Self-RAG (this recipe)** — should I retrieve at all, and are the passages useful? Reflection happens per query and per passage.",
        "- **CRAG (Recipe 25)** — given that I retrieved, are the passages good enough? If not, fall back to web search. Reflection happens after retrieval, before generation.",
        "- **Adaptive-RAG (Recipe 26)** — classify the question into {no retrieval, single-shot, multi-hop} and dispatch. Reflection happens before retrieval, on the query alone.",
        "",
        "Real production systems often stack all three. Adaptive-RAG routes; Self-RAG decides per-passage usefulness; CRAG catches the case where every passage is unhelpful. The cost is three extra LLM calls per query; the payoff is sharply lower hallucination and much better behaviour on out-of-corpus questions.",
    ],
    theory_when_to_use=[
        "Use Self-RAG when your corpus covers some but not all of what users will ask. Customer-support knowledge bases, internal wikis, technical documentation — anywhere the user might reasonably ask something the corpus does not address. The reflection layer turns those queries into honest \"I don't have evidence for that\" answers instead of confident hallucinations.",
        "Skip it when your corpus is exhaustive and on-topic by construction. A FAQ retrieval system over a closed Q&A set rarely needs reflection — the answer is always there. The extra LLM calls just add latency.",
        "Skip it also when latency matters more than quality. Each reflection adds 200–500 ms; on a fast model that is fine, on a slower model it adds up. Recipe 27 (Speculative RAG) is the right answer when you need both reflection and low latency.",
    ],
    theory_intuition=[
        "Three intuitions that explain why this works:",
        "",
        "**Models are honest when explicitly asked.** Ask a model \"Is this passage useful for answering X?\" with a yes/no constraint and it tells you. Ask it to answer X with that passage in context and it will use the passage even when it should not. The reflection prompt is a different lens than the generation prompt.",
        "",
        "**Per-passage filtering scales gracefully.** k=10 retrieval with passage-level filtering routinely beats k=5 retrieval without. You can be generous with retrieval recall as long as the reflection layer is cheap enough to scale linearly.",
        "",
        "**Refusals are a feature, not a bug.** A system that says \"I cannot answer this confidently from the corpus\" is more trustworthy than one that hallucinates. Self-RAG buys you that refusal capability without changing the base model.",
    ],

    architecture_mermaid="""
flowchart TB
  Q[User question] --> DEC{Retrieve?<br/>LLM yes/no}
  DEC -->|no| ANS1[Answer<br/>from world knowledge]
  DEC -->|yes| R[Retrieve top-k]
  R --> F{For each passage:<br/>useful?}
  F -->|drop| F
  F -->|keep| K[Useful passages]
  K --> GEN[Answer using<br/>kept passages]
  GEN --> SUP{Is the answer<br/>supported?}
  SUP -->|yes| OUT[Return answer]
  SUP -->|no| RETRY[Retry or refuse]
  ANS1 --> OUT
""",

    references=[
        Reference(
            title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
            url="https://arxiv.org/abs/2310.11511",
            kind="paper",
            note="The original ICLR 2024 paper from Asai et al.",
        ),
        Reference(
            title="Self-RAG official repository",
            url="https://github.com/AkariAsai/self-rag",
            kind="repo",
            note="Trained reflection-token weights and training scripts.",
        ),
        Reference(
            title="LangGraph Self-RAG tutorial",
            url="https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/",
            kind="docs",
            note="Reference implementation as a stateful graph.",
        ),
        Reference(
            title="CRAG: Corrective Retrieval Augmented Generation",
            url="https://arxiv.org/abs/2401.15884",
            kind="paper",
            note="Related work; covered in Recipe 25.",
        ),
        Reference(
            title="Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity",
            url="https://arxiv.org/abs/2403.14403",
            kind="paper",
            note="The third cousin; covered in Recipe 26.",
        ),
        Reference(
            title="The Self-RAG paper, one-page summary",
            url="https://arize.com/blog/self-rag/",
            kind="blog",
            note="Useful illustrated walk-through of the reflection tokens.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Set up the corpus and an index",
                "",
                "We use the Rust book chapters. The corpus deliberately covers some questions (Rust syntax, ownership, async) but not others (assembly, Python, history of programming languages). That mix lets us exercise the reflection logic on both kinds of queries.",
            ],
            code=[
                "from cookbook.corpora import load_rust_book",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_rust_book())",
                "chunks = sentence_window(docs, sentences_per_chunk=4, overlap=1)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('selfrag', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} chunks from {len(docs)} chapters.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — The `[Retrieve?]` decision",
                "",
                "Before any retrieval, ask the model whether this corpus is relevant to the question. We constrain the response to exactly two tokens so parsing is trivial. This is the cheapest possible reflection step and it pays for itself on out-of-corpus questions.",
            ],
            code=[
                "RETRIEVE_DECISION = (",
                "    'You have access to a knowledge base about the Rust programming language. '",
                "    'For the following question, would consulting that knowledge base materially help? '",
                "    'Reply with exactly one word: RETRIEVE or NO_RETRIEVE.\\n\\n'",
                "    'Question: {q}\\n\\nAnswer:'",
                ")",
                "",
                "def needs_retrieval(question: str) -> bool:",
                "    out = client.chat(RETRIEVE_DECISION.format(q=question)).strip().upper()",
                "    return 'NO_RETRIEVE' not in out.split()[:2]",
                "",
                "probes = [",
                "    'What is the borrow checker in Rust?',",
                "    'Who wrote The Great Gatsby?',",
                "    'How do I send a value across threads safely in Rust?',",
                "    'What is the chemical formula for water?',",
                "]",
                "for p in probes:",
                "    print(f'  {needs_retrieval(p)!s:5s}  -> {p}')",
            ],
            expected_output_md=[
                "The model should say `True` for Rust questions and `False` for the others. If it says `True` for everything, the prompt is too lenient — tighten the wording. If it says `False` for Rust questions, the prompt is too strict.",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 3 — Retrieve with k=8 (generous recall)",
                "",
                "With reflection filtering downstream, we can afford generous initial recall. We pull eight candidates and let the next step drop unhelpful ones. Without reflection we would stick to k=5 to keep the prompt tight.",
            ],
            code=[
                "def retrieve_candidates(question: str, k: int = 8):",
                "    qv = client.embed([question])[0]",
                "    return store.search(qv, top_k=k)",
                "",
                "candidates = retrieve_candidates('How does the borrow checker enforce exclusive mutable references?')",
                "for i, h in enumerate(candidates, 1):",
                "    print(f'  {i}. score={h.score:.3f}  {h.text[:140]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — The `[IsRel]` per-passage filter",
                "",
                "For each candidate passage, ask the model whether it materially helps answer the question. Drop passages that get `NOT_USEFUL`. This is where the bulk of the reflection budget is spent — k LLM calls per query — but each call is short and the impact on quality is large.",
            ],
            code=[
                "USEFUL_DECISION = (",
                "    'Question: {q}\\n\\n'",
                "    'Passage: {p}\\n\\n'",
                "    'Does this passage materially help answer the question? '",
                "    'Reply with exactly one word: USEFUL or NOT_USEFUL.'",
                ")",
                "",
                "def filter_useful(question: str, candidates) -> list:",
                "    kept = []",
                "    for h in candidates:",
                "        verdict = client.chat(USEFUL_DECISION.format(q=question, p=h.text[:800])).strip().upper()",
                "        if 'USEFUL' in verdict and 'NOT_USEFUL' not in verdict:",
                "            kept.append(h)",
                "    return kept",
                "",
                "question = 'How does the borrow checker enforce exclusive mutable references?'",
                "kept = filter_useful(question, candidates)",
                "print(f'Kept {len(kept)} of {len(candidates)} candidates.')",
                "for h in kept:",
                "    print(f'  - {h.text[:140]}')",
            ],
            expected_output_md=[
                "Usually 3–5 of the 8 candidates survive. The dropped passages tend to be marginally on-topic — they mention the right keywords but discuss a different aspect of the topic. Compare the dropped passages to the kept ones manually to build intuition for what your model considers \"useful\".",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Generate from the surviving passages",
                "",
                "Standard stuffed-context generation, but now only over passages that survived the filter. If zero passages survived, refuse honestly rather than answer from nothing.",
            ],
            code=[
                "GENERATE_PROMPT = (",
                "    'Use ONLY the passages below to answer the question. '",
                "    'If they do not contain the answer, say so plainly.\\n\\n'",
                "    'Passages:\\n{context}\\n\\nQuestion: {question}\\nAnswer:'",
                ")",
                "",
                "def generate_answer(question: str, passages: list) -> str:",
                "    if not passages:",
                "        return 'No useful evidence retrieved. Cannot answer from this corpus.'",
                "    context = '\\n\\n'.join(p.text for p in passages)",
                "    return client.chat(GENERATE_PROMPT.format(context=context, question=question))",
                "",
                "answer = generate_answer(question, kept)",
                "print(answer)",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 6 — The `[IsSup]` support check",
                "",
                "Ask the model whether its own answer is supported by the passages it used. This catches the case where the model wrote a confident-sounding answer that drifted beyond the retrieved evidence. The check is one LLM call; on production systems it gates the response.",
            ],
            code=[
                "SUPPORT_CHECK = (",
                "    'Passages:\\n{context}\\n\\n'",
                "    'Answer: {answer}\\n\\n'",
                "    'Is every factual claim in the answer supported by the passages? '",
                "    'Reply with exactly one word: SUPPORTED or UNSUPPORTED.'",
                ")",
                "",
                "def check_support(answer: str, passages: list) -> bool:",
                "    if not passages:",
                "        return False",
                "    context = '\\n\\n'.join(p.text for p in passages)",
                "    verdict = client.chat(SUPPORT_CHECK.format(context=context, answer=answer)).strip().upper()",
                "    return 'SUPPORTED' in verdict and 'UNSUPPORTED' not in verdict",
                "",
                "supported = check_support(answer, kept)",
                "print(f'Answer supported by retrieved passages: {supported}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 7 — Glue it all together",
                "",
                "The whole Self-RAG loop in one function. This is what the rest of the cookbook compares against the vanilla baseline.",
            ],
            code=[
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    if not needs_retrieval(question):",
                "        free = client.chat(f'Answer concisely from your own knowledge: {question}')",
                "        return free, []",
                "    candidates = retrieve_candidates(question, k=8)",
                "    kept = filter_useful(question, candidates)",
                "    answer = generate_answer(question, kept)",
                "    if not check_support(answer, kept):",
                "        answer = 'I do not have sufficient evidence in the corpus to answer that confidently.\\n\\n' + answer",
                "    return answer, [h.text for h in kept]",
                "",
                "ans, ctxs = answer_question('What is interior mutability and how does RefCell expose it?')",
                "print(ans)",
                "print()",
                "print(f'(used {len(ctxs)} passages)')",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — does the `[Retrieve?]` decision behave well across question types?",
                "",
                "Run a small battery of clearly-on-topic, clearly-off-topic, and ambiguous questions. The decision should be on for the first group, off for the second, and the ambiguous ones tell you where your prompt sits.",
            ],
            code=[
                "battery = [",
                "    ('on-topic',  'What is ownership in Rust?'),",
                "    ('on-topic',  'How does async/await work in Rust?'),",
                "    ('off-topic', 'What year was the French Revolution?'),",
                "    ('off-topic', 'What is the chemical formula of water?'),",
                "    ('ambiguous', 'How is concurrency different from parallelism in general?'),",
                "    ('ambiguous', 'Compare garbage collection to manual memory management.'),",
                "]",
                "for kind, q in battery:",
                "    print(f'  {kind:10s}  retrieve={needs_retrieval(q)!s:5s}  q={q}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what fraction of passages survive the [IsRel] filter?",
                "",
                "Across several questions, see how aggressive the filter is. A typical Self-RAG run drops 30–60 % of candidates. If your filter keeps everything, it is too lenient and saves you nothing. If it drops everything, it is too strict and the model will refuse perfectly answerable questions.",
            ],
            code=[
                "import statistics",
                "rates = []",
                "for q in [",
                "    'How does the borrow checker enforce exclusive references?',",
                "    'What is Rc and when should I use it?',",
                "    'When should I pick Arc over Mutex?',",
                "    'What is the difference between String and &str?',",
                "    'How does Rust handle error propagation with the question mark operator?',",
                "]:",
                "    cs = retrieve_candidates(q, k=8)",
                "    kept = filter_useful(q, cs)",
                "    rates.append(len(kept) / len(cs))",
                "    print(f'  kept {len(kept):2d}/{len(cs)} for: {q[:60]}')",
                "print()",
                "print(f'Mean keep rate: {statistics.mean(rates):.0%}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what happens on an out-of-corpus question?",
                "",
                "Ask something the Rust book genuinely does not cover. The whole Self-RAG loop should either skip retrieval or refuse gracefully — not hallucinate.",
            ],
            code=[
                "out_of_corpus_q = 'What is the relationship between Hamiltonian mechanics and Lagrangian mechanics?'",
                "ans, used = answer_question(out_of_corpus_q)",
                "print(ans)",
                "print()",
                "print(f'(used {len(used)} passages)')",
            ],
            expected_output_md=[
                "Two acceptable behaviours: (1) the `[Retrieve?]` step returns `NO_RETRIEVE` and the model answers from world knowledge — no retrieval noise, no refusal; (2) the model retrieves, every passage gets dropped by `[IsRel]`, and the system refuses honestly. The unacceptable behaviour — confident hallucination with cited-but-irrelevant Rust passages — is what Self-RAG prevents.",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost breakdown per query",
                "",
                "A reflection pipeline does more LLM work than vanilla RAG. Quantify it so you know what you are spending. We count the calls in a single end-to-end answer.",
            ],
            code=[
                "from cookbook import _cache",
                "before = _cache.stats()['entries']",
                "_ = answer_question('Walk through how lifetimes work in a function that returns a reference to its longest argument.')",
                "after = _cache.stats()['entries']",
                "calls = after - before",
                "print(f'New cache entries this query: {calls}')",
                "print('Rough breakdown:')",
                "print('  1   needs_retrieval decision')",
                "print('  1   query embedding')",
                "print('  k   per-passage IsRel checks  (k=8 -> 8)')",
                "print('  1   final generation')",
                "print('  1   IsSup check')",
                "print(f'  ~{1+1+8+1+1} = 12 calls in the worst case')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a representative Rust question, so a reader sees the full pipeline output.",
    ],
    run_code=[
        "q = 'When should I prefer Arc over Rc, and what guarantees do I lose if I switch?'",
        "ans, ctxs = answer_question(q)",
        "print('=== Self-RAG answer ===')",
        "print(ans)",
        "print()",
        "print(f'(used {len(ctxs)} passages)')",
    ],

    comparison_md=[
        "Show vanilla baseline vs Self-RAG on the same question. The baseline tends to retrieve more aggressively and stuff lower-quality passages; Self-RAG filters them out. Sometimes the answers are similar; sometimes the baseline hallucinates where Self-RAG refuses or stays grounded.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'When should I prefer Arc over Rc, and what guarantees do I lose if I switch?'",
        "base = vanilla_pipeline(q, corpus='rust-book', top_k=5)",
        "ours_ans, ours_ctxs = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'n_contexts': len(base.contexts), 'preview': base.answer[:160]},",
        "    {'pipeline': 'self-rag', 'n_contexts': len(ours_ctxs), 'preview': ours_ans[:160]},",
        "])",
    ],

    tuning_md=[
        "Five knobs, ranked by impact:",
        "",
        "1. **The `[Retrieve?]` prompt.** Decides which queries skip retrieval entirely. Loosen the prompt to bias toward retrieval (safer for unknown domains) or tighten it to save tokens (better when your corpus is narrow and questions usually fall outside).",
        "2. **Initial `k` for retrieval.** With per-passage filtering downstream, you can crank `k` up. We use 8; production systems often go to 12–20.",
        "3. **The `[IsRel]` prompt strictness.** Aggressive filters cut faithfulness errors but raise refusal rate. Tune by measuring both on a held-out eval set.",
        "4. **Use a cheaper model for reflection.** The reflection calls do not need your best model. Many production systems use a small fast model (Qwen-7B, GPT-4o-mini, Groq Llama-3.3-70b-instant) for reflection and reserve the bigger model for the final generation.",
        "5. **Reflection caching.** Reflection answers depend only on (question, passage). With the cookbook's disk cache on, repeated queries are free; in production, cache keys by hash of `(question, passage)` for the same payoff.",
    ],

    discussion_md=[
        "Three places Self-RAG falls short:",
        "",
        "- **Latency.** 8 reflection calls + 1 generation + 1 support check means 10 calls per query. Even at 200 ms each, that is 2 seconds. Recipe 27 (Speculative RAG) is the right answer if you need both reflection and speed.",
        "- **Drift across calls.** The reflection model and the generation model do not always agree. If you generate with Llama-3.3 and reflect with a smaller model, you will see cases where the reflection rejects passages the generator could have used well. Match the reflection model to the generation model when this matters.",
        "- **Subtle off-topic.** A passage that mentions the right keywords but discusses a different sub-topic often slips through `[IsRel]`. The reflection is yes/no; sometimes the right answer is \"partially useful, use for context only\". Recipe 22 (cross-encoder rerank) is a better tool when the corpus has lots of near-misses.",
        "",
        "Self-RAG remains the cheapest way to add basic safety to a RAG system. Stack it with CRAG (Recipe 25) for the case where every passage fails the filter; stack it with Adaptive-RAG (Recipe 26) for cheap query routing.",
    ],
)


# =============================================================================
# CRAG — Corrective Retrieval with web fallback
# =============================================================================
CRAG = Recipe(
    path="recipes/06-adaptive-and-agentic/corrective-retrieval-with-web-fallback.ipynb",
    title="Corrective Retrieval — Catching Bad Hits, Falling Back to Web",
    category="adaptive-and-agentic",
    corpus_filter="arxiv-mamba",

    theory_problem=[
        "Self-RAG decides per passage. CRAG asks a different question: when retrieval *as a whole* failed, what do we do? The vanilla pipeline answers from bad context anyway. CRAG triggers a fallback — usually a web search — and incorporates the new evidence.",
        "Concretely: an evaluator scores the retrieved passages. If the best is above a threshold, use them. If the worst is below another threshold, ignore retrieval entirely and search the web. If they sit in between, decompose and combine. The three-way split lets the system stay grounded in the corpus when it can and reach beyond when it must.",
    ],
    theory_origin=[
        "CRAG was published in early 2024 by Yan et al. (Tencent + Tsinghua). Their main contribution was a lightweight T5-based evaluator that scored retrieval quality and routed to one of three branches: correct (use), incorrect (web fallback), ambiguous (decompose). They showed 4–10 point improvements on PopQA and Biography depending on the base model.",
        "By mid-2025 the pattern had spread widely. Production systems use it with Tavily, Brave Search, or Exa as the web fallback; the prompt-engineered variant replaces the T5 evaluator with a small LLM. We use the prompt-engineered variant and mock the web fallback so the notebook runs deterministically.",
    ],
    theory_landscape=[
        "CRAG sits between Self-RAG and Agentic RAG in the decision hierarchy:",
        "",
        "- **Self-RAG (Recipe 24)** decides yes/no per passage.",
        "- **CRAG (this recipe)** decides good/borderline/bad over the whole retrieval, then routes.",
        "- **Agentic RAG (Recipe 28)** lets the model loop — retrieve, critique, re-query, retry. CRAG is one step of an agentic loop; LangGraph wraps the loop.",
        "",
        "All three can stack. Many production systems run Adaptive-RAG (Recipe 26) to classify the question, then CRAG to check whether retrieval covered it, then Self-RAG to filter the surviving passages. The cost is real; the quality lift on broad domains is real too.",
    ],
    theory_when_to_use=[
        "Use CRAG when your corpus is good but incomplete. The classic example: a customer support agent whose retrieval covers 80 % of common questions but misses recent product changes. CRAG keeps the 80 % grounded in the corpus and reaches the web for the 20 %.",
        "Skip it when web access is not allowed (regulated industries, air-gapped deployments). The structure still works — you can fall back to a different internal corpus, or to an LLM with explicit \"I do not know\" instructions — but the value drops sharply.",
        "Skip it also when retrieval recall is the bottleneck rather than retrieval correctness. If you cannot find anything good in the corpus because retrieval is broken (wrong embedder, wrong chunks), fix the retrieval before adding CRAG. CRAG is not a substitute for working retrieval.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**Two thresholds, not one.** A single \"is the retrieval good\" classifier loses information. The good/borderline/bad three-way split — high threshold for \"trust completely\", low threshold for \"abandon\" — degrades gracefully into the ambiguous middle.",
        "",
        "**Web search is a different distribution.** Web results are a fresh-but-noisy distribution. A confident integration prompt (\"use these web snippets, do not speculate\") matters more than for in-corpus results because the model is more tempted to over-summarise web noise.",
        "",
        "**The evaluator is cheap.** It does not need to be smart, only consistent. A small fast model scoring \"how relevant is this passage to this question on a 0-1 scale\" is fine. The fine-tuned T5 in the original paper is overkill for most teams.",
    ],

    architecture_mermaid="""
flowchart TB
  Q[User question] --> R[Retrieve top-k]
  R --> E[Evaluator<br/>score each passage]
  E --> S{Best score?}
  S -->|high<br/>>= 0.7| GOOD[Use corpus only]
  S -->|low<br/>< 0.3| BAD[Web search<br/>only]
  S -->|middle| MIX[Use corpus<br/>+ web snippets]
  GOOD --> GEN[LLM answer]
  BAD --> GEN
  MIX --> GEN
  GEN --> OUT[Return answer]
""",

    references=[
        Reference(
            title="CRAG: Corrective Retrieval Augmented Generation",
            url="https://arxiv.org/abs/2401.15884",
            kind="paper",
            note="Yan et al., 2024. The paper that introduced the pattern.",
        ),
        Reference(
            title="LangGraph CRAG tutorial",
            url="https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/",
            kind="docs",
            note="Reference implementation as a stateful graph.",
        ),
        Reference(
            title="Tavily Search API",
            url="https://tavily.com/docs",
            kind="docs",
            note="The web-search API most CRAG implementations target.",
        ),
        Reference(
            title="Self-RAG paper (cousin technique)",
            url="https://arxiv.org/abs/2310.11511",
            kind="paper",
            note="Per-passage filtering; covered in Recipe 24.",
        ),
        Reference(
            title="Adaptive-RAG paper (cousin technique)",
            url="https://arxiv.org/abs/2403.14403",
            kind="paper",
            note="Pre-retrieval routing; covered in Recipe 26.",
        ),
        Reference(
            title="Exa Search API",
            url="https://docs.exa.ai/",
            kind="docs",
            note="Alternative web-search API often used in CRAG.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Index the corpus",
                "",
                "We use the arXiv Mamba survey. It is a long technical paper; we expect retrieval to be good for questions inside the paper and bad for questions adjacent to it (about earlier S4 models, about non-SSM techniques the paper does not cover). That mix is what CRAG exists to handle.",
            ],
            code=[
                "from cookbook.corpora import load_arxiv_mamba",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_arxiv_mamba())",
                "chunks = sentence_window(docs, sentences_per_chunk=4, overlap=1)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('crag', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Build the evaluator",
                "",
                "For each retrieved passage, ask the model how relevant it is on a 0.0–1.0 scale. We constrain the output to just a number so parsing is easy. The original CRAG paper used a fine-tuned T5; we use the base chat model and let prompt structure carry the weight.",
            ],
            code=[
                "import re",
                "",
                "EVAL_PROMPT = (",
                "    'Score how well the passage helps answer the question, on a scale from 0.0 (no help) to 1.0 (fully answers). '",
                "    'Reply with just the number, nothing else.\\n\\n'",
                "    'Question: {q}\\n\\n'",
                "    'Passage: {p}\\n\\n'",
                "    'Score:'",
                ")",
                "",
                "def score_passage(question: str, passage: str) -> float:",
                "    raw = client.chat(EVAL_PROMPT.format(q=question, p=passage[:800]))",
                "    m = re.search(r'[01](?:\\.\\d+)?', raw)",
                "    return float(m.group()) if m else 0.0",
                "",
                "# quick spot check",
                "score_passage('What is selective scan?', 'Selective scan makes the SSM parameters input-dependent.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 3 — Define a mock web search",
                "",
                "Real CRAG calls Tavily, Brave, or Exa. To keep the notebook deterministic and free, we mock the web step: a function that returns a one-line \"web snippet\" containing the question itself. In production, replace this with whatever search API your stack uses.",
            ],
            code=[
                "def web_search_stub(question: str) -> list[str]:",
                "    return [",
                "        f'(simulated web snippet for: {question[:120]})',",
                "        '(In production, this is where Tavily/Brave/Exa results land.)',",
                "    ]",
                "",
                "print(web_search_stub('What is selective scan?'))",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Route on the evaluator's verdict",
                "",
                "Three-way branch. We pick thresholds at 0.7 for \"good\" and 0.3 for \"bad\". Anything in between gets the hybrid treatment — keep the best in-corpus passages, decompose the question, augment with web evidence.",
            ],
            code=[
                "def crag(question: str, k: int = 5):",
                "    qv = client.embed([question])[0]",
                "    hits = store.search(qv, top_k=k)",
                "    scored = [(h, score_passage(question, h.text)) for h in hits]",
                "    best = max(s for _, s in scored)",
                "    worst = min(s for _, s in scored)",
                "",
                "    if best >= 0.7:",
                "        branch = 'GOOD'",
                "        contexts = [h.text for h, s in scored if s >= 0.5]",
                "    elif worst < 0.3 and best < 0.5:",
                "        branch = 'BAD (web fallback)'",
                "        contexts = web_search_stub(question)",
                "    else:",
                "        branch = 'AMBIGUOUS (corpus + web)'",
                "        contexts = [h.text for h, s in scored[:3]] + web_search_stub(question)",
                "    return branch, contexts, scored",
                "",
                "branch, contexts, scored = crag('What is selective scan and why does it matter?')",
                "print(f'Branch: {branch}')",
                "print()",
                "for h, s in scored:",
                "    print(f'  score={s:.2f}  {h.text[:120]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Generate from the routed context",
                "",
                "Once the branch is chosen, generation is straightforward: stuff the selected contexts and ask the model. We make the prompt branch-aware so the model knows whether it is reading from a trusted corpus, web snippets, or both.",
            ],
            code=[
                "def generate(question: str, branch: str, contexts: list[str]) -> str:",
                "    preamble = {",
                "        'GOOD': 'Use these passages from a trusted technical paper.',",
                "        'BAD (web fallback)': 'The corpus did not contain the answer. Use these web snippets carefully.',",
                "        'AMBIGUOUS (corpus + web)': 'Use the corpus passages where reliable; the web snippets fill in gaps.',",
                "    }[branch]",
                "    rendered = '\\n\\n'.join(contexts)",
                "    return client.chat(",
                "        f'{preamble}\\n\\n{rendered}\\n\\nQuestion: {question}\\nAnswer:'",
                "    )",
                "",
                "answer = generate('What is selective scan and why does it matter?', branch, contexts)",
                "print(answer)",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 6 — Wrap it as `answer_question`",
                "",
                "The cookbook contract: every recipe exposes an `answer_question(q) -> (answer, contexts)`. The eval cell at the bottom calls it the same way for every recipe.",
            ],
            code=[
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    branch, contexts, _ = crag(question)",
                "    answer = generate(question, branch, contexts)",
                "    return answer, contexts",
                "",
                "ans, ctxs = answer_question('How does Mamba achieve linear-time sequence modelling?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — branch distribution across a small battery",
                "",
                "Run several questions and count how many land in each branch. Healthy CRAG over the Mamba corpus shows most in-paper questions hit `GOOD`, adjacent questions hit `AMBIGUOUS`, and totally unrelated questions hit `BAD`.",
            ],
            code=[
                "battery = [",
                "    'What is selective scan and why does it matter?',",
                "    'How is Mamba related to S4?',",
                "    'What did the original Transformer paper introduce?',",
                "    'How does Mamba compare to RWKV on long contexts?',",
                "    'Who designed the Apollo 11 mission?',",
                "]",
                "for q in battery:",
                "    branch, _, _ = crag(q)",
                "    print(f'  {branch:30s}  {q}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — score distribution per question",
                "",
                "Look at the raw evaluator scores across k=8 for one question. The shape of the distribution tells you whether the evaluator is confident (sharp peak) or uncertain (flat).",
            ],
            code=[
                "import matplotlib.pyplot as plt",
                "",
                "q = 'What is selective scan and why does it matter?'",
                "qv = client.embed([q])[0]",
                "hits = store.search(qv, top_k=8)",
                "scores = [score_passage(q, h.text) for h in hits]",
                "fig, ax = plt.subplots(figsize=(6, 2.8))",
                "ax.bar(range(1, len(scores) + 1), scores)",
                "ax.set_xlabel('Rank')",
                "ax.set_ylabel('Evaluator score')",
                "ax.set_title(f'Per-passage scores: {q[:60]}')",
                "ax.set_ylim(0, 1)",
                "ax.grid(alpha=0.3)",
                "plt.tight_layout()",
                "plt.show()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — out-of-corpus fallback behaviour",
                "",
                "Force the BAD branch with a clearly out-of-corpus question. Confirm the system falls back to web evidence and the answer is appropriately hedged.",
            ],
            code=[
                "q = 'Who painted the ceiling of the Sistine Chapel and in what years?'",
                "ans, ctx = answer_question(q)",
                "print(ans)",
                "print()",
                "print(f'Contexts used (count={len(ctx)}):')",
                "for c in ctx:",
                "    print(f'  - {c[:100]}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does the ambiguous branch look like?",
                "",
                "An adjacent question that should land in AMBIGUOUS. The corpus has partial coverage (Mamba discusses related work), the answer benefits from both corpus context and web augmentation.",
            ],
            code=[
                "q = 'How does Mamba compare to the original Transformer architecture in terms of memory complexity?'",
                "branch, ctx, scored = crag(q)",
                "print(f'Branch: {branch}')",
                "print(f'Scores: {[round(s, 2) for _, s in scored]}')",
                "ans = generate(q, branch, ctx)",
                "print()",
                "print(ans)",
            ],
        ),
    ],

    run_md=[
        "Representative end-to-end run.",
    ],
    run_code=[
        "q = 'In one paragraph, explain why state-space models can scale to long contexts more cheaply than attention.'",
        "ans, ctxs = answer_question(q)",
        "print('=== CRAG answer ===')",
        "print(ans)",
        "print()",
        "print(f'(used {len(ctxs)} contexts)')",
    ],

    comparison_md=[
        "On the same question, run vanilla baseline and CRAG side by side. The interesting case is a borderline question where vanilla retrieves three weak passages and answers from them; CRAG should detect the weakness and either add web evidence or refuse.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'How does Mamba compare to the original Transformer architecture in terms of memory complexity?'",
        "base = vanilla_pipeline(q, corpus='arxiv-mamba', top_k=5)",
        "ours_ans, ours_ctx = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'crag', 'preview': ours_ans[:160]},",
        "])",
    ],

    tuning_md=[
        "Four knobs:",
        "",
        "1. **Threshold for `GOOD`.** Default 0.7. Lower it to trust corpus more aggressively (cheaper, less web fallback). Raise it to be more cautious (more web fallback, higher cost).",
        "2. **Threshold for `BAD`.** Default 0.3. Lower it to never fully abandon the corpus. Raise it to fall back to web more eagerly.",
        "3. **Evaluator model.** A small fast model is fine. The original T5 evaluator runs at < 50 ms per pair on a single GPU; in cloud terms a `gpt-4o-mini` or `Qwen-2.5-7B` evaluator is comparable.",
        "4. **Web fallback provider.** Tavily for cleaned snippets, Brave for raw breadth, Exa for semantic-search-first behaviour. Each has different snippet length and noise characteristics; tune your `generate` prompt for the one you pick.",
    ],

    discussion_md=[
        "Three places CRAG falls down:",
        "",
        "- **Latency.** Evaluator runs over every retrieved passage. With k=10 and a 200 ms evaluator, that is 2 seconds before generation starts. Recipe 27 (Speculative RAG) is the right move when you need both routing and speed.",
        "- **Web noise.** Web results are uneven; a single bad snippet can derail the answer. Always pair web fallback with a guardrail (Recipe 40) for production.",
        "- **Adversarial corpus.** If the evaluator is gameable — e.g. an injected passage repeats the question text — it can falsely score high. The trained T5 in the original paper is harder to game than a prompt-based scorer.",
        "",
        "CRAG composes well: stack it on top of Self-RAG for per-passage filtering, under Adaptive-RAG for cheap pre-routing, and gate the web fallback behind a small budget so a query cannot burn unlimited tokens chasing fallbacks.",
    ],
)


# =============================================================================
# Adaptive-RAG — classify questions, dispatch to the right strategy
# =============================================================================
ADAPTIVE = Recipe(
    path="recipes/06-adaptive-and-agentic/adaptive-routing-by-question-class.ipynb",
    title="Adaptive-RAG — Classify the Question, Dispatch the Strategy",
    category="adaptive-and-agentic",
    corpus_filter="wikipedia-superconductors",

    theory_problem=[
        "RAG systems usually treat every query the same: retrieve five chunks, stuff them, answer. That is wasteful for trivia the model already knows, insufficient for multi-hop questions, and miscalibrated for the wide range of complexities real users send. Adaptive-RAG classifies the question first and dispatches to the right retrieval strategy.",
        "The published technique (Jeong et al., NAACL 2024) uses three classes: `no retrieval` (model answers from world knowledge), `single-step retrieval` (vanilla RAG), `multi-step retrieval` (iterative — retrieve, re-query based on what we found, retrieve again). The classifier picks one; the pipeline runs accordingly. Total cost is *lower* than uniform RAG because simple questions skip retrieval entirely.",
    ],
    theory_origin=[
        "Adaptive-RAG was published at NAACL 2024 by Soyeong Jeong and colleagues at KAIST. Their classifier was a fine-tuned DistilBERT-class model trained on a labelled question complexity dataset. The headline result: 22% lower latency at similar quality, with selective gains on multi-hop questions (HotpotQA, MuSiQue) thanks to the multi-step branch.",
        "By late 2025 the prompt-engineered variant — ask the chat model to classify — had become dominant in production for the same reason as Self-RAG: it does not require a separately trained model and works on any base LLM. We use the prompt variant here.",
    ],
    theory_landscape=[
        "Adaptive-RAG sits before retrieval. Self-RAG and CRAG sit after retrieval:",
        "",
        "- **Adaptive-RAG (this recipe)** decides *strategy* before retrieving. Cheapest, runs once per query.",
        "- **Self-RAG (Recipe 24)** filters each retrieved passage. Runs k times.",
        "- **CRAG (Recipe 25)** scores retrieval as a whole, triggers fallback. Runs k+1 times.",
        "",
        "All three compose. Production systems route with Adaptive-RAG, retrieve, then run Self-RAG over the survivors, then CRAG-style fallback if everything fails. The pre-routing in Adaptive-RAG is the single biggest cost saver — most queries skip the expensive post-retrieval reflection entirely.",
    ],
    theory_when_to_use=[
        "Use Adaptive-RAG when your traffic is heterogeneous. Public-facing assistants, multi-domain agents, anywhere users send a mix of \"what is X\" trivia, \"how do I do Y\" how-tos, and \"compare X and Y across documents\" multi-hop questions. The classifier turns those into different code paths at low cost.",
        "Skip it when traffic is uniform. A single-domain Q&A bot with consistent question shapes does not benefit — every query hits the same branch.",
        "Skip it also when the classifier is unreliable on your domain. If your classifier mis-routes 30 % of queries to the wrong branch, you are worse than vanilla RAG. Evaluate the classifier on a labelled slice before shipping.",
    ],
    theory_intuition=[
        "Three intuitions:",
        "",
        "**Cheapest decisions first.** Routing on a single LLM call (a few hundred tokens) is far cheaper than retrieving, reranking, generating, then realising the question was trivia. Pre-routing pays back the most when traffic is mixed.",
        "",
        "**Multi-hop is a structural problem.** Some questions genuinely need iterative retrieval — retrieve, see the answer references X, retrieve about X, combine. A single-shot retrieval on those questions retrieves passages adjacent to the answer but missing the bridging fact. The multi-hop branch is what makes Adaptive-RAG genuinely lift quality on HotpotQA-class corpora.",
        "",
        "**Three classes is the sweet spot.** Two classes (retrieve / don't) is too coarse for production. Four+ classes invite confusion. The Adaptive-RAG paper landed on three after a lot of measurement.",
    ],

    architecture_mermaid="""
flowchart TB
  Q[User question] --> CLS{Classifier:<br/>complexity?}
  CLS -->|NO_RETRIEVAL| A[Direct LLM<br/>answer]
  CLS -->|SIMPLE| SR[Single-shot<br/>retrieval]
  CLS -->|MULTI_HOP| MR[Iterative<br/>retrieve-and-refine]
  SR --> SG[LLM answer]
  MR --> MR2{Done?}
  MR2 -->|no| MR
  MR2 -->|yes| MG[Compose answer]
  A --> OUT[Return]
  SG --> OUT
  MG --> OUT
""",

    references=[
        Reference(
            title="Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity",
            url="https://arxiv.org/abs/2403.14403",
            kind="paper",
            note="Jeong et al., NAACL 2024.",
        ),
        Reference(
            title="LangGraph Adaptive RAG tutorial",
            url="https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/",
            kind="docs",
            note="Reference implementation as a stateful graph.",
        ),
        Reference(
            title="HotpotQA dataset",
            url="https://hotpotqa.github.io/",
            kind="paper",
            note="Standard multi-hop QA benchmark Adaptive-RAG targets.",
        ),
        Reference(
            title="MuSiQue multi-hop QA",
            url="https://github.com/StonyBrookNLP/musique",
            kind="repo",
            note="Harder multi-hop benchmark with controllable hop count.",
        ),
        Reference(
            title="Self-RAG (cousin technique)",
            url="https://arxiv.org/abs/2310.11511",
            kind="paper",
            note="Recipe 24 — per-passage filtering.",
        ),
        Reference(
            title="CRAG (cousin technique)",
            url="https://arxiv.org/abs/2401.15884",
            kind="paper",
            note="Recipe 25 — post-retrieval scoring and fallback.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Index the corpus",
                "",
                "We use the Wikipedia superconductors subset. It is many short documents, perfect for multi-hop questions that span pages.",
            ],
            code=[
                "from cookbook.corpora import load_wikipedia_superconductors",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_wikipedia_superconductors())",
                "chunks = sentence_window(docs, sentences_per_chunk=4, overlap=1)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('adapt', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Build the classifier",
                "",
                "One prompt, three labels. We constrain the output and parse the first token. Keeping the prompt short matters — this call runs on every query.",
            ],
            code=[
                "CLASSIFY = (",
                "    'Classify the following question into exactly one of three classes:\\n'",
                "    '  NO_RETRIEVAL  — trivia or common knowledge the model already knows.\\n'",
                "    '  SIMPLE        — a single look-up in a domain knowledge base would suffice.\\n'",
                "    '  MULTI_HOP     — needs combining facts from multiple documents to answer.\\n\\n'",
                "    'Reply with exactly one word from NO_RETRIEVAL, SIMPLE, or MULTI_HOP.\\n\\n'",
                "    'Question: {q}\\nAnswer:'",
                ")",
                "",
                "def classify(question: str) -> str:",
                "    out = client.chat(CLASSIFY.format(q=question)).strip().upper()",
                "    for label in ('NO_RETRIEVAL', 'MULTI_HOP', 'SIMPLE'):",
                "        if label in out:",
                "            return label",
                "    return 'SIMPLE'   # safe default",
                "",
                "probes = [",
                "    'What year did World War II end?',",
                "    'What is the Meissner effect?',",
                "    'How did the discovery of YBCO and the later development of high-pressure hydride superconductors together change what counts as room-temperature superconductivity?',",
                "]",
                "for p in probes:",
                "    print(f'  {classify(p):14s}  {p}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Build the three branches",
                "",
                "Each branch is one function. The `NO_RETRIEVAL` branch is just a direct chat call. `SIMPLE` is vanilla RAG. `MULTI_HOP` is the interesting one — we let the model propose a follow-up question after seeing the first retrieval, then retrieve again.",
            ],
            code=[
                "def branch_no_retrieval(question: str) -> tuple[str, list[str]]:",
                "    answer = client.chat(f'Answer concisely from your own knowledge: {question}')",
                "    return answer, []",
                "",
                "def branch_simple(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    hits = store.search(qv, top_k=k)",
                "    contexts = [h.text for h in hits]",
                "    prompt = 'Use these passages:\\n' + '\\n\\n'.join(contexts) + f'\\n\\nQuestion: {question}\\nAnswer:'",
                "    return client.chat(prompt), contexts",
                "",
                "def branch_multi_hop(question: str, max_hops: int = 3) -> tuple[str, list[str]]:",
                "    history, contexts = [], []",
                "    follow_up = question",
                "    for hop in range(max_hops):",
                "        partial, ctx = branch_simple(follow_up, k=4)",
                "        history.append(partial)",
                "        contexts.extend(ctx)",
                "        rewrite = client.chat(",
                "            'Given the partial answers below, what specific FOLLOW-UP question would close the remaining gap? '",
                "            'Reply with just the question, or the single word DONE if complete.\\n\\n'",
                "            + '\\n'.join(history) + f'\\n\\nOriginal: {question}'",
                "        ).strip()",
                "        if rewrite.upper().startswith('DONE') or len(rewrite) < 5:",
                "            break",
                "        follow_up = rewrite",
                "    final = client.chat(",
                "        'Combine the partial answers below into a single coherent response.\\n\\n'",
                "        + '\\n\\n'.join(history) + f'\\n\\nOriginal question: {question}\\nFinal answer:'",
                "    )",
                "    return final, contexts",
                "",
                "print('Three branches defined.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — The dispatcher",
                "",
                "Tie classifier output to branch. This is the entire \"adaptive\" part. Everything else is vanilla RAG primitives we built earlier.",
            ],
            code=[
                "def adaptive(question: str) -> tuple[str, list[str], str]:",
                "    cls = classify(question)",
                "    if cls == 'NO_RETRIEVAL':",
                "        ans, ctx = branch_no_retrieval(question)",
                "    elif cls == 'MULTI_HOP':",
                "        ans, ctx = branch_multi_hop(question)",
                "    else:",
                "        ans, ctx = branch_simple(question)",
                "    return ans, ctx, cls",
                "",
                "for q in [",
                "    'What is the boiling point of liquid nitrogen?',",
                "    'What is the Meissner effect?',",
                "    'How do the discoveries of YBCO and high-pressure hydride superconductors together change the practical meaning of \\\"room temperature\\\" superconductivity?',",
                "]:",
                "    ans, _, cls = adaptive(q)",
                "    print(f'[{cls}] {q}')",
                "    print(f'  -> {ans[:200]}')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — The cookbook contract: `answer_question`",
                "",
                "Same shape as every other recipe. Strips the classification label since the contract is `(answer, contexts)` only.",
            ],
            code=[
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    ans, ctx, _ = adaptive(question)",
                "    return ans, ctx",
                "",
                "ans, ctx = answer_question('Compare BCS theory with the standard explanation of high-temperature superconductivity.')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — classifier behaviour across a small battery",
                "",
                "Run several questions and see how they classify. The classifier is the foundation of everything; if it mis-routes, every branch breaks.",
            ],
            code=[
                "battery = [",
                "    'What is the freezing point of water in Celsius?',",
                "    'Define superconductivity in one sentence.',",
                "    'How does the Meissner effect distinguish a superconductor from a perfect conductor?',",
                "    'What materials connect the discovery of high-temperature superconductivity to current applications in MRI?',",
                "    'In what decade was BCS theory proposed and how did it relate to earlier London-equation work?',",
                "]",
                "for q in battery:",
                "    print(f'  {classify(q):14s}  {q}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does the multi-hop trace look like?",
                "",
                "Manually drive the multi-hop branch on a question that needs it and print every hop. Useful for debugging when the final answer is wrong — usually the failure is in the follow-up question, not the retrieval.",
            ],
            code=[
                "q = 'What materials connect the discovery of high-temperature superconductivity to applications in MRI and particle accelerators?'",
                "history = []",
                "follow_up = q",
                "for hop in range(3):",
                "    partial, _ = branch_simple(follow_up, k=4)",
                "    history.append((follow_up, partial))",
                "    next_q = client.chat(",
                "        'Given the partial answers, what FOLLOW-UP question best closes the gap? Or reply DONE.\\n\\n'",
                "        + '\\n'.join(p for _, p in history) + f'\\n\\nOriginal: {q}'",
                "    ).strip()",
                "    if next_q.upper().startswith('DONE'):",
                "        break",
                "    follow_up = next_q",
                "for i, (qq, ans) in enumerate(history):",
                "    print(f'Hop {i}: {qq[:80]}')",
                "    print(f'    -> {ans[:200]}')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — token budget per branch",
                "",
                "How many LLM calls does each branch make? Cost-aware design.",
            ],
            code=[
                "from cookbook import _cache",
                "",
                "def measure(q: str):",
                "    before = _cache.stats()['entries']",
                "    _ = adaptive(q)",
                "    after = _cache.stats()['entries']",
                "    return after - before",
                "",
                "for q in [",
                "    'What is the freezing point of water?',                                    # NO_RETRIEVAL",
                "    'What is the Meissner effect?',                                            # SIMPLE",
                "    \"How did Bednorz and Mueller's discovery, combined with later YBCO work, change practical superconductivity?\",  # MULTI_HOP",
                "]:",
                "    calls = measure(q)",
                "    print(f'  {classify(q):14s}  {calls:2d} new calls  {q[:60]}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — sensitivity to the classifier prompt",
                "",
                "Slightly different classifier prompts produce different routing. Print the same question against two prompt variants and look at the disagreement.",
            ],
            code=[
                "alt_prompt = (",
                "    'Is the following question best answered from world knowledge alone (skip retrieval), '",
                "    'from a single corpus lookup, or by combining multiple sources?\\n'",
                "    'Reply with one of NO_RETRIEVAL, SIMPLE, or MULTI_HOP.\\n\\nQuestion: {q}\\nAnswer:'",
                ")",
                "",
                "def classify_alt(question: str) -> str:",
                "    out = client.chat(alt_prompt.format(q=question)).strip().upper()",
                "    for label in ('NO_RETRIEVAL', 'MULTI_HOP', 'SIMPLE'):",
                "        if label in out:",
                "            return label",
                "    return 'SIMPLE'",
                "",
                "for q in [",
                "    'What is the boiling point of liquid nitrogen?',",
                "    'How is BCS theory connected to superconducting magnets used in MRI?',",
                "]:",
                "    print(f'  default={classify(q):14s}  alt={classify_alt(q):14s}  {q[:60]}')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a representative multi-hop question.",
    ],
    run_code=[
        "q = 'How did the discovery of YBCO and the more recent high-pressure hydride superconductors together change the practical meaning of \\\"room-temperature\\\" superconductivity?'",
        "ans, ctxs = answer_question(q)",
        "print('=== Adaptive-RAG answer ===')",
        "print(ans)",
        "print()",
        "print(f'(used {len(ctxs)} contexts)')",
    ],

    comparison_md=[
        "On a multi-hop question, vanilla retrieval often returns chunks from the right area but misses the bridging fact. Adaptive-RAG's multi-hop branch retrieves twice and composes. Show both.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'How did the discovery of YBCO and the more recent high-pressure hydride superconductors together change the practical meaning of \\\"room-temperature\\\" superconductivity?'",
        "base = vanilla_pipeline(q, corpus='wikipedia-superconductors', top_k=5)",
        "ours_ans, ours_ctx = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:200], 'n_contexts': len(base.contexts)},",
        "    {'pipeline': 'adaptive', 'preview': ours_ans[:200], 'n_contexts': len(ours_ctx)},",
        "])",
    ],

    tuning_md=[
        "Five knobs:",
        "",
        "1. **Classifier prompt.** The whole pipeline pivots on it. Test on a labelled slice (50–100 questions you have hand-routed) before relying on it.",
        "2. **`max_hops` in the multi-hop branch.** Default 3. Higher means deeper multi-hop coverage but longer latency. Cap at 5 in production; runaway loops are a real failure mode.",
        "3. **Classifier model.** A small fast model is fine. The Adaptive-RAG paper uses DistilBERT; in cloud terms `gpt-4o-mini` or Groq Llama-3.3-70b-instant are cheap and fast.",
        "4. **`SIMPLE` branch `k`.** 5 is the cookbook default; tune for your corpus.",
        "5. **Multi-hop follow-up prompt.** The most fragile part of multi-hop is the rewrite. If the model proposes a too-general follow-up, the second retrieval returns the same chunks. Tighten the prompt to demand specificity.",
    ],

    discussion_md=[
        "Three places Adaptive-RAG falls down:",
        "",
        "- **Misclassification at the boundary.** A question that is `SIMPLE` but classifier calls `NO_RETRIEVAL` will be answered from world knowledge and silently miss the corpus's specific facts. Catch this with periodic eval-set sampling.",
        "- **Multi-hop runaway.** If the follow-up rewrite keeps proposing minor variations, the loop never returns. The hop cap and the `DONE` shortcut both matter.",
        "- **Latency tail.** Multi-hop queries take 3–5x longer than simple ones. If your latency SLO is tight, route multi-hop to a separate async path or refuse for synchronous use.",
        "",
        "Adaptive-RAG is the cheapest way to handle heterogeneous traffic. Stack it under Self-RAG (Recipe 24) for the `SIMPLE` branch, and CRAG (Recipe 25) for the `MULTI_HOP` branch's final answer-grounding step. The composition is what makes a production system feel polished.",
    ],
)


# =============================================================================
# Speculative RAG (drafter + verifier)
# =============================================================================
SPECULATIVE = Recipe(
    path="recipes/06-adaptive-and-agentic/speculative-rag-drafter-verifier.ipynb",
    title="Speculative RAG — Drafter and Verifier in Parallel",
    category="adaptive-and-agentic",
    corpus_filter="arxiv-mamba",

    theory_problem=[
        "Standard RAG runs one expensive LLM call per query. The model receives k chunks, reasons over them, and produces an answer. The reasoning is sequential, the cost is linear in answer length, and the latency is whatever the model takes — usually 1–3 seconds.",
        "Speculative RAG borrows the idea from speculative decoding: run several cheap drafter models in parallel, then have one expensive verifier pick the best draft. Each drafter sees a different retrieval subset, so their drafts are diverse. The verifier reads them together and picks. Latency drops because drafting is parallel; quality often improves because the verifier sees diverse evidence.",
    ],
    theory_origin=[
        "Speculative RAG was published at ICLR 2025 (Wang et al.). The paper demonstrated 50% latency reductions at comparable or better quality on TriviaQA, HotpotQA, and PubMedQA. The technique borrowed from speculative decoding (Stern et al., 2018; Leviathan et al., 2023) which uses a small drafter to predict tokens that a large verifier accepts or rejects, then lifted that pattern to the document-retrieval level.",
        "By 2026 the pattern has spread to anything where latency budgets are tight. Customer-support assistants, real-time chat agents, and voice interfaces all benefit from the parallel-draft structure. The technique pairs especially well with prompt caching — drafter prompts are very similar across queries, so cache hit rates are high.",
    ],
    theory_landscape=[
        "Three approaches to fast RAG you should know:",
        "",
        "- **Caching (Recipe 2 baseline).** Cache embeddings and LLM responses for repeat queries. Free first-line defence.",
        "- **Speculative RAG (this recipe).** Parallel cheap drafts, single verify. Cuts latency by parallelising the drafting step.",
        "- **Streaming generation.** Start emitting tokens before all chunks arrive. Reduces perceived latency.",
        "",
        "Speculative composes with caching and streaming — they attack different parts of the latency budget. A production stack often runs all three.",
    ],
    theory_when_to_use=[
        "Use speculative when you need fast RAG with comparable quality. The technique shines when the corpus has multiple plausible retrieval subsets — i.e., when reranking improves things and the right top-k isn't obvious.",
        "Skip it when you have only one obvious retrieval set. With k=3 and only 4 viable chunks, you can't run diverse drafters and the parallel structure wastes budget.",
        "Skip it when budget matters more than latency. Speculative makes N+1 LLM calls per query instead of 1, which is a meaningful multiplier at high volume.",
    ],
    theory_intuition=[
        "Three intuitions to carry:",
        "",
        "**Parallel drafts give the verifier choices.** With three drafters seeing different chunk subsets, the verifier sees three candidate answers and picks. Without speculative, the LLM would have to produce one answer and accept it.",
        "",
        "**The verifier only picks — it doesn't generate.** This keeps the expensive call short. The verifier prompt is just \"which of these is best?\".",
        "",
        "**Latency = max(drafters) + verify, not sum.** If drafters take 800 ms each and the verifier takes 200 ms, total wall time is 1 second — not 3 seconds.",
        "",
        "**Drafter diversity matters.** All drafters seeing the same chunks would produce the same draft and waste the parallel structure. The cookbook splits the candidate pool into N disjoint subsets, one per drafter, so each drafter sees genuinely different evidence and produces a genuinely different draft.",
    ],

    architecture_mermaid="""
flowchart TB
  Q[Query] --> R[Retrieve top-k*N]
  R --> S1[Drafter 1<br/>subset 1]
  R --> S2[Drafter 2<br/>subset 2]
  R --> S3[Drafter 3<br/>subset 3]
  S1 --> V[Verifier:<br/>pick best draft]
  S2 --> V
  S3 --> V
  V --> A[Final answer]
""",

    references=[
        Reference(
            title="Speculative RAG — Enhancing Retrieval Augmented Generation through Drafting (Wang et al., 2025)",
            url="https://arxiv.org/abs/2407.08223",
            kind="paper",
            note="The ICLR 2025 paper.",
        ),
        Reference(
            title="Speculative Decoding (Leviathan et al., 2023)",
            url="https://arxiv.org/abs/2211.17192",
            kind="paper",
            note="The decoding-time precursor that inspired Speculative RAG.",
        ),
        Reference(
            title="LlamaIndex Speculative RAG reference",
            url="https://developers.llamaindex.ai/python/examples/cookbooks/speculative_rag/",
            kind="docs",
            note="Reference implementation.",
        ),
        Reference(
            title="Self-RAG (Recipe 24)",
            url="https://arxiv.org/abs/2310.11511",
            kind="paper",
            note="Cousin technique — sequential reflection rather than parallel drafting.",
        ),
        Reference(
            title="OpenAI prompt caching",
            url="https://platform.openai.com/docs/guides/prompt-caching",
            kind="docs",
            note="Composes well — the verifier sees similar inputs across queries.",
        ),
        Reference(
            title="Streaming generation cookbook (LiteLLM)",
            url="https://docs.litellm.ai/docs/completion/stream",
            kind="docs",
            note="Combine with speculative for additional latency wins.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build the index",
                "",
                "Standard Mamba paper setup.",
            ],
            code=[
                "from cookbook.corpora import load_arxiv_mamba",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_arxiv_mamba())",
                "chunks = sentence_window(docs, sentences_per_chunk=4)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('spec', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 2 — Configure drafter and verifier",
                "",
                "Drafter is a small fast model; verifier is the default chat model. In production you'd pick a smaller drafter (Qwen 7B) and reserve frontier for the verifier.",
            ],
            code=[
                "drafter = client",
                "verifier = client",
                "print(f'Drafter model:  {drafter.chat_model}')",
                "print(f'Verifier model: {verifier.chat_model}')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 3 — Split candidates into drafter subsets",
                "",
                "Retrieve k*N candidates, partition into N disjoint subsets. Each drafter sees one subset.",
            ],
            code=[
                "import math",
                "",
                "def split_subsets(hits, n_drafters: int):",
                "    return [hits[i::n_drafters] for i in range(n_drafters)]",
                "",
                "q = 'How does the parallel scan algorithm exploit GPU memory hierarchies?'",
                "qv = client.embed([q])[0]",
                "pool = store.search(qv, top_k=15)",
                "subsets = split_subsets(pool, n_drafters=3)",
                "for i, s in enumerate(subsets):",
                "    print(f'Subset {i+1}: {len(s)} chunks; first chunk preview: {s[0].text[:60]}...')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Draft in parallel",
                "",
                "We run N drafters concurrently. Each produces a candidate answer.",
            ],
            code=[
                "from concurrent.futures import ThreadPoolExecutor",
                "",
                "DRAFT_PROMPT = (",
                "    'Use the passages below to answer the question concisely.\\n'",
                "    'Passages:\\n{ctx}\\nQuestion: {q}\\nAnswer:'",
                ")",
                "",
                "def draft_one(s):",
                "    ctx = '\\n\\n'.join(h.text for h in s)",
                "    return drafter.chat(DRAFT_PROMPT.format(ctx=ctx, q=q))",
                "",
                "with ThreadPoolExecutor(max_workers=len(subsets)) as ex:",
                "    drafts = list(ex.map(draft_one, subsets))",
                "",
                "for i, d in enumerate(drafts):",
                "    print(f'Draft {i+1}: {d[:200]}...')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Verify and pick",
                "",
                "The verifier reads all N drafts and picks the best by index. We constrain output to just the index for cheap parsing.",
            ],
            code=[
                "rendered = '\\n\\n'.join(f'[{i}] {d}' for i, d in enumerate(drafts))",
                "pick_prompt = (",
                "    'Pick the best draft index given the question. Reply with just the index.\\n'",
                "    + rendered + f'\\nQuestion: {q}'",
                ")",
                "pick_raw = verifier.chat(pick_prompt)",
                "import re",
                "m = re.search(r'\\d+', pick_raw)",
                "idx = int(m.group()) if m else 0",
                "idx = max(0, min(idx, len(drafts) - 1))",
                "final = drafts[idx]",
                "print(f'Verifier picked draft {idx}.')",
                "print()",
                "print(final)",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 6 — Wrap as `answer_question`",
                "",
                "Cookbook contract.",
            ],
            code=[
                "def answer_question(question: str, n_drafters: int = 3, k: int = 5) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    pool = store.search(qv, top_k=k * n_drafters)",
                "    subsets = split_subsets(pool, n_drafters)",
                "    def draft_one_inner(s):",
                "        ctx = '\\n\\n'.join(h.text for h in s)",
                "        return drafter.chat(DRAFT_PROMPT.format(ctx=ctx, q=question))",
                "    with ThreadPoolExecutor(max_workers=n_drafters) as ex:",
                "        drafts = list(ex.map(draft_one_inner, subsets))",
                "    rendered = '\\n\\n'.join(f'[{i}] {d}' for i, d in enumerate(drafts))",
                "    pick_raw = verifier.chat(",
                "        'Pick the best draft index given the question. Reply with just the index.\\n'",
                "        + rendered + f'\\nQuestion: {question}'",
                "    )",
                "    m = re.search(r'\\d+', pick_raw)",
                "    idx = max(0, min(int(m.group()) if m else 0, n_drafters - 1))",
                "    return drafts[idx], [h.text for h in subsets[idx]]",
                "",
                "ans, _ = answer_question('Why is HiPPO initialization useful?')",
                "print(ans)",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — how often does the verifier pick non-default drafts?",
                "",
                "If the verifier always picks draft 0, speculative isn't doing work — drafts 1 and 2 are wasted.",
            ],
            code=[
                "from collections import Counter",
                "picks = Counter()",
                "for q in [",
                "    'What is selective scan?',",
                "    'How does Mamba differ from S4?',",
                "    'Why is parallel scan important?',",
                "    'When should I use a state-space model?',",
                "]:",
                "    qv = client.embed([q])[0]",
                "    pool = store.search(qv, top_k=15)",
                "    subsets = split_subsets(pool, n_drafters=3)",
                "    def d(s):",
                "        ctx = '\\n\\n'.join(h.text for h in s)",
                "        return drafter.chat(DRAFT_PROMPT.format(ctx=ctx, q=q))",
                "    with ThreadPoolExecutor(max_workers=3) as ex:",
                "        ds = list(ex.map(d, subsets))",
                "    rendered = '\\n\\n'.join(f'[{i}] {x}' for i, x in enumerate(ds))",
                "    p = verifier.chat(",
                "        'Pick the best draft index. Reply with just the index.\\n'",
                "        + rendered + f'\\nQuestion: {q}'",
                "    )",
                "    m = re.search(r'\\d+', p)",
                "    picks[int(m.group()) if m else 0] += 1",
                "print(f'Picks by index: {dict(picks)}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — draft diversity",
                "",
                "Drafters seeing different chunks should produce different drafts. We measure character-level overlap to confirm.",
            ],
            code=[
                "from difflib import SequenceMatcher",
                "",
                "q = 'How does Mamba scale linearly with sequence length?'",
                "qv = client.embed([q])[0]",
                "pool = store.search(qv, top_k=12)",
                "subsets = split_subsets(pool, n_drafters=3)",
                "def df(s):",
                "    ctx = '\\n\\n'.join(h.text for h in s)",
                "    return drafter.chat(DRAFT_PROMPT.format(ctx=ctx, q=q))",
                "with ThreadPoolExecutor(max_workers=3) as ex:",
                "    drafts = list(ex.map(df, subsets))",
                "for i, d in enumerate(drafts):",
                "    print(f'Draft {i+1} length: {len(d)} chars')",
                "for i in range(len(drafts)):",
                "    for j in range(i+1, len(drafts)):",
                "        ratio = SequenceMatcher(None, drafts[i], drafts[j]).ratio()",
                "        print(f'  similarity draft {i+1} vs {j+1}: {ratio:.2f}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — latency breakdown",
                "",
                "Drafts run in parallel; verify runs once. The bottleneck is the slowest drafter plus the verifier.",
            ],
            code=[
                "import time",
                "q = 'What is selective scan?'",
                "qv = client.embed([q])[0]",
                "pool = store.search(qv, top_k=15)",
                "subsets = split_subsets(pool, n_drafters=3)",
                "",
                "t0 = time.perf_counter()",
                "def d(s):",
                "    ctx = '\\n\\n'.join(h.text for h in s)",
                "    return drafter.chat(DRAFT_PROMPT.format(ctx=ctx, q=q))",
                "with ThreadPoolExecutor(max_workers=3) as ex:",
                "    drafts = list(ex.map(d, subsets))",
                "draft_ms = (time.perf_counter() - t0) * 1000",
                "",
                "t0 = time.perf_counter()",
                "rendered = '\\n\\n'.join(f'[{i}] {x}' for i, x in enumerate(drafts))",
                "_ = verifier.chat('Pick the best draft index. Reply with just the index.\\n' + rendered + f'\\nQuestion: {q}')",
                "verify_ms = (time.perf_counter() - t0) * 1000",
                "",
                "print(f'Parallel draft (wall-clock): {draft_ms:.1f} ms')",
                "print(f'Verify (single call):        {verify_ms:.1f} ms')",
                "print(f'Total:                        {draft_ms + verify_ms:.1f} ms')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost",
                "",
                "Speculative makes N+1 LLM calls per query. Track them.",
            ],
            code=[
                "from cookbook import _cache",
                "before = _cache.stats()['entries']",
                "_ = answer_question('What is HiPPO initialization?')",
                "after = _cache.stats()['entries']",
                "print(f'New cache entries: {after - before}')",
                "print('Rough breakdown for n_drafters=3:')",
                "print('  3   drafter LLM calls')",
                "print('  1   verifier LLM call')",
                "print('  1   query embed')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a representative question.",
    ],
    run_code=[
        "ans, _ = answer_question('What does Mamba claim about long-context efficiency, and what trade-off makes it possible?')",
        "print('=== Speculative answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla (single LLM call) vs speculative (N drafters + 1 verifier).",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What does Mamba claim about long-context efficiency?'",
        "base = vanilla_pipeline(q, corpus='arxiv-mamba', top_k=5)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'speculative', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Six knobs in priority order:",
        "",
        "1. **`n_drafters`.** 3 is the sweet spot. Higher pays diminishing returns and inflates cost without quality gain.",
        "2. **Drafter model.** Use a small fast model. A frontier drafter wastes budget without quality gain.",
        "3. **Verifier model.** A mid-tier model is enough. The verifier only picks, doesn't generate.",
        "4. **Subset size.** k chunks per drafter × N drafters = pool size. We use k=5 per drafter, 3 drafters → 15 total.",
        "5. **Streaming verifier.** When the verifier is streaming, you can return the picked draft as soon as the index token arrives.",
        "6. **Parallel-thread cap.** Cap the executor's max_workers to your provider's concurrency limit to avoid rate-limit errors.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Drafters too similar.** If all drafters produce similar drafts, the verifier has no real choice. Tune subset diversity.",
        "- **Verifier mis-pick.** A bad verifier can pick a worse draft. Always include the original retrieval in one drafter's subset so the worst case is vanilla quality.",
        "- **Cost on simple queries.** Speculative is wasted on questions where vanilla is already correct. Combine with adaptive routing (Recipe 26) so speculative only fires on hard queries.",
        "",
        "Compose with caching (always on), streaming (for the verifier output), and adaptive routing (to gate when speculative fires).",
    ],
)


# =============================================================================
# LangGraph Agentic RAG
# =============================================================================
LANGGRAPH = Recipe(
    path="recipes/06-adaptive-and-agentic/langgraph-agentic-rag.ipynb",
    title="LangGraph Agentic RAG — Stateful Retrieval Loops",
    category="adaptive-and-agentic",
    corpus_filter="rust-book",

    theory_problem=[
        "Real questions sometimes need multiple retrieval passes. The model gets some context, realizes it needs different chunks for the next step, and retrieves again. Implementing that as a chain of function calls works for two steps but breaks down at three — the state grows, the error handling balloons, and debugging becomes archaeology. By the time you're four nodes deep, your agent looks like a bowl of spaghetti with conditional branches you can't reason about.",
        "LangGraph treats the loop as a state machine. Each retrieval, critique, and re-retrieval is a node; the transitions are edges; the state carries question, contexts, answer, and loop counter. The graph runtime handles checkpointing, retries, and conditional routing. The result: stateful agents that are debuggable as graphs, not as stack traces, and that you can visualise as Mermaid diagrams to confirm the topology matches your intent.",
    ],
    theory_origin=[
        "LangGraph shipped from LangChain in early 2024 as a stateful graph runtime for LLM agents. It built on the foundations laid by the AgentExecutor and ReAct patterns from 2022-2023, but the explicit state-graph abstraction made multi-step retrieval and tool use much easier to author and reason about. The abstraction proved sticky — by 2026 most production agentic systems run on something graph-shaped.",
        "By 2026 LangGraph and LlamaIndex Workflows are the two canonical patterns for production agentic RAG. The cookbook uses LangGraph here because of its tight integration with the LangChain ecosystem and OpenTelemetry tracing, but the LlamaIndex Workflows variant has the same shape with a slightly different API.",
    ],
    theory_landscape=[
        "Three orchestration patterns to know:",
        "",
        "- **Linear chain.** Each step's output feeds the next. Cookbook's vanilla pipeline.",
        "- **DAG (LlamaIndex Workflows).** Branches and joins, no loops. Good for fan-out RAG and parallel retrieval.",
        "- **State graph with cycles (this recipe).** Loops, checkpointing, retries — what real agents need.",
        "",
        "Self-RAG, CRAG, and Adaptive-RAG can all be implemented as state graphs. LangGraph is the chassis; the recipes are the patterns. The cookbook factors them as separate recipes for clarity, but in production they often live as nodes inside one bigger graph.",
    ],
    theory_when_to_use=[
        "Use LangGraph when your RAG needs loops, retries, or non-trivial conditional logic. Multi-hop questions, tool-using agents, anything where the next step depends on intermediate results.",
        "Skip it when a linear chain is enough. Adding a graph runtime to vanilla RAG is over-engineering.",
        "Skip it when LangChain is not your stack. LlamaIndex Workflows offers a similar abstraction with different tradeoffs.",
    ],
    theory_intuition=[
        "Four intuitions to internalise:",
        "",
        "**State is explicit.** Every node reads and writes to a typed state dict. No hidden globals, no implicit shared state — what each node touches is visible in its signature.",
        "",
        "**Edges are conditional.** A node can route to different next nodes based on the state. That's how loops and branches work; conditional routing is what makes a graph more powerful than a chain.",
        "",
        "**Caps prevent runaway.** Always bound the loop count. An LLM that always asks for one more retrieval will loop forever, and you'll have an LLM bill to prove it.",
        "",
        "**Traces light up.** Every node is a span in Phoenix or LangSmith. Debugging is reading the trace, and the trace is structured the way the graph is structured.",
    ],

    architecture_mermaid="""
flowchart TB
  START([Start]) --> RET[Retrieve node]
  RET --> CRIT[Critique node]
  CRIT -->|need more| RET
  CRIT -->|done| END([End])
""",

    references=[
        Reference(
            title="LangGraph documentation",
            url="https://langchain-ai.github.io/langgraph/",
            kind="docs",
            note="The official LangGraph site.",
        ),
        Reference(
            title="LangGraph RAG tutorial",
            url="https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/",
            kind="docs",
            note="The agentic RAG reference implementation we adapt.",
        ),
        Reference(
            title="ReAct: Synergizing Reasoning and Acting in Language Models",
            url="https://arxiv.org/abs/2210.03629",
            kind="paper",
            note="The reasoning loop pattern LangGraph generalises.",
        ),
        Reference(
            title="LlamaIndex Workflows",
            url="https://developers.llamaindex.ai/python/framework/module_guides/workflow/",
            kind="docs",
            note="The DAG alternative.",
        ),
        Reference(
            title="Self-RAG (Recipe 24)",
            url="https://arxiv.org/abs/2310.11511",
            kind="paper",
            note="Often implemented as a LangGraph.",
        ),
        Reference(
            title="Anthropic agentic coding patterns",
            url="https://docs.anthropic.com/en/docs/build-with-claude/agents",
            kind="docs",
            note="Production agentic patterns.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build the index",
                "",
                "Standard Rust book setup.",
            ],
            code=[
                "from cookbook.corpora import load_rust_book",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_rust_book())",
                "chunks = sentence_window(docs, sentences_per_chunk=4)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('lg', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Define the state",
                "",
                "Typed dict. Each field is read and written by various nodes. LangGraph routes based on these values.",
            ],
            code=[
                "from typing import TypedDict",
                "",
                "class State(TypedDict):",
                "    question: str",
                "    query: str",
                "    contexts: list[str]",
                "    answer: str",
                "    loops: int",
                "print('State schema defined.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Define the retrieve and critique nodes",
                "",
                "Each node takes the state and returns a dict of updates. The runtime merges them into the state.",
            ],
            code=[
                "def retrieve(s: State) -> dict:",
                "    qv = client.embed([s['query']])[0]",
                "    hits = store.search(qv, top_k=5)",
                "    new_contexts = s.get('contexts', []) + [h.text for h in hits]",
                "    return {'contexts': new_contexts}",
                "",
                "def critique(s: State) -> dict:",
                "    raw = client.chat(",
                "        'Given the question and retrieved passages, reply with EITHER:\\n'",
                "        '  ANSWER: <final answer>\\n  NEED: <a more specific follow-up query>\\n'",
                "        f'Question: {s[\"question\"]}\\nPassages:\\n'",
                "        + '\\n\\n'.join(s['contexts'][-10:])",
                "    )",
                "    if raw.strip().upper().startswith('ANSWER'):",
                "        return {'answer': raw.split(':', 1)[1].strip()}",
                "    return {'query': raw.split(':', 1)[1].strip(), 'loops': s.get('loops', 0) + 1}",
                "",
                "print('Nodes defined.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Build the graph",
                "",
                "Start at retrieve, route through critique, loop back if needed, exit when we have an answer or hit the loop cap.",
            ],
            code=[
                "from langgraph.graph import StateGraph, END",
                "",
                "def decide(s: State) -> str:",
                "    if s.get('answer'):",
                "        return END",
                "    if s.get('loops', 0) >= 3:",
                "        return END",
                "    return 'retrieve'",
                "",
                "g = StateGraph(State)",
                "g.add_node('retrieve', retrieve)",
                "g.add_node('critique', critique)",
                "g.set_entry_point('retrieve')",
                "g.add_edge('retrieve', 'critique')",
                "g.add_conditional_edges('critique', decide, {'retrieve': 'retrieve', END: END})",
                "app = g.compile()",
                "print('Graph compiled.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Run the graph",
                "",
                "Invoke with initial state. The runtime walks the nodes until END.",
            ],
            code=[
                "result = app.invoke({",
                "    'question': 'When should I prefer Arc over Rc, and what about RefCell?',",
                "    'query': 'Rc Arc RefCell trade-offs',",
                "    'contexts': [],",
                "    'loops': 0,",
                "})",
                "print(f'Final answer:\\n{result.get(\"answer\", \"(no answer)\")[:400]}')",
                "print()",
                "print(f'Total loops: {result.get(\"loops\", 0)}')",
                "print(f'Contexts collected: {len(result.get(\"contexts\", []))}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 6 — Wrap as `answer_question`",
                "",
                "Cookbook contract.",
            ],
            code=[
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    result = app.invoke({",
                "        'question': question,",
                "        'query': question,",
                "        'contexts': [],",
                "        'loops': 0,",
                "    })",
                "    return result.get('answer', '(no answer)'), result.get('contexts', [])[-5:]",
                "",
                "ans, _ = answer_question('When should I prefer Arc over Rc, and what about RefCell?')",
                "print(ans[:400])",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — visualise the graph",
                "",
                "LangGraph can render the compiled graph as Mermaid for easy inspection.",
            ],
            code=[
                "try:",
                "    mermaid = app.get_graph().draw_mermaid()",
                "    print(mermaid)",
                "except Exception as e:",
                "    print(f'(Mermaid render failed: {e})')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — trace the loop count",
                "",
                "Run several questions, count how many loops each takes. Simple questions resolve in one loop; harder ones bounce.",
            ],
            code=[
                "import pandas as pd",
                "rows = []",
                "for q in [",
                "    'What is Rc?',                                    # easy",
                "    'When do I use Arc instead of Rc?',               # easy",
                "    'How do Arc and Mutex compose for shared state?', # medium",
                "    'Walk through how lifetimes affect a function that returns its longest argument.',  # hard",
                "]:",
                "    result = app.invoke({'question': q, 'query': q, 'contexts': [], 'loops': 0})",
                "    rows.append({'q': q[:55], 'loops': result.get('loops', 0), 'has_answer': bool(result.get('answer'))})",
                "pd.DataFrame(rows)",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does the critique propose at each loop?",
                "",
                "Look at the rewrite proposed at each iteration. Good rewrites narrow the search; bad ones repeat or drift.",
            ],
            code=[
                "# Step through manually with prints",
                "s: State = {'question': 'How do Arc and Mutex compose?', 'query': 'Arc Mutex', 'contexts': [], 'loops': 0}",
                "for step in range(3):",
                "    s.update(retrieve(s))",
                "    print(f'Loop {step+1} retrieved {len(s[\"contexts\"])} contexts so far.')",
                "    update = critique(s)",
                "    if 'answer' in update:",
                "        print(f'  Critique decided to answer.')",
                "        s.update(update)",
                "        break",
                "    print(f'  Next query: {update[\"query\"]}')",
                "    s.update(update)",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost",
                "",
                "Each loop is one retrieval + one critique LLM call. We measure.",
            ],
            code=[
                "from cookbook import _cache",
                "before = _cache.stats()['entries']",
                "_ = answer_question('How does the borrow checker reason about overlapping references?')",
                "after = _cache.stats()['entries']",
                "print(f'New cache entries: {after - before}')",
                "print('Each loop: 1 retrieval embed + 1 critique LLM call.')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a multi-hop question.",
    ],
    run_code=[
        "ans, _ = answer_question('Walk through how the borrow checker reasons about overlapping references in a function that returns the longer of two arguments.')",
        "print('=== LangGraph answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Vanilla (single retrieval) vs LangGraph (looping). On multi-hop questions the loop adds value; on single-hop, the loop is wasted work but doesn't hurt quality.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'Walk through how the borrow checker reasons about overlapping references.'",
        "base = vanilla_pipeline(q, corpus='rust-book', top_k=5)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'langgraph', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Six knobs in priority order:",
        "",
        "1. **Loop cap.** Default 3. Higher allows deeper multi-hop but invites runaway. Cap aggressively in production — a hung agent is worse than a partial answer.",
        "2. **Critique prompt.** Specific failure-mode prompts work better than generic \"is this answer good\". Phrase it as \"what evidence is still missing?\".",
        "3. **Retrieval `k` per loop.** We use 5. Lower keeps the context tight; higher gathers more evidence per loop.",
        "4. **State persistence.** LangGraph supports checkpointing — useful for long-running agents and resumable conversations across page refreshes.",
        "5. **Tracing.** Always enable Phoenix or LangSmith. Debugging an agent without traces is essentially impossible.",
        "6. **Node-level retries.** Wrap each node in a retry policy so transient model failures don't kill the whole loop.",
    ],

    discussion_md=[
        "Four failure modes you'll meet in production:",
        "",
        "- **Runaway loops.** Without the cap the agent loops forever. Always cap.",
        "- **Drifting critique.** The critique proposes worse follow-up questions over time. Tighten the critique prompt or reset the state on each loop.",
        "- **State bloat.** Contexts accumulate; the prompt eventually overflows. Truncate or summarise older contexts.",
        "- **Cyclic loops with no progress.** The critique keeps asking for the same retrieval. Detect by hashing the query each loop and break on repeat.",
        "",
        "Compose with Self-RAG (Recipe 24) inside critique, CRAG (Recipe 25) for fallback, and adaptive routing (Recipe 26) to decide whether to invoke the graph at all.",
    ],
)


# =============================================================================
# MCP Tool Retrieval
# =============================================================================
MCP_TOOLS = Recipe(
    path="recipes/06-adaptive-and-agentic/mcp-tool-retrieval.ipynb",
    title="MCP Tool Retrieval — Retrieving Across Tool Servers",
    category="adaptive-and-agentic",
    corpus_filter=None,

    theory_problem=[
        "Production assistants don't retrieve from one corpus — they retrieve from many: an internal wiki, a code search, a customer-data database, a docs site. Each one has its own API. Hard-coding that mesh into the agent breaks the day someone adds another tool.",
        "The Model Context Protocol (MCP) standardises how an agent discovers and calls remote tools. For retrieval, MCP servers expose vector stores, knowledge graphs, and search APIs as tools the model can call by name. The agent doesn't know it's talking to Pinecone vs Postgres vs Confluence — just to an MCP tool with a name and a description.",
    ],
    theory_origin=[
        "MCP was announced by Anthropic in November 2024 and donated to the Linux Foundation in December 2025. The spec describes a JSON-RPC protocol for tool discovery, invocation, and resource subscription. The retrieval pattern is a natural application: expose every searchable store as an MCP tool, let the agent pick which one to call. The standardisation was overdue — every framework had been inventing its own tool-use protocol, and the proliferation was hurting the ecosystem.",
        "By 2026 MCP servers exist for most major data stores and APIs — Notion, Slack, Linear, GitHub, the major vector stores. The cookbook mocks the protocol here to keep the notebook deterministic; production MCP setups use the official `mcp` Python client. The pattern below maps directly to real MCP usage — only the transport layer differs.",
    ],
    theory_landscape=[
        "MCP is one of three production tool-use patterns to know:",
        "",
        "- **OpenAI function calling.** Native JSON schemas, vendor-bound. Works inside OpenAI's API but not portable.",
        "- **LangChain Tool / LlamaIndex Tool.** Framework-bound abstractions over function calling. Portable across LLM providers but tied to one framework.",
        "- **MCP (this recipe).** Protocol-level standard for tool discovery and use, vendor-agnostic and framework-agnostic.",
        "",
        "MCP composes with everything — it's a transport, not a runtime. The agent loop logic still has to live somewhere; LangGraph (Recipe 28) is a good chassis. The protocol layer is for the discovery and invocation, not the orchestration.",
    ],
    theory_when_to_use=[
        "Use MCP when your agent needs to retrieve from multiple distinct stores, and you want a single protocol to handle them. Multi-tool agents, enterprise search assistants, anywhere ad-hoc API integrations would be painful — and where tools come from teams that don't want to maintain framework-specific adapters.",
        "Skip it for single-corpus systems. There's no benefit to the protocol layer when you have one store.",
        "Skip it when your tools change rarely. Hard-coded integrations are simpler when stable; MCP's value is at the discovery layer, not at the runtime layer.",
    ],
    theory_intuition=[
        "Five intuitions to carry:",
        "",
        "**Tools are named with descriptions.** The model picks by reading the descriptions, so descriptions matter as much as code. Tune them.",
        "",
        "**Tool calls are LLM-driven.** The model decides which tool to call and what arguments to pass. This is not a deterministic dispatcher.",
        "",
        "**Validate before executing.** A model can call any tool; validation belongs in the agent loop, not in the LLM call.",
        "",
        "**Observe everything.** Tool calls are the most leveraged points in your stack — instrument them with tracing.",
        "",
        "**Treat tool output as untrusted.** Anything coming back from a tool becomes context for the next LLM call. Don't let it invoke destructive tools.",
    ],

    architecture_mermaid="""
flowchart TB
  Q[Query] --> AG[Agent loop]
  AG --> LLM{LLM: pick<br/>tool & args}
  LLM --> T1[MCP tool: arxiv]
  LLM --> T2[MCP tool: rust]
  LLM --> T3[MCP tool: wiki]
  T1 --> AG
  T2 --> AG
  T3 --> AG
  AG --> A[Compose answer]
""",

    references=[
        Reference(
            title="Model Context Protocol — Anthropic announcement",
            url="https://www.anthropic.com/news/model-context-protocol",
            kind="blog",
            note="Original announcement.",
        ),
        Reference(
            title="MCP specification",
            url="https://modelcontextprotocol.io/",
            kind="docs",
            note="The protocol spec.",
        ),
        Reference(
            title="MCP Python SDK",
            url="https://github.com/modelcontextprotocol/python-sdk",
            kind="repo",
            note="Official Python client and server.",
        ),
        Reference(
            title="LangGraph + MCP tutorial",
            url="https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/",
            kind="docs",
            note="Composing MCP with stateful agents.",
        ),
        Reference(
            title="Anthropic agentic coding patterns",
            url="https://docs.anthropic.com/en/docs/build-with-claude/agents",
            kind="docs",
            note="Production patterns.",
        ),
        Reference(
            title="OpenAI function calling",
            url="https://platform.openai.com/docs/guides/function-calling",
            kind="docs",
            note="The vendor-bound predecessor.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build two retrieval stores",
                "",
                "We'll register them as MCP tools. The cookbook mocks the protocol layer; the rest of the pattern matches real MCP usage.",
            ],
            code=[
                "from cookbook.corpora import load_arxiv_mamba, load_rust_book",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "stores = {}",
                "for name, loader in [('arxiv', load_arxiv_mamba), ('rust', load_rust_book)]:",
                "    docs = list(loader())",
                "    chunks = sentence_window(docs, sentences_per_chunk=4)",
                "    vecs = client.embed([c.text for c in chunks])",
                "    s = QdrantBackend(f'mcp-{name}', dim=len(vecs[0]))",
                "    s.add([c.text for c in chunks], vecs, ids=[c.chunk_id for c in chunks])",
                "    stores[name] = s",
                "    print(f'Built tool: search_{name}_corpus ({len(chunks)} chunks)')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Define the tool registry",
                "",
                "Each tool has a name and a short description. The model picks by description.",
            ],
            code=[
                "TOOLS = [",
                "    {'name': 'search_arxiv_mamba', 'description': 'Search the Mamba state-space-model survey paper.'},",
                "    {'name': 'search_rust_book',   'description': 'Search The Rust Programming Language book.'},",
                "]",
                "print('Tools:')",
                "for t in TOOLS:",
                "    print(f'  - {t[\"name\"]}: {t[\"description\"]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Build the agent prompt",
                "",
                "The model is shown the tool list and asked to issue a call.",
            ],
            code=[
                "TOOL_PROMPT = (",
                "    'You have access to these tools (call by writing CALL: tool_name(\"query\")):\\n{tools}\\n\\n'",
                "    'Question: {q}\\nCall the most appropriate tool first.'",
                ")",
                "",
                "def render_tools():",
                "    return '\\n'.join(f\"- {t['name']}: {t['description']}\" for t in TOOLS)",
                "",
                "print(TOOL_PROMPT.format(tools=render_tools(), q='What is selective scan?'))",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Parse and execute tool calls",
                "",
                "We parse the model's output for `CALL: tool_name(\"query\")` and route to the right store.",
            ],
            code=[
                "import re",
                "",
                "def execute_call(tool: str, query: str) -> list[str]:",
                "    key = 'arxiv' if 'arxiv' in tool else 'rust'",
                "    qv = client.embed([query])[0]",
                "    return [h.text for h in stores[key].search(qv, top_k=3)]",
                "",
                "def parse_call(text: str) -> tuple[str, str] | None:",
                "    m = re.search(r'CALL:\\s*(\\w+)\\(\"([^\"]+)\"\\)', text)",
                "    return (m.group(1), m.group(2)) if m else None",
                "",
                "raw = client.chat(TOOL_PROMPT.format(tools=render_tools(), q='What is selective scan?'))",
                "print('Model raw output:')",
                "print(raw)",
                "print()",
                "call = parse_call(raw)",
                "print(f'Parsed call: {call}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Multi-step agent loop",
                "",
                "Loop until the model produces an answer or we hit the cap. Each iteration: ask the model what to do, execute the tool, accumulate observations.",
            ],
            code=[
                "def mcp_loop(question: str, max_steps: int = 3):",
                "    history = []",
                "    for step in range(max_steps):",
                "        prompt = TOOL_PROMPT.format(tools=render_tools(), q=question)",
                "        if history:",
                "            prompt += '\\n\\nObservations:\\n' + '\\n'.join(history)",
                "        out = client.chat(prompt)",
                "        call = parse_call(out)",
                "        if not call:",
                "            return out, history",
                "        tool, q = call",
                "        results = execute_call(tool, q)",
                "        history.append(f'[{tool}({q!r})] -> ' + ' | '.join(r[:120] for r in results))",
                "    final = client.chat(",
                "        'Summarize and answer based on these observations.\\n'",
                "        + '\\n'.join(history) + f'\\nQuestion: {question}'",
                "    )",
                "    return final, history",
                "",
                "ans, hist = mcp_loop('In the Mamba paper, what is selective scan, and how would I implement it in Rust given ownership constraints?')",
                "print(ans[:400])",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 6 — Wrap as `answer_question`",
                "",
                "Cookbook contract.",
            ],
            code=[
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    return mcp_loop(question)",
                "",
                "ans, _ = answer_question('What is selective scan?')",
                "print(ans[:300])",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — which tool gets picked for various questions?",
                "",
                "Test the model's tool selection across a battery.",
            ],
            code=[
                "for q in [",
                "    'What is selective scan?',                       # arxiv",
                "    'When should I use Arc over Rc?',                # rust",
                "    'How do state-space models scale linearly?',     # arxiv",
                "    'Walk me through borrow checking.',              # rust",
                "]:",
                "    raw = client.chat(TOOL_PROMPT.format(tools=render_tools(), q=q))",
                "    call = parse_call(raw)",
                "    tool = call[0] if call else '(no call)'",
                "    print(f'  {q[:55]:55s} -> {tool}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what happens when no tool fits?",
                "",
                "Ask a question outside both corpora. The model should either pick the closest tool or refuse.",
            ],
            code=[
                "raw = client.chat(TOOL_PROMPT.format(tools=render_tools(), q='What is the capital of France?'))",
                "print(raw)",
                "print(f'Parsed: {parse_call(raw)}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — full agent loop trace",
                "",
                "Print every step of the loop for one cross-corpus question. Useful for debugging the agent's reasoning.",
            ],
            code=[
                "ans, hist = mcp_loop('In the Mamba paper, what is selective scan, and how would I implement it in Rust given ownership constraints?')",
                "print('=== Agent trace ===')",
                "for step, obs in enumerate(hist):",
                "    print(f'Step {step+1}: {obs[:200]}')",
                "    print()",
                "print('=== Final answer ===')",
                "print(ans[:400])",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost",
                "",
                "Each agent step is one LLM call. Track them.",
            ],
            code=[
                "from cookbook import _cache",
                "before = _cache.stats()['entries']",
                "_ = answer_question('Explain Rc.')",
                "after = _cache.stats()['entries']",
                "print(f'New cache entries: {after - before}')",
                "print('Each step: 1 LLM call + 1 query embed + 1 vector search.')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a cross-corpus question.",
    ],
    run_code=[
        "ans, _ = answer_question('How would I write selective scan in idiomatic Rust given borrow-checker constraints?')",
        "print('=== MCP-routed answer ===')",
        "print(ans[:400])",
    ],

    comparison_md=[
        "Vanilla (single corpus) vs MCP (router across corpora). Vanilla can only see one corpus; MCP picks.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'What is selective scan in Mamba?'",
        "base = vanilla_pipeline(q, corpus='rust-book', top_k=5)  # wrong corpus on purpose",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla (rust only)', 'preview': base.answer[:160]},",
        "    {'pipeline': 'mcp routed', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Six knobs in priority order:",
        "",
        "1. **Tool descriptions.** The most important lever. Vague descriptions cause mis-routing; specific descriptions with example queries route well.",
        "2. **Tool list size.** Long lists confuse the model. Cap at ~10 visible tools per call. For larger tool sets, route to a tool-list first (a tool that finds tools).",
        "3. **Loop cap.** Default 3. Cap aggressively to prevent runaway agents and unbounded LLM bills.",
        "4. **Argument parsing.** Validate before executing. Models will pass bad arguments; treat tool input like any other user input.",
        "5. **Real MCP client.** Swap the mock for the `mcp` library in production. The cookbook pattern matches MCP usage.",
        "6. **Authentication.** Many MCP servers require authentication. Plan for credential management before going to production.",
    ],

    discussion_md=[
        "Four failure modes you'll meet:",
        "",
        "- **Tool description drift.** As you add tools, descriptions get vague. Refactor periodically; treat tool descriptions like API documentation.",
        "- **Hallucinated tool names.** The model may invent tool names. Validate against the registry before executing.",
        "- **Loop runaway.** Without the cap the agent loops forever. Cap.",
        "- **Prompt-injection via tool output.** Tool outputs go into the model's context. Treat them as untrusted input; never let them invoke destructive tools without explicit confirmation.",
        "",
        "Compose with LangGraph (Recipe 28) for the loop logic. Compose with prompt caching for the tool registry — it's identical across queries and is the largest single block in the prompt.",
    ],
)


# =============================================================================
# DSPy compiled RAG
# =============================================================================
DSPY_RAG = Recipe(
    path="recipes/06-adaptive-and-agentic/dspy-compiled-rag.ipynb",
    title="DSPy Compiled RAG — Optimise the Prompt, Don't Engineer It",
    category="adaptive-and-agentic",
    corpus_filter="wikipedia-superconductors",

    theory_problem=[
        "Prompt engineering is brittle. You write a prompt that works on five queries, and the sixth produces nonsense. You add an example to the prompt; that helps the sixth but breaks the third. Iterating by hand doesn't scale, and the resulting prompts are usually one inscrutable string of instructions, few-shot examples, and edge-case patches.",
        "DSPy treats prompts as parameters to optimise, not strings to handcraft. You write a Signature (inputs/outputs) and a Module (RAG pipeline), then compile against a small training set. The optimiser searches over instruction wording, few-shot examples, and chain-of-thought structure. The compiled program routinely outperforms hand-tuned prompts and produces an artifact you can version, test, and ship like any other compiled binary.",
    ],
    theory_origin=[
        "DSPy was published by the Stanford NLP group in 2023 (Khattab et al.). The framework consolidated several earlier ideas — auto-prompting, demonstration mining, programmatic LLM use — into a single composable system. By 2026 DSPy is the canonical tool for any team that wants to ship a prompt-engineered pipeline with measurable quality bounds.",
        "The optimisers — BootstrapFewShot, MIPRO, COPRO — search the prompt space differently. The cookbook uses BootstrapFewShot for simplicity; production setups often run MIPRO for better results. The framework's appeal is that it treats prompts as software, with all the engineering machinery that implies: versioning, testing, CI integration, and reproducible compilation.",
    ],
    theory_landscape=[
        "DSPy is one of several programmatic LLM tools to know:",
        "",
        "- **DSPy (this recipe).** Compile prompts against a training set with an optimiser.",
        "- **LangChain Hub.** Versioned prompt registry; no auto-optimisation.",
        "- **Instructor.** JSON-schema-constrained outputs; complementary to DSPy.",
        "- **Promptfoo / OpenAI Evals.** Evaluation-only; pair with DSPy for the compile-then-evaluate loop.",
        "",
        "DSPy composes with everything — it's a pre-processor that produces optimised prompts, which any runtime can use. The compiled artifact is just a JSON state file plus the Python program that loads it.",
    ],
    theory_when_to_use=[
        "Use DSPy when you have a labelled eval set and want to systematically improve quality. Research benchmarks, production systems with feedback loops, anywhere you can measure success programmatically and want the optimiser to do the prompt-engineering work for you.",
        "Skip it when you don't have an eval set. The optimiser needs feedback to improve, and without an eval set you're tuning blindly.",
        "Skip it on one-off prototypes. The compilation overhead doesn't pay off until you're shipping a stable artifact that gets evaluated regularly.",
    ],
    theory_intuition=[
        "Four intuitions to carry:",
        "",
        "**Signatures are typed contracts.** Each module declares inputs and outputs. The optimiser fills in the prompts that connect them, letting you focus on the data flow rather than the wording.",
        "",
        "**Optimisers search the prompt space.** Bootstrap finds good few-shot examples; MIPRO co-optimises instructions and examples; COPRO refines instructions. Each one is a different search strategy over the same prompt-shape space.",
        "",
        "**Compilation is the workflow.** Write the program once, compile against your eval set, ship the compiled artifact. The compile step replaces the iterative hand-tuning loop.",
        "",
        "**The metric is the optimisation target.** The optimiser is only as good as the metric you give it. Spend time on the metric before spending time tuning the optimiser.",
    ],

    architecture_mermaid="""
flowchart TB
  T[Training set<br/>questions+answers] --> O[Optimiser:<br/>search prompt space]
  P[Program: signatures<br/>+ modules] --> O
  O --> C[Compiled program<br/>with tuned prompts]
  Q[Query] --> C
  C --> A[Answer]
""",

    references=[
        Reference(
            title="DSPy — Programming foundation models",
            url="https://github.com/stanfordnlp/dspy",
            kind="repo",
            note="Official repository.",
        ),
        Reference(
            title="DSPy paper",
            url="https://arxiv.org/abs/2310.03714",
            kind="paper",
            note="The original Stanford paper.",
        ),
        Reference(
            title="MIPRO optimiser",
            url="https://arxiv.org/abs/2406.11695",
            kind="paper",
            note="The MIPRO optimiser used in production DSPy setups.",
        ),
        Reference(
            title="DSPy documentation",
            url="https://dspy.ai/",
            kind="docs",
            note="Tutorial and API reference.",
        ),
        Reference(
            title="Instructor — structured outputs",
            url="https://python.useinstructor.com/",
            kind="docs",
            note="JSON-schema-constrained outputs; complementary to DSPy.",
        ),
        Reference(
            title="LangChain Hub prompt registry",
            url="https://python.langchain.com/docs/integrations/providers/langchain_hub/",
            kind="docs",
            note="Alternative for prompt versioning.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build the corpus index",
                "",
                "Standard Wikipedia superconductors setup.",
            ],
            code=[
                "from cookbook.corpora import load_wikipedia_superconductors, load_eval_questions",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_wikipedia_superconductors())",
                "chunks = sentence_window(docs, sentences_per_chunk=4)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('dspy', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} chunks.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Configure DSPy with Nebius",
                "",
                "DSPy needs an LM. We point it at the same Nebius endpoint the rest of the cookbook uses.",
            ],
            code=[
                "import dspy, os",
                "",
                "lm = dspy.LM(",
                "    model=f'openai/{client.chat_model}',",
                "    api_base=os.getenv(client._spec.base_url_env) if client._spec.base_url_env else None,",
                "    api_key=os.getenv(client._spec.api_key_env),",
                ")",
                "dspy.configure(lm=lm)",
                "print('DSPy configured.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Define a Signature",
                "",
                "Signatures declare the input/output shape. The optimiser fills in the prompt.",
            ],
            code=[
                "class AnswerWithCitations(dspy.Signature):",
                "    \"\"\"Answer the question using the passages. Cite passages by their index.\"\"\"",
                "    question: str = dspy.InputField()",
                "    passages: list[str] = dspy.InputField()",
                "    answer: str = dspy.OutputField()",
                "",
                "print('Signature defined.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Build the RAG Module",
                "",
                "A Module composes signatures. Ours is just \"retrieve + answer\".",
            ],
            code=[
                "class CookbookRAG(dspy.Module):",
                "    def __init__(self, k: int = 5):",
                "        super().__init__()",
                "        self.k = k",
                "        self.answer = dspy.ChainOfThought(AnswerWithCitations)",
                "    def forward(self, question: str):",
                "        qv = client.embed([question])[0]",
                "        hits = store.search(qv, top_k=self.k)",
                "        return self.answer(question=question, passages=[h.text for h in hits])",
                "",
                "raw = CookbookRAG(k=5)",
                "print('Module built.')",
                "print()",
                "preview = raw(question='What is the Meissner effect?')",
                "print(preview.answer[:300])",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Build a small training set",
                "",
                "Pull a few labelled examples from the cookbook eval set. The optimiser uses them as ground truth.",
            ],
            code=[
                "trainset = []",
                "for row in [q for q in load_eval_questions() if q['corpus'] == 'wikipedia-superconductors'][:6]:",
                "    trainset.append(dspy.Example(question=row['question'], answer=row['answer']).with_inputs('question'))",
                "print(f'Training examples: {len(trainset)}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 6 — Compile with BootstrapFewShot",
                "",
                "The optimiser searches for good few-shot demonstrations and a tuned instruction.",
            ],
            code=[
                "def em_metric(example, pred, trace=None):",
                "    return float(example.answer.split('.')[0].lower() in (pred.answer or '').lower())",
                "",
                "tele = dspy.BootstrapFewShot(metric=em_metric, max_bootstrapped_demos=3, max_labeled_demos=3)",
                "compiled = tele.compile(raw, trainset=trainset)",
                "print('Compiled program ready.')",
                "",
                "pred = compiled(question='What does BCS theory explain?')",
                "print(pred.answer[:300])",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 7 — Wrap as `answer_question`",
                "",
                "Cookbook contract.",
            ],
            code=[
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    hits = store.search(qv, top_k=5)",
                "    pred = compiled(question=question)",
                "    return pred.answer, [h.text for h in hits]",
                "",
                "ans, _ = answer_question('How does flux pinning enable stable levitation?')",
                "print(ans[:400])",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does the compiled prompt look like?",
                "",
                "DSPy stores the compiled prompt inside the program. Print it.",
            ],
            code=[
                "try:",
                "    saved = compiled.dump_state()",
                "    import json",
                "    print(json.dumps(saved, indent=2)[:1200])",
                "except Exception as e:",
                "    print(f'(dump_state failed: {e})')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — raw vs compiled on the same question",
                "",
                "The compiled program should answer at least as well, often better.",
            ],
            code=[
                "q = 'What is critical temperature?'",
                "print('Raw answer:')",
                "print(raw(question=q).answer[:300])",
                "print()",
                "print('Compiled answer:')",
                "print(compiled(question=q).answer[:300])",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — chain-of-thought trace",
                "",
                "The ChainOfThought module exposes its reasoning. Look at it.",
            ],
            code=[
                "trace = compiled(question='What is the Meissner effect?')",
                "for attr in dir(trace):",
                "    if attr.startswith('_'): continue",
                "    val = getattr(trace, attr)",
                "    if isinstance(val, str):",
                "        print(f'{attr}: {val[:120]}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost",
                "",
                "Compilation costs N LLM calls per training example per optimiser step. After compilation, queries are vanilla.",
            ],
            code=[
                "print('Compile-time cost: O(trainset_size * candidates * steps) LLM calls.')",
                "print('Query-time cost: 1 LLM call per query (plus retrieval).')",
                "print('Recommendation: cache compile-time results aggressively.')",
            ],
        ),
    ],

    run_md=[
        "End-to-end on a representative question.",
    ],
    run_code=[
        "ans, _ = answer_question('Explain BCS theory and its prediction of Cooper pairing.')",
        "print('=== DSPy-compiled answer ===')",
        "print(ans[:400])",
    ],

    comparison_md=[
        "Vanilla vs DSPy-compiled. Same retrieval, different prompt.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'Explain BCS theory and its prediction of Cooper pairing.'",
        "base = vanilla_pipeline(q, corpus='wikipedia-superconductors', top_k=5)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'dspy-compiled', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Six knobs in priority order:",
        "",
        "1. **Training set size.** 5-10 examples are enough for BootstrapFewShot. MIPRO benefits from 30+. More is better up to a point.",
        "2. **Metric.** Define a faithful, fast metric. The optimiser is only as good as your metric — invest time here before tuning anything else.",
        "3. **Optimiser.** BootstrapFewShot is fast; MIPRO is slower but stronger. Try both and measure on held-out data.",
        "4. **Number of demos.** 3-5 demonstrations is the sweet spot. Higher inflates prompt length without much quality lift.",
        "5. **Caching.** Compilation is expensive; cache the compiled artifact and reuse across runs.",
        "6. **Hold-out validation set.** Always evaluate the compiled artifact on data the optimiser didn't see during compilation.",
    ],

    discussion_md=[
        "Four failure modes:",
        "",
        "- **Bad metric.** A noisy metric leads the optimiser astray. Spend time getting the metric right.",
        "- **Overfitting to small training sets.** The compiled prompt may memorise examples rather than generalise. Hold out a validation set.",
        "- **Compilation cost.** MIPRO can spend hundreds of LLM calls per compile. Plan accordingly and cache aggressively.",
        "- **Provider mismatch.** A program compiled against one model may not transfer to another. Re-compile when you change the answer LM.",
        "",
        "Compose with everything: DSPy can wrap Self-RAG modules, CRAG branches, and listwise rerankers. It is the prompt-optimisation layer that other recipes can plug into.",
    ],
)


RECIPES = [SELF_REFLECTIVE, CRAG, ADAPTIVE, SPECULATIVE, LANGGRAPH, MCP_TOOLS, DSPY_RAG]
