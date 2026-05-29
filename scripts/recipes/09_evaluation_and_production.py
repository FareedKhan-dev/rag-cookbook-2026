"""Category 9 — Evaluation & Production."""
from __future__ import annotations

from authoring import CodeStep, Recipe, Reference


# =============================================================================
# RAGAS triad evaluation
# =============================================================================
RAGAS = Recipe(
    path="recipes/09-evaluation-and-production/ragas-triad-eval.ipynb",
    title="RAGAS Triad — Faithfulness, Relevance, Precision, Recall",
    category="evaluation-and-production",
    corpus_filter="arxiv-mamba",

    theory_problem=[
        "You changed the chunker, the retriever, the reranker, and the prompt — did anything actually improve? Without metrics, every change becomes vibes. \"It feels better\" is not a release decision. RAGAS gives you four reference-free metrics that turn the question into a number.",
        "The triad — faithfulness, answer relevance, context precision, context recall — covers the four ways a RAG pipeline can fail: hallucinating beyond the context, answering the wrong question, retrieving noise, missing crucial passages. Each metric uses an LLM-as-judge with calibrated prompts. The numbers aren't ground truth, but they're consistent enough to move with the technique under test.",
    ],
    theory_origin=[
        "RAGAS was published by ExplodingGradients in mid-2023 (\"Automated Evaluation of Retrieval Augmented Generation\"). The framework consolidated several earlier evaluation patterns into a single library and made them composable. By 2026 RAGAS is the default open-source evaluation tool for RAG; competitors (DeepEval, Phoenix, Langfuse) all support similar metrics with slightly different framing.",
        "The reference-free design was the key innovation. Earlier eval frameworks required ground-truth answers; RAGAS computes faithfulness and context precision from the (question, contexts, answer) triple alone. That makes it practical to run on production traffic, not just benchmarks.",
    ],
    theory_landscape=[
        "Four RAG-evaluation tools to know in 2026:",
        "",
        "- **RAGAS (this recipe).** Reference-free metrics, mature ecosystem, integration with LlamaIndex and LangChain.",
        "- **DeepEval (Recipe 38).** Pytest-native metrics; better for CI integration with pass/fail thresholds.",
        "- **Arize Phoenix.** Traces + evaluations in one tool; visualisation-first.",
        "- **Langfuse / TruLens.** Production observability with evaluation; strong cost tracking.",
        "",
        "Pick by integration: RAGAS for ad-hoc analysis, DeepEval for CI, Phoenix for tracing, Langfuse for production. The four tools overlap on metric definitions; they differ on workflow integration and visualisation. Most production stacks use at least two of them together.",
    ],
    theory_when_to_use=[
        "Use RAGAS when you need to measure a change. Tuning chunk size, swapping embedders, adding a reranker — RAGAS gives the move a number.",
        "Skip it for tiny prototypes. The eval overhead doesn't pay off until you're iterating.",
        "Skip it when ground-truth labels are available and you can do simple metric comparisons. Faithfulness adds value; exact-match accuracy on labelled QA is simpler.",
    ],
    theory_intuition=[
        "Four intuitions to carry:",
        "",
        "**The triad is four metrics.** Faithfulness (no hallucination), answer relevance (answers the question), context precision (no junk retrieved), context recall (nothing important missed).",
        "",
        "**The LLM is the judge.** Each metric uses a prompt to score the (question, contexts, answer) triple. The judge model matters; consistent scoring requires a consistent judge.",
        "",
        "**Numbers are relative, not absolute.** RAGAS scores aren't comparable across teams or datasets. They are comparable within one team's eval set across pipeline versions.",
        "",
        "**80 questions is enough.** Larger eval sets give tighter confidence intervals but rarely change the rank order of pipelines. The cookbook ships 80 questions across 4 corpora.",
    ],

    architecture_mermaid="""
flowchart TB
  Q[Question] --> P[RAG pipeline]
  P --> A[Answer]
  P --> C[Contexts]
  Q --> J1[Faithfulness<br/>judge]
  A --> J1
  C --> J1
  Q --> J2[Answer relevance<br/>judge]
  A --> J2
  Q --> J3[Context precision<br/>judge]
  C --> J3
  Q --> J4[Context recall<br/>judge]
  C --> J4
  GT[Ground truth] --> J4
  J1 --> M[Metric dict]
  J2 --> M
  J3 --> M
  J4 --> M
""",

    references=[
        Reference(
            title="RAGAS — Automated Evaluation of Retrieval Augmented Generation",
            url="https://arxiv.org/abs/2309.15217",
            kind="paper",
            note="The original paper.",
        ),
        Reference(
            title="RAGAS documentation",
            url="https://docs.ragas.io/",
            kind="docs",
            note="Official docs.",
        ),
        Reference(
            title="explodinggradients/ragas repository",
            url="https://github.com/explodinggradients/ragas",
            kind="repo",
            note="Source and examples.",
        ),
        Reference(
            title="DeepEval (Recipe 38)",
            url="https://docs.confident-ai.com/",
            kind="docs",
            note="The pytest-native alternative.",
        ),
        Reference(
            title="Arize Phoenix",
            url="https://docs.arize.com/phoenix",
            kind="docs",
            note="Tracing + evaluation tool.",
        ),
        Reference(
            title="Langfuse evaluation",
            url="https://langfuse.com/docs/scores",
            kind="docs",
            note="Production-observability variant.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build a pipeline to evaluate",
                "",
                "Standard vanilla pipeline on the Mamba paper. We'll measure this as the baseline; later recipes plug in better pipelines.",
            ],
            code=[
                "from cookbook.corpora import load_arxiv_mamba, load_eval_questions",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_arxiv_mamba())",
                "chunks = sentence_window(docs, sentences_per_chunk=4)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('ragas', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "",
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    hits = store.search(qv, top_k=5)",
                "    ctx = [h.text for h in hits]",
                "    a = client.chat('Use only these passages.\\n' + '\\n\\n'.join(ctx) + f'\\nQ: {question}\\nA:')",
                "    return a, ctx",
                "",
                "print('Pipeline ready.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Run the pipeline on the eval slice",
                "",
                "Pick 8 questions from the cookbook eval set, run the pipeline, collect (question, expected, actual, contexts).",
            ],
            code=[
                "samples = []",
                "for row in [q for q in load_eval_questions() if q['corpus'] == 'arxiv-mamba'][:8]:",
                "    actual, contexts = answer_question(row['question'])",
                "    samples.append({",
                "        'question': row['question'],",
                "        'expected_answer': row['answer'],",
                "        'actual_answer': actual,",
                "        'contexts': contexts,",
                "    })",
                "print(f'Collected {len(samples)} samples.')",
                "print()",
                "print(samples[0]['actual_answer'][:240])",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Run RAGAS metrics",
                "",
                "We use `cookbook.eval.evaluate`, which wraps RAGAS with cookbook defaults. Each metric is computed per-sample and averaged.",
            ],
            code=[
                "import os",
                "# RAGAS uses an LLM judge — point it at Nebius via OpenAI-compatible env vars.",
                "os.environ.setdefault('OPENAI_API_KEY', os.environ.get(client._spec.api_key_env, ''))",
                "os.environ.setdefault('OPENAI_BASE_URL', os.environ.get(client._spec.base_url_env, '') or '')",
                "",
                "from cookbook.eval import EvalSample, evaluate",
                "",
                "ev_samples = [",
                "    EvalSample(",
                "        question=s['question'],",
                "        expected_answer=s['expected_answer'],",
                "        contexts=s['contexts'],",
                "        actual_answer=s['actual_answer'],",
                "    ) for s in samples",
                "]",
                "metrics = evaluate(ev_samples, use_ragas=True, use_deepeval=False)",
                "metrics",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Step 4 — Interpret the numbers",
                "",
                "Faithfulness near 1.0 means the answer is grounded; below 0.7 means the model hallucinated. Answer relevance scores how well the answer addresses the question. Context precision / recall measure whether retrieval was tight (precision) and complete (recall).",
            ],
            code=[
                "import pandas as pd",
                "rows = [{'metric': k, 'score': round(v, 3)} for k, v in metrics.items()]",
                "df = pd.DataFrame(rows)",
                "df['interpretation'] = df['score'].apply(lambda s: 'good' if s >= 0.7 else 'investigate' if s >= 0.5 else 'broken')",
                "df",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Define a function to evaluate any pipeline",
                "",
                "Now you can evaluate any `answer_question` against the eval set. Use this template to compare pipelines from earlier recipes.",
            ],
            code=[
                "def evaluate_pipeline(fn, corpus_key: str, k: int = 8):",
                "    rows = [q for q in load_eval_questions() if q['corpus'] == corpus_key][:k]",
                "    ev = []",
                "    for r in rows:",
                "        ans, ctx = fn(r['question'])",
                "        ev.append(EvalSample(",
                "            question=r['question'], expected_answer=r['answer'],",
                "            contexts=ctx, actual_answer=ans,",
                "        ))",
                "    return evaluate(ev, use_ragas=True)",
                "",
                "print('evaluate_pipeline ready.')",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — read a low-scoring sample",
                "",
                "When the average is low, find the worst sample and read it. Often the failure is concentrated in one or two queries.",
            ],
            code=[
                "for s in samples[:3]:",
                "    print(f'Q: {s[\"question\"][:80]}')",
                "    print(f'  expected: {s[\"expected_answer\"][:120]}')",
                "    print(f'  actual:   {s[\"actual_answer\"][:120]}')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — does context count matter?",
                "",
                "Re-run the eval at different `k` values to see how chunk count affects metrics.",
            ],
            code=[
                "import pandas as pd",
                "rows = []",
                "for k in (3, 5, 8):",
                "    def fn_k(q, k=k):",
                "        qv = client.embed([q])[0]",
                "        hits = store.search(qv, top_k=k)",
                "        ctx = [h.text for h in hits]",
                "        a = client.chat('Use only these passages.\\n' + '\\n\\n'.join(ctx) + f'\\nQ: {q}\\nA:')",
                "        return a, ctx",
                "    m = evaluate_pipeline(fn_k, 'arxiv-mamba', k=5)",
                "    rows.append({'k': k, **{key.replace('ragas_', ''): round(v, 3) for key, v in m.items()}})",
                "pd.DataFrame(rows)",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost of running RAGAS",
                "",
                "Each metric uses an LLM judge call per sample. Track them.",
            ],
            code=[
                "from cookbook import _cache",
                "before = _cache.stats()['entries']",
                "print(f'Cache before: {before}')",
                "print(f'RAGAS adds ~3-4 LLM calls per sample across the four metrics.')",
                "print(f'For 8 samples: ~30 LLM calls.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does the per-sample distribution look like?",
                "",
                "Averages hide the spread. A pipeline with mean 0.7 and high variance is different from mean 0.7 and tight variance.",
            ],
            code=[
                "print('Score interpretation guide:')",
                "print('  faithfulness ≥ 0.75: grounded answer')",
                "print('  answer_relevancy ≥ 0.75: addresses the question')",
                "print('  context_precision ≥ 0.65: low junk in retrieved set')",
                "print('  context_recall ≥ 0.55: catches required information')",
            ],
        ),
    ],

    run_md=[
        "Run the evaluation end-to-end on the eval slice.",
    ],
    run_code=[
        "import pandas as pd",
        "metrics = evaluate_pipeline(answer_question, 'arxiv-mamba')",
        "pd.DataFrame([{'metric': k, 'score': round(v, 3)} for k, v in metrics.items()])",
    ],

    comparison_md=[
        "Vanilla pipeline vs the same pipeline with a reranker (loose proxy — we won't actually run rerank here to keep the cell fast). The point of the comparison is to show the evaluation harness; the recipes you've already authored have the techniques.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "def vanilla_fn(q):",
        "    r = vanilla_pipeline(q, corpus='arxiv-mamba', top_k=5)",
        "    return r.answer, r.contexts",
        "",
        "import pandas as pd",
        "m_vanilla = evaluate_pipeline(vanilla_fn, 'arxiv-mamba')",
        "m_ours    = evaluate_pipeline(answer_question, 'arxiv-mamba')",
        "",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', **{k.replace('ragas_', ''): round(v, 3) for k, v in m_vanilla.items()}},",
        "    {'pipeline': 'ours',    **{k.replace('ragas_', ''): round(v, 3) for k, v in m_ours.items()}},",
        "])",
    ],

    tuning_md=[
        "Seven knobs in priority order:",
        "",
        "1. **Eval-set size.** 8 questions is the smoke-test floor; 80+ for confidence in rank ordering.",
        "2. **Judge model.** Use a model at least as capable as your answer model. Mismatched judges produce noisy scores.",
        "3. **Metric selection.** All four metrics or only some. Faithfulness and answer relevance are universally useful.",
        "4. **Per-corpus splits.** Run eval per corpus to see where the pipeline excels and where it struggles.",
        "5. **Caching judge calls.** RAGAS judge calls are expensive; cookbook cache helps re-runs.",
        "6. **Statistical tests.** When comparing pipelines, run paired tests on per-sample scores, not just averages.",
        "7. **Per-difficulty stratification.** Slice the eval set by difficulty (easy/medium/hard) and report per-bucket so improvements aren't hidden by averaging.",
    ],

    discussion_md=[
        "Four failure modes you'll meet:",
        "",
        "- **Noisy judges.** Cheap judge models produce noisy scores. Use the same judge across runs to keep comparisons valid.",
        "- **Small eval sets.** 8 questions is enough to detect huge regressions, not enough for fine ranking. Scale up before claiming wins.",
        "- **Metric over-optimisation.** Tuning to RAGAS scores can produce pipelines that look great on RAGAS but fail in user studies. Always sanity-check with humans.",
        "- **Cost.** RAGAS is expensive at scale. Sample, don't evaluate every production query.",
        "",
        "Compose with DeepEval (Recipe 38) for CI integration. Compose with Phoenix (Recipe 39) for trace-level diagnostics on failing samples. Compose with Lynx guardrails (Recipe 40) for production-time grounding checks.",
    ],
)


# =============================================================================
# DeepEval in Pytest CI
# =============================================================================
DEEPEVAL = Recipe(
    path="recipes/09-evaluation-and-production/deepeval-pytest-ci.ipynb",
    title="DeepEval — Metrics in Your CI Pipeline",
    category="evaluation-and-production",
    corpus_filter="rust-book",

    theory_problem=[
        "RAGAS gives you metrics on demand. CI gives you metrics on every commit. Without CI integration, RAG quality regressions ship to production silently. DeepEval treats LLM evaluations as pytest tests: write a metric threshold, the CI build fails if a PR drops below it. The setup takes a few hours; the savings show up the first time a refactor would have shipped a regression.",
        "The CI gate is what makes RAG evolution sustainable. Without it, every team eventually hits the same pattern — quality drifts down quietly, users complain, someone runs a manual eval, the team scrambles. The gate prevents the silent drift by forcing every PR to defend its quality bar before merging.",
    ],
    theory_origin=[
        "DeepEval was released by Confident AI in 2024 as a pytest-first LLM evaluation framework. Its design choice — `LLMTestCase` objects, metric classes with `measure()` and `is_successful()`, pytest decorators — made it the natural fit for CI pipelines. By 2026 DeepEval is the default for teams that want quality gates without writing custom pytest fixtures, and the framework has settled into a mature shape with stable metric definitions.",
        "The pytest-first design is the reason it sticks. Engineers already know pytest; adding LLM tests doesn't introduce a new framework. CI pipelines need no special integration; DeepEval tests run alongside unit tests under the same `pytest` invocation, get the same pass/fail signals, and ship through the same release-gate machinery.",
    ],
    theory_landscape=[
        "Where DeepEval fits among RAG evaluation tools:",
        "",
        "- **RAGAS (Recipe 37).** Notebook-first, ad-hoc evaluation. Best for exploration and one-off analysis.",
        "- **DeepEval (this recipe).** Pytest-first, CI evaluation with hard pass/fail thresholds.",
        "- **Phoenix (Recipe 39).** Trace-first, debugging. Best when you want to inspect individual failures.",
        "- **Langfuse.** Production observability with cost tracking.",
        "",
        "All four overlap on metrics; they differ on workflow integration. Pick by where in your stack the evaluation lives. DeepEval is the only one designed to gate releases via pytest, so it's the natural choice for CI-driven teams.",
    ],
    theory_when_to_use=[
        "Use DeepEval to gate releases. Any team shipping RAG changes through PR should have a DeepEval test suite that runs on every PR.",
        "Skip it for exploratory work. RAGAS notebook usage is faster for iteration.",
        "Skip it when CI is slow. DeepEval suites that take 30 minutes will get bypassed; design for fast feedback.",
    ],
    theory_intuition=[
        "Four intuitions:",
        "",
        "**Tests are assertions on metrics.** Each test creates an `LLMTestCase`, calls a metric's `measure()`, asserts that `is_successful()`.",
        "",
        "**Thresholds are the contract.** Each metric has a threshold (default 0.7). Below threshold = test fails = PR blocked.",
        "",
        "**Cache the judge calls.** Without caching, DeepEval re-runs every judge on every CI build. With caching, only changed pipelines re-evaluate.",
        "",
        "**Subset the eval set.** Run a fast subset on every PR, the full set nightly.",
    ],

    architecture_mermaid="""
flowchart LR
  PR[PR] --> CI[CI build]
  CI --> T[Run pytest<br/>with DeepEval suite]
  T --> M[Each test:<br/>measure metric]
  M --> P{All pass?}
  P -->|yes| MERGE[Allow merge]
  P -->|no| BLOCK[Block merge]
""",

    references=[
        Reference(
            title="DeepEval documentation",
            url="https://docs.confident-ai.com/",
            kind="docs",
            note="Official docs.",
        ),
        Reference(
            title="confident-ai/deepeval repository",
            url="https://github.com/confident-ai/deepeval",
            kind="repo",
            note="Source.",
        ),
        Reference(
            title="DeepEval metrics overview",
            url="https://docs.confident-ai.com/docs/metrics-introduction",
            kind="docs",
            note="The metrics catalogue.",
        ),
        Reference(
            title="pytest fixtures for LLM evaluation",
            url="https://docs.pytest.org/en/stable/how-to/fixtures.html",
            kind="docs",
            note="The pytest mechanism DeepEval builds on.",
        ),
        Reference(
            title="RAGAS (Recipe 37)",
            url="https://docs.ragas.io/",
            kind="docs",
            note="The notebook-first alternative.",
        ),
        Reference(
            title="GitHub Actions for CI",
            url="https://docs.github.com/en/actions",
            kind="docs",
            note="Where the DeepEval suite typically runs.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build a small pipeline",
                "",
                "Standard Rust book RAG. We'll write tests against it.",
            ],
            code=[
                "from cookbook.corpora import load_rust_book",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_rust_book())",
                "chunks = sentence_window(docs, sentences_per_chunk=4)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('deepeval', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "",
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    hits = store.search(qv, top_k=5)",
                "    ctx = [h.text for h in hits]",
                "    return client.chat('Use these.\\n' + '\\n\\n'.join(ctx) + f'\\nQ: {question}\\nA:'), ctx",
                "",
                "print('Pipeline ready.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Build a single DeepEval test case",
                "",
                "An `LLMTestCase` carries everything DeepEval needs: input, output, expected output, retrieved context.",
            ],
            code=[
                "import os",
                "os.environ.setdefault('OPENAI_API_KEY', os.environ.get(client._spec.api_key_env, ''))",
                "os.environ.setdefault('OPENAI_BASE_URL', os.environ.get(client._spec.base_url_env, '') or '')",
                "",
                "from deepeval.test_case import LLMTestCase",
                "from cookbook.eval import _build_deepeval_model",
                "judge = _build_deepeval_model()",
                "",
                "q = 'What is ownership in Rust?'",
                "ans, ctx = answer_question(q)",
                "tc = LLMTestCase(",
                "    input=q,",
                "    actual_output=ans,",
                "    expected_output='Ownership is a set of rules that govern memory management in Rust.',",
                "    retrieval_context=ctx,",
                ")",
                "print(f'Test case input: {tc.input[:60]}')",
                "print(f'Test case actual_output: {tc.actual_output[:120]}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Run a single metric",
                "",
                "`FaithfulnessMetric` reads the test case and produces a score. `is_successful()` checks against the threshold.",
            ],
            code=[
                "from deepeval.metrics import FaithfulnessMetric",
                "",
                "m = FaithfulnessMetric(threshold=0.7, model=judge)",
                "m.measure(tc)",
                "print(f'Faithfulness score: {m.score:.3f}')",
                "print(f'Is successful: {m.is_successful()}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Write the pytest file",
                "",
                "DeepEval tests are normal pytest. The file goes in `tests/test_metrics.py` and is picked up by `pytest`.",
            ],
            code=[
                "TEST = '''",
                "from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric",
                "from deepeval.test_case import LLMTestCase",
                "",
                "def test_baseline_faithfulness():",
                "    tc = LLMTestCase(",
                "        input='What is ownership in Rust?',",
                "        actual_output='Ownership is a set of rules that govern how Rust manages memory.',",
                "        expected_output='Ownership rules govern memory management.',",
                "        retrieval_context=['Ownership is Rust's most unique feature; it has deep implications for memory safety.'],",
                "    )",
                "    m = FaithfulnessMetric(threshold=0.7)",
                "    m.measure(tc)",
                "    assert m.is_successful(), f'faithfulness={m.score}'",
                "",
                "def test_baseline_relevancy():",
                "    tc = LLMTestCase(",
                "        input='What is ownership in Rust?',",
                "        actual_output='Ownership is a set of rules that govern how Rust manages memory.',",
                "        expected_output='Ownership rules govern memory management.',",
                "        retrieval_context=['Ownership is Rust's most unique feature.'],",
                "    )",
                "    m = AnswerRelevancyMetric(threshold=0.7)",
                "    m.measure(tc)",
                "    assert m.is_successful(), f'answer_relevancy={m.score}'",
                "'''",
                "from pathlib import Path",
                "p = Path('/tmp/test_metrics.py')",
                "p.write_text(TEST, encoding='utf-8')",
                "print(f'Wrote {p}')",
                "print()",
                "print('Run with: pytest /tmp/test_metrics.py')",
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
                "# `answer_question` was defined earlier; nothing new to wire here.",
                "ans, _ = answer_question('What is borrowing?')",
                "print(ans[:200])",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — score multiple metrics on one test case",
                "",
                "Each metric is an independent assertion; CI builds usually run several.",
            ],
            code=[
                "from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric",
                "",
                "for M in (FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric):",
                "    m = M(threshold=0.7, model=judge)",
                "    m.measure(tc)",
                "    print(f'  {M.__name__:30s}  score={m.score:.3f}  pass={m.is_successful()}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — design a CI subset and a nightly suite",
                "",
                "The CI suite should run in under 5 minutes. Subset aggressively.",
            ],
            code=[
                "print('Recommended CI design:')",
                "print('  PR suite: 5-10 representative tests, <5 min runtime.')",
                "print('  Nightly:  full 80+ question eval set with all metrics.')",
                "print('  Quarterly: human eval on a labelled subset.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — how to handle flakiness",
                "",
                "LLM judges are noisy. Plan for retries.",
            ],
            code=[
                "print('Flakiness mitigation:')",
                "print('  pytest-retry for transient flakes.')",
                "print('  Lower thresholds on CI than on quarterly review.')",
                "print('  Compute confidence intervals on the eval set; reject only outside CI.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost",
                "",
                "DeepEval costs one judge call per test per metric. At 10 tests × 3 metrics × $0.001 per judge call = $0.03 per CI run.",
            ],
            code=[
                "n_tests = 10",
                "n_metrics = 3",
                "cost_per_judge = 0.001  # rough",
                "total_cost = n_tests * n_metrics * cost_per_judge",
                "print(f'Estimated cost per CI run: ${total_cost:.2f}')",
            ],
        ),
    ],

    run_md=[
        "Run a small DeepEval suite against the pipeline.",
    ],
    run_code=[
        "from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric",
        "from deepeval.test_case import LLMTestCase",
        "",
        "from cookbook.corpora import load_eval_questions",
        "results = []",
        "for row in [q for q in load_eval_questions() if q['corpus'] == 'rust-book'][:3]:",
        "    actual, ctx = answer_question(row['question'])",
        "    tc = LLMTestCase(",
        "        input=row['question'], actual_output=actual,",
        "        expected_output=row['answer'], retrieval_context=ctx,",
        "    )",
        "    fm = FaithfulnessMetric(threshold=0.7, model=judge); fm.measure(tc)",
        "    am = AnswerRelevancyMetric(threshold=0.7, model=judge); am.measure(tc)",
        "    results.append({",
        "        'q': row['question'][:50],",
        "        'faithfulness': round(fm.score, 2),",
        "        'relevancy': round(am.score, 2),",
        "    })",
        "import pandas as pd",
        "pd.DataFrame(results)",
    ],

    comparison_md=[
        "Vanilla vs vanilla — DeepEval is a measurement tool, not a pipeline; the comparison just shows the same pipeline can be measured.",
    ],
    comparison_code=[
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'notes': 'Measured with DeepEval Faithfulness / AnswerRelevancy metrics.'},",
        "])",
    ],

    tuning_md=[
        "Six knobs in priority order:",
        "",
        "1. **Threshold per metric.** Default 0.7. Higher fails more PRs; lower lets through regressions. Calibrate on a labelled set.",
        "2. **Test selection.** Pick representative tests for CI; full set for nightly. Aim for under 5 minute CI feedback.",
        "3. **Judge model.** Mid-tier is fine. Bigger judges add cost without quality lift.",
        "4. **Caching.** Cache judge calls so unchanged code doesn't re-evaluate. Keys on (model, input, output).",
        "5. **Retry on flakes.** pytest-retry handles transient judge failures and provider rate limits.",
        "6. **Parametrise eval data.** Use pytest's `parametrize` to run the same metric across many test cases without code duplication.",
    ],

    discussion_md=[
        "Four failure modes you'll meet:",
        "",
        "- **Flakes.** LLM judges aren't deterministic. Retries handle most; the rest needs threshold tolerance.",
        "- **Tests too slow.** Engineers will bypass slow CI. Subset aggressively to keep feedback under 5 minutes.",
        "- **Over-fitting to tests.** Tuning to the CI suite produces pipelines that pass CI but fail in production. Diversify the test set.",
        "- **Schema drift.** When the cookbook eval JSONL changes, DeepEval tests against old schemas break silently. Pin schemas.",
        "",
        "Compose with RAGAS (Recipe 37) for ad-hoc analysis, Phoenix (Recipe 39) for trace inspection on failing tests, and Lynx guardrails (Recipe 40) for production-time enforcement.",
    ],
)


# =============================================================================
# Phoenix Tracing
# =============================================================================
PHOENIX = Recipe(
    path="recipes/09-evaluation-and-production/phoenix-tracing-debugging.ipynb",
    title="Arize Phoenix — Trace Every Call, Debug What Matters",
    category="evaluation-and-production",
    corpus_filter="rust-book",

    theory_problem=[
        "When a RAG pipeline gives a bad answer, the debugging question is: was the retrieval wrong, or did the model just hallucinate from good retrieval? Without traces, you can't tell. You see the bad answer, you can't reach the chain of decisions that produced it, and you end up rerunning by hand to reproduce, which is slow and often impossible at production scale.",
        "Arize Phoenix gives every LLM call, embedding call, and vector search a span in an OpenTelemetry trace. When an answer goes wrong, you walk the trace tree — which chunks got retrieved, what prompt was assembled, what the model said. Production debugging stops being archaeology and becomes reading a tree.",
    ],
    theory_origin=[
        "Phoenix shipped from Arize in 2023 as an open-source LLM observability tool. The original goal — trace, evaluate, and visualise LLM apps — overlapped with what LangChain and LlamaIndex were doing internally; the productisation was making it framework-agnostic via OpenTelemetry semantic conventions. The OTel standardisation was the durable contribution: spans emitted from Phoenix are consumable by other tools, and vice versa.",
        "By 2026 Phoenix is the de facto open-source LLM tracing tool. Competitors (LangSmith, Langfuse, Helicone) all support similar functionality with different framing; Phoenix's openness and local-first design have kept it dominant for development workflows where engineers want to inspect traces without authenticating to a hosted service.",
    ],
    theory_landscape=[
        "LLM tracing tools to know in 2026:",
        "",
        "- **Phoenix (this recipe).** Open-source, local, OpenTelemetry-native. Best for development workflows.",
        "- **LangSmith.** Hosted, LangChain-integrated. Best for LangChain teams that want hosted persistence.",
        "- **Langfuse.** Open + hosted, strong scoring + cost tracking. Best for production observability.",
        "- **Helicone.** Hosted, proxy-based, cheap. Best for teams that want zero-code observability via API proxy.",
        "",
        "Pick by where you want spans: Phoenix for local dev, LangSmith for LangChain teams, Langfuse for production observability. The OpenTelemetry standardisation means the same instrumentation code can target multiple back-ends, which keeps you from getting locked in.",
    ],
    theory_when_to_use=[
        "Use Phoenix during development. Every recipe in this cookbook benefits from tracing enabled.",
        "Use it during incident response. When users report a bad answer, the trace tells you why.",
        "Skip it only when you cannot run locally. There's no downside to enabling it.",
    ],
    theory_intuition=[
        "Four intuitions:",
        "",
        "**Every call is a span.** LLM calls, embedding calls, vector search, even framework code — each gets a span with timing, input, output.",
        "",
        "**Spans nest into traces.** A query that triggers retrieval + generation produces a trace tree.",
        "",
        "**OpenTelemetry semantic conventions standardise the shape.** Phoenix, LangSmith, and Langfuse all consume the same span format.",
        "",
        "**Local-first is the win.** Phoenix runs as `phoenix.launch_app()` on http://localhost:6006. No accounts, no cloud, no waiting.",
    ],

    architecture_mermaid="""
flowchart TB
  P[RAG pipeline] --> E[Embedding call]
  P --> R[Vector search]
  P --> L[LLM call]
  E --> S[OTel span]
  R --> S
  L --> S
  S --> PH[(Phoenix UI<br/>localhost:6006)]
""",

    references=[
        Reference(
            title="Arize Phoenix documentation",
            url="https://docs.arize.com/phoenix",
            kind="docs",
            note="Official docs.",
        ),
        Reference(
            title="Arize-ai/phoenix repository",
            url="https://github.com/Arize-ai/phoenix",
            kind="repo",
            note="Source.",
        ),
        Reference(
            title="OpenTelemetry GenAI semantic conventions",
            url="https://opentelemetry.io/docs/specs/semconv/gen-ai/",
            kind="docs",
            note="The span format Phoenix consumes.",
        ),
        Reference(
            title="LangSmith documentation",
            url="https://docs.smith.langchain.com/",
            kind="docs",
            note="The hosted alternative.",
        ),
        Reference(
            title="Langfuse documentation",
            url="https://langfuse.com/docs",
            kind="docs",
            note="Open + hosted with scoring.",
        ),
        Reference(
            title="OpenInference instrumentation",
            url="https://github.com/Arize-ai/openinference",
            kind="repo",
            note="The library that emits OTel spans from popular frameworks.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build a small pipeline",
                "",
                "We'll deliberately break it slightly to make the trace interesting. Tiny chunks produce poor retrievals; the trace will show this.",
            ],
            code=[
                "from cookbook.corpora import load_rust_book",
                "from cookbook.chunkers import fixed_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_rust_book())",
                "chunks = fixed_window(docs, target_tokens=64, overlap_tokens=0)  # deliberately too small",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('phoenix', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print(f'Indexed {len(chunks)} tiny chunks.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Enable Phoenix tracing",
                "",
                "We call `cookbook.tracing.init_tracing(backend='phoenix')`. This launches a local Phoenix UI at http://localhost:6006 and configures OpenTelemetry to emit spans there.",
            ],
            code=[
                "from cookbook.tracing import init_tracing",
                "print(init_tracing(backend='off'))",
                "print()",
                "print('In a normal dev workflow you would call init_tracing(backend=\"phoenix\")')",
                "print('which launches http://localhost:6006 with one trace per LLM call.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Issue a query and inspect the trace",
                "",
                "With Phoenix enabled, each `client.chat()` and `client.embed()` call produces a span. The notebook just shows the answer; the UI shows the tree.",
            ],
            code=[
                "q = 'Walk through how the borrow checker reasons about overlapping references.'",
                "qv = client.embed([q])[0]",
                "hits = store.search(qv, top_k=5)",
                "answer = client.chat(",
                "    'Use these passages:\\n' + '\\n\\n'.join(h.text for h in hits)",
                "    + f'\\n\\nQ: {q}\\nA:'",
                ")",
                "print(answer[:300])",
                "print()",
                "print('Open http://localhost:6006 to inspect the trace tree.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Wrap as `answer_question`",
                "",
                "Cookbook contract.",
            ],
            code=[
                "def answer_question(question: str, k: int = 5) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    hits = store.search(qv, top_k=k)",
                "    ctx = [h.text for h in hits]",
                "    a = client.chat('Use these.\\n' + '\\n\\n'.join(ctx) + f'\\nQ: {question}\\nA:')",
                "    return a, ctx",
                "",
                "ans, _ = answer_question('What is interior mutability?')",
                "print(ans[:200])",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Step 5 — Show how to convert a trace into an eval test",
                "",
                "Production failures become CI tests. The pattern: capture the trace, extract the (question, contexts, answer), assert it satisfies the regression test you wrote.",
            ],
            code=[
                "captured = {",
                "    'question': 'What is interior mutability?',",
                "    'contexts': ['Interior mutability is a design pattern in Rust that lets you mutate data through an immutable reference.'],",
                "    'answer': 'Interior mutability lets you mutate data even through an immutable reference, enforced dynamically.',",
                "}",
                "print('Captured trace:')",
                "for k, v in captured.items():",
                "    s = v if isinstance(v, str) else ' / '.join(v)",
                "    print(f'  {k}: {s[:120]}')",
                "print()",
                "print('This dict is ready to seed a DeepEval test (Recipe 38).')",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what fields does a span carry?",
                "",
                "OpenTelemetry GenAI spans include input, output, model name, token counts, latency. Phoenix renders them all.",
            ],
            code=[
                "print('Spans typically include:')",
                "print('  - input text / messages')",
                "print('  - output text')",
                "print('  - model name')",
                "print('  - prompt tokens / completion tokens')",
                "print('  - latency')",
                "print('  - parent / child span IDs')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — how does Phoenix help during debugging?",
                "",
                "When a query gives a bad answer, the trace tells you where it went wrong: bad retrieval, bad prompt assembly, or bad generation.",
            ],
            code=[
                "print('Common diagnosis patterns:')",
                "print('  Top-1 retrieval irrelevant -> retrieval is broken (wrong embedder, bad chunks).')",
                "print('  Top-3 relevant but answer wrong -> prompt or model is the issue.')",
                "print('  High token count -> chunks are too long; truncate.')",
                "print('  Slow latency on retrieval -> vector index needs tuning.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — exporting traces to a dataset",
                "",
                "Phoenix can convert traces into evaluation datasets. One failing trace becomes a regression test.",
            ],
            code=[
                "print('Phoenix exports:')",
                "print('  - Trace -> evaluation dataset')",
                "print('  - Dataset -> RAGAS / DeepEval input')",
                "print('  - Failing dataset -> regression suite')",
                "print()",
                "print('This closes the loop: production failures become CI tests.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — local vs hosted",
                "",
                "Phoenix runs locally for free. For production, use the hosted Arize platform with the same span format.",
            ],
            code=[
                "print('Local Phoenix: free, ephemeral, no auth.')",
                "print('Hosted Arize: persistent, alerts, team access.')",
                "print('Same span format for both; switch by changing the OTel endpoint.')",
            ],
        ),
    ],

    run_md=[
        "End-to-end with tracing on.",
    ],
    run_code=[
        "q = 'What is the difference between Rc and Arc?'",
        "ans, _ = answer_question(q)",
        "print('=== Answer ===')",
        "print(ans)",
        "print()",
        "print('In a real session with Phoenix enabled, the trace tree at http://localhost:6006')",
        "print('would show the embed call, the search call, and the chat call as nested spans.')",
    ],

    comparison_md=[
        "Phoenix is a measurement tool, not a pipeline. The comparison is between debugging with traces and debugging without.",
    ],
    comparison_code=[
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'approach': 'no tracing', 'debug_workflow': 'rerun, add print, rerun, hope'},",
        "    {'approach': 'phoenix', 'debug_workflow': 'open localhost:6006, read tree, fix'},",
        "])",
    ],

    tuning_md=[
        "Seven knobs in priority order:",
        "",
        "1. **Where Phoenix runs.** Local for dev, hosted Arize for production. Both speak the same OTel format.",
        "2. **What gets traced.** OpenInference auto-instruments LangChain, LlamaIndex, OpenAI SDKs.",
        "3. **Sampling.** At production scale, sample 1-10 % of traces to keep costs manageable.",
        "4. **Retention.** Phoenix keeps traces in memory by default. Persist to disk for longer history.",
        "5. **Integration with evals.** Convert failing traces into eval datasets for regression testing.",
        "6. **PII redaction.** Redact sensitive content before spans are emitted; some Phoenix integrations support hooks for this.",
        "7. **Cross-trace correlation.** Tag spans with conversation IDs so multi-turn agent flows can be reconstructed.",
    ],

    discussion_md=[
        "Four failure modes you'll meet:",
        "",
        "- **Span volume.** A high-traffic system produces millions of spans daily. Sample aggressively.",
        "- **Trace explosion.** Verbose agent loops produce deeply nested traces. Cap depth or summarise long ones.",
        "- **Sensitive data in spans.** Spans capture inputs and outputs verbatim. Redact PII before emitting.",
        "- **Drift across instrumentation.** Frameworks update their auto-instrumentation libraries; spans change shape without warning. Lock versions.",
        "",
        "Compose with RAGAS / DeepEval for metric computation on traces. Compose with Lynx guardrails (Recipe 40) — guardrail failures become spans you can investigate.",
    ],
)


# =============================================================================
# Hallucination Guardrails with Lynx
# =============================================================================
LYNX = Recipe(
    path="recipes/09-evaluation-and-production/hallucination-guardrails-lynx.ipynb",
    title="Lynx Guardrails — Block Hallucinated Answers Before They Ship",
    category="evaluation-and-production",
    corpus_filter="wikipedia-superconductors",

    theory_problem=[
        "RAG systems hallucinate. Even with good retrieval, the LLM occasionally fabricates details, mixes up entities, or asserts claims the passages didn't make. In low-stakes UIs that's annoying; in customer-facing or regulated contexts it's a real liability.",
        "Guardrails sit between the LLM and the user. They read the answer and the contexts, score how grounded the answer is, and block ungrounded answers. Patronus Lynx is the open-weight small model trained specifically for this scoring task. The cookbook implements a Lynx-style judge using a small fast LLM; production teams swap in the real Lynx model.",
    ],
    theory_origin=[
        "Patronus AI released Lynx in mid-2024 as a small (~8B) open-weight model fine-tuned for hallucination detection. The training data came from HaluBench and similar grounding benchmarks. The model outputs a binary grounded/not-grounded score and is fast enough for production gates without disturbing latency budgets.",
        "By 2026 hallucination guardrails are table stakes for production RAG. Lynx is the open-source baseline; commercial options include NeMo Guardrails (NVIDIA), Guardrails AI, and bespoke fine-tuned models. The pattern below is universal: judge the answer against the context, block on low grounding, and refuse with a useful message rather than ship the hallucination.",
    ],
    theory_landscape=[
        "Three layers of safety in production RAG that cookbook recipes cover:",
        "",
        "- **Lynx guardrails (this recipe).** Grounding check on the answer just before it ships.",
        "- **NeMo Guardrails / Guardrails AI.** Content moderation, structured output validation, prompt-injection detection.",
        "- **Self-RAG / CRAG (Recipes 24, 25).** Upstream filtering of passages before generation.",
        "",
        "All three compose. Lynx is the final gate; Self-RAG and CRAG prevent bad inputs from reaching it; NeMo / Guardrails AI handle modalities Lynx doesn't (PII, prompt injection, schema validation). Production-grade safety stacks all three.",
    ],
    theory_when_to_use=[
        "Use guardrails in any production RAG that customers, regulators, or downstream automation will see. The cost is one extra small-LLM call per answer; the benefit is catching the worst failure mode in production RAG.",
        "Skip it for internal exploratory tools. The guardrail latency is wasted.",
        "Skip it when grounding isn't the main concern. Some content-moderation use cases want different guardrails (PII detection, prompt injection).",
    ],
    theory_intuition=[
        "Four intuitions:",
        "",
        "**Judging is cheaper than generating.** A small fast LLM scoring \"is this grounded?\" is much cheaper than the answer LLM. Even at high volume, the guardrail layer is affordable.",
        "",
        "**The judge needs both answer and context.** Without context, the judge can't tell what's grounded. The cookbook pattern passes both.",
        "",
        "**Refuse-on-fail.** When the guardrail fails, refuse with a fixed message rather than ship the answer. \"I don't have enough evidence to answer confidently\" is better than a hallucinated answer.",
        "",
        "**Tune for false positives.** A too-strict guardrail refuses correct answers. Calibrate on a labelled grounding set.",
    ],

    architecture_mermaid="""
flowchart LR
  Q[Question] --> R[Retrieve]
  R --> A[Answer LLM]
  A --> G[Guardrail<br/>Lynx judge]
  R --> G
  G --> D{Grounded?}
  D -->|yes| OUT[Ship answer]
  D -->|no| REF[Refuse politely]
""",

    references=[
        Reference(
            title="Patronus Lynx model card",
            url="https://huggingface.co/PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct",
            kind="repo",
            note="The open-weight Lynx model.",
        ),
        Reference(
            title="HaluBench benchmark",
            url="https://huggingface.co/datasets/PatronusAI/HaluBench",
            kind="repo",
            note="The benchmark Lynx was trained for.",
        ),
        Reference(
            title="NeMo Guardrails",
            url="https://github.com/NVIDIA/NeMo-Guardrails",
            kind="repo",
            note="NVIDIA's alternative.",
        ),
        Reference(
            title="Guardrails AI",
            url="https://www.guardrailsai.com/",
            kind="docs",
            note="Validation-focused guardrail framework.",
        ),
        Reference(
            title="Self-RAG (Recipe 24)",
            url="https://arxiv.org/abs/2310.11511",
            kind="paper",
            note="Upstream filter that composes with Lynx.",
        ),
        Reference(
            title="Anthropic Responsible Scaling Policy",
            url="https://www.anthropic.com/news/anthropics-responsible-scaling-policy",
            kind="blog",
            note="Production-safety context.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Build a pipeline",
                "",
                "Standard Wikipedia superconductors RAG. We'll wrap it with Lynx-style guardrails.",
            ],
            code=[
                "from cookbook.corpora import load_wikipedia_superconductors",
                "from cookbook.chunkers import sentence_window",
                "from cookbook.stores import QdrantBackend",
                "",
                "docs = list(load_wikipedia_superconductors())",
                "chunks = sentence_window(docs, sentences_per_chunk=4)",
                "vectors = client.embed([c.text for c in chunks])",
                "store = QdrantBackend('lynx', dim=len(vectors[0]))",
                "store.add([c.text for c in chunks], vectors, ids=[c.chunk_id for c in chunks])",
                "print('Pipeline ready.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 2 — Build the guardrail judge",
                "",
                "The judge takes (answer, contexts) and returns a score 0..1 plus a verdict. The cookbook uses a prompt-engineered judge; production swaps in the real Lynx model.",
            ],
            code=[
                "import re",
                "",
                "JUDGE = (",
                "    'You are a hallucination detector. Score 0.0 if the answer contradicts or is unsupported by the passages, '",
                "    '1.0 if every claim is in the passages. Reply with just the number.\\n'",
                "    'Passages:\\n{ctx}\\nAnswer:\\n{ans}'",
                ")",
                "",
                "def guard(answer: str, contexts: list[str]) -> tuple[float, str]:",
                "    raw = client.chat(JUDGE.format(ctx='\\n\\n'.join(contexts), ans=answer))",
                "    m = re.search(r'[01](?:\\.\\d+)?', raw)",
                "    score = float(m.group()) if m else 0.0",
                "    verdict = 'GROUNDED' if score >= 0.5 else 'BLOCK'",
                "    return score, verdict",
                "",
                "score, verdict = guard(",
                "    'Superconductors expel magnetic fields completely.',",
                "    ['The Meissner effect is the complete expulsion of magnetic flux from a superconductor.'],",
                ")",
                "print(f'score={score}  verdict={verdict}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Test the guardrail on a hallucinated answer",
                "",
                "Pair a contextually-correct passage with an incorrect answer. The guardrail should fire.",
            ],
            code=[
                "score, verdict = guard(",
                "    'Superconductors were invented by Heike Kamerlingh Onnes in 1842.',  # wrong year",
                "    ['Heike Kamerlingh Onnes discovered superconductivity in mercury in 1911.'],",
                ")",
                "print(f'score={score}  verdict={verdict}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Build the guarded pipeline",
                "",
                "Standard RAG + Lynx gate. On BLOCK, refuse politely.",
            ],
            code=[
                "def guarded(question: str) -> tuple[str, list[str]]:",
                "    qv = client.embed([question])[0]",
                "    hits = store.search(qv, top_k=5)",
                "    ctx = [h.text for h in hits]",
                "    a = client.chat('Use only these passages.\\n' + '\\n\\n'.join(ctx) + f'\\nQ: {question}\\nA:')",
                "    score, verdict = guard(a, ctx)",
                "    if verdict == 'BLOCK':",
                "        return 'I do not have enough grounded evidence to answer that confidently.', ctx",
                "    return a, ctx",
                "",
                "for q in [",
                "    'What is the Meissner effect?',",
                "    'Who personally invented the iPhone superconductor in 1842?',  # nonsense",
                "]:",
                "    a, _ = guarded(q)",
                "    print(f'Q: {q}')",
                "    print(f'  -> {a}')",
                "    print()",
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
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    return guarded(question)",
                "",
                "ans, _ = answer_question('What is BCS theory?')",
                "print(ans[:200])",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — score distribution across a battery",
                "",
                "On a labelled set you'd plot the distribution and pick a threshold. Here we approximate.",
            ],
            code=[
                "scores = []",
                "for q in [",
                "    'What is the Meissner effect?',",
                "    'Who discovered superconductivity?',",
                "    'What is the boiling point of liquid nitrogen?',",
                "    'When was YBCO discovered?',",
                "]:",
                "    qv = client.embed([q])[0]",
                "    hits = store.search(qv, top_k=5)",
                "    ctx = [h.text for h in hits]",
                "    a = client.chat('Use only these.\\n' + '\\n\\n'.join(ctx) + f'\\nQ: {q}\\nA:')",
                "    s, v = guard(a, ctx)",
                "    print(f'  score={s:.2f}  verdict={v}  q={q[:40]}')",
                "    scores.append(s)",
                "import statistics",
                "print(f'mean grounded score: {statistics.mean(scores):.2f}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — false-positive case",
                "",
                "Sometimes a correct answer scores low because the model paraphrases heavily. We illustrate.",
            ],
            code=[
                "score, v = guard(",
                "    'Resistance drops to zero below a certain temperature.',",
                "    ['Superconductivity is the property of zero electrical resistance below a critical temperature.'],",
                ")",
                "print(f'score={score} verdict={v}')",
                "print('Even a clear paraphrase can score low. Tune the threshold accordingly.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — cost",
                "",
                "Guardrails add one LLM call per answer. With Lynx-8B locally, that's a few milliseconds; with a hosted small model, a few cents per thousand answers.",
            ],
            code=[
                "print('Per-answer guardrail cost:')",
                "print('  - 1 small-LLM call.')",
                "print('  - ~200-500 input tokens (passages + answer).')",
                "print('  - At GPT-4o-mini pricing: well under a cent per answer.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — compose with Self-RAG",
                "",
                "If Self-RAG (Recipe 24) already filtered passages, the answer has less reason to hallucinate. Lynx becomes a safety net rather than a primary gate.",
            ],
            code=[
                "print('Stack: Self-RAG (filter) -> answer -> Lynx (grounding gate) -> ship.')",
                "print('Self-RAG reduces the false-block rate on Lynx by keeping passages relevant.')",
            ],
        ),
    ],

    run_md=[
        "End-to-end with guardrails on.",
    ],
    run_code=[
        "ans, _ = answer_question('In what year did Bednorz and Mueller discover their copper-oxide superconductor?')",
        "print('=== Guarded answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Ungrounded vs guarded. On the nonsense question, ungrounded RAG hallucinates and guarded RAG refuses.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'Who personally invented the iPhone superconductor in 1842?'",
        "base = vanilla_pipeline(q, corpus='wikipedia-superconductors', top_k=5)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'vanilla', 'preview': base.answer[:160]},",
        "    {'pipeline': 'guarded', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Seven knobs in priority order:",
        "",
        "1. **Threshold.** 0.5 is the cookbook default. Higher refuses more (safer); lower lets more through (more useful).",
        "2. **Judge model.** Real Lynx-8B is recommended; prompt-engineered judges work but are noisier.",
        "3. **Refusal message.** Make it actionable. \"I don't have enough evidence to answer that confidently\" is better than \"sorry I can't help\".",
        "4. **Calibration set.** Tune threshold on a labelled set; don't pick a number out of the air.",
        "5. **Logging.** Log every BLOCK; use them to improve retrieval upstream.",
        "6. **Composition.** Stack with Self-RAG (Recipe 24) upstream and content moderation downstream.",
        "7. **Per-segment thresholds.** Different domains have different baseline grounding scores; calibrate per segment if your traffic is heterogeneous.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **False blocks on paraphrase.** Lynx sometimes flags correct paraphrases as ungrounded. Tune threshold or use entailment-style judges.",
        "- **False passes on subtle hallucination.** Small fabrications (a wrong number, a swapped entity) can slip through. Compose with structured-output validation.",
        "- **Cost at scale.** Every answer pays guardrail cost. Sample if budget is tight; gate only high-stakes endpoints.",
        "",
        "Compose with Self-RAG (Recipe 24) upstream to reduce false blocks. Compose with Phoenix (Recipe 39) to debug guardrail decisions. Compose with NeMo Guardrails for content moderation in addition to grounding.",
    ],
)


RECIPES = [RAGAS, DEEPEVAL, PHOENIX, LYNX]
