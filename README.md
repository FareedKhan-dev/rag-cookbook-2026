# The RAG Cookbook 2026

> Recipes for production-grade retrieval — forty hands-on chapters covering the state of the art in retrieval-augmented generation, all runnable end-to-end on Nebius AI Studio (or any OpenAI-compatible provider).

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-Apache--2.0-green) ![Docs](https://img.shields.io/badge/docs-mkdocs--material-526CFE) ![Recipes](https://img.shields.io/badge/recipes-40-orange)

---

## Why this exists

Most RAG tutorials stop at "embed a PDF, do cosine search, hand it to GPT." That worked in 2023. In 2026, production RAG looks completely different: late chunking, contextual retrieval, multi-vector late interaction, listwise rerankers, speculative drafters, agentic workflows over MCP servers, dual-level graph indexes, page-as-image vision retrieval, and compiled DSPy pipelines that out-perform anything a prompt engineer can hand-tune.

This cookbook teaches all of it. Forty self-contained recipes. Every recipe ships with theory, implementation, output discussion, and an evaluation harness. Nothing is hidden behind framework magic.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/FareedKhan-dev/rag-cookbook-2026.git
cd rag-cookbook-2026
uv sync                          # or: pip install -e ".[docs,dev]"

# 2. Add your keys (Nebius is the default; OpenAI/Anthropic/Groq optional)
cp .env.example .env
$EDITOR .env

# 3. Download the public-domain corpus (one-time, ~150 MB)
uv run python scripts/fetch_corpus.py

# 4. Launch a recipe
uv run jupyter lab recipes/01-foundations/vanilla-pipeline.ipynb
```

Want to read the cookbook in your browser instead? `uv run mkdocs serve` → http://localhost:8000

## Recipe Index

| # | Category | Recipe | Idea | Best for |
|---|---|---|---|---|
|  1 | Foundations | [cookbook-tour](recipes/01-foundations/cookbook-tour.ipynb) | Provider switcher, corpus, tracing | First read |
|  2 | Foundations | [vanilla-pipeline](recipes/01-foundations/vanilla-pipeline.ipynb) | The baseline you'll beat 38 times | Mental model |
|  3 | Foundations | [embedding-zoo](recipes/01-foundations/embedding-zoo.ipynb) | Voyage / BGE-M3 / Nebius / Qwen3 head-to-head | Picking a model |
|  4 | Foundations | [chunk-size-sensitivity](recipes/01-foundations/chunk-size-sensitivity.ipynb) | Sweep sizes, measure recall | Tuning |
|  5 | Chunking | [semantic-boundary-splitting](recipes/02-chunking-and-indexing/semantic-boundary-splitting.ipynb) | Split at embedding-distance breakpoints | Mixed-topic docs |
|  6 | Chunking | [late-chunking-jina](recipes/02-chunking-and-indexing/late-chunking-jina.ipynb) | Embed first, chunk after | Long context |
|  7 | Chunking | [contextual-retrieval-anthropic](recipes/02-chunking-and-indexing/contextual-retrieval-anthropic.ipynb) | LLM-written "where this fits" headers | Highest ROI tweak |
|  8 | Chunking | [proposition-decomposition](recipes/02-chunking-and-indexing/proposition-decomposition.ipynb) | Atomic-claim units | Dense factual recall |
|  9 | Chunking | [sentence-window-and-parent-child](recipes/02-chunking-and-indexing/sentence-window-and-parent-child.ipynb) | Small to find, big to read | Precision + synthesis |
| 10 | Chunking | [raptor-trees](recipes/02-chunking-and-indexing/raptor-trees.ipynb) | Recursive summary tree | Multi-hop reasoning |
| 11 | Chunking | [document-summary-routing](recipes/02-chunking-and-indexing/document-summary-routing.ipynb) | Doc summaries → sub-chunks | Heterogeneous corpora |
| 12 | Chunking | [matryoshka-coarse-to-fine](recipes/02-chunking-and-indexing/matryoshka-coarse-to-fine.ipynb) | MRL embeddings, two-stage | Low-latency search |
| 13 | Query | [hypothetical-document-embeddings](recipes/03-query-transformation/hypothetical-document-embeddings.ipynb) | HyDE | Zero-shot retrieval |
| 14 | Query | [multi-query-rag-fusion](recipes/03-query-transformation/multi-query-rag-fusion.ipynb) | N rewrites + RRF | Recall boost |
| 15 | Query | [step-back-abstraction](recipes/03-query-transformation/step-back-abstraction.ipynb) | Abstract then ground | Reasoning questions |
| 16 | Query | [sub-question-decomposition](recipes/03-query-transformation/sub-question-decomposition.ipynb) | Tree of sub-queries | Complex queries |
| 17 | Query | [semantic-router](recipes/03-query-transformation/semantic-router.ipynb) | Classify and dispatch | Multi-index systems |
| 18 | Retrieval | [hybrid-dense-plus-bm25](recipes/04-retrieval/hybrid-dense-plus-bm25.ipynb) | Dense + BM25 + RRF on Qdrant | Real workloads |
| 19 | Retrieval | [colbert-late-interaction](recipes/04-retrieval/colbert-late-interaction.ipynb) | Token-level MaxSim | Highest accuracy |
| 20 | Retrieval | [auto-retrieval-with-metadata](recipes/04-retrieval/auto-retrieval-with-metadata.ipynb) | LLM extracts filters | Structured corpora |
| 21 | Retrieval | [mmr-diversity](recipes/04-retrieval/mmr-diversity.ipynb) | Penalize redundancy | Diverse top-k |
| 22 | Reranking | [cross-encoder-rerank](recipes/05-reranking-and-fusion/cross-encoder-rerank.ipynb) | bge / Cohere / Jina | Quality lift |
| 23 | Reranking | [listwise-llm-rerank](recipes/05-reranking-and-fusion/listwise-llm-rerank.ipynb) | RankGPT-style | When budget allows |
| 24 | Agentic | [self-reflective-retrieval](recipes/06-adaptive-and-agentic/self-reflective-retrieval.ipynb) | Self-RAG reflection tokens | Selective retrieval |
| 25 | Agentic | [corrective-retrieval-with-web-fallback](recipes/06-adaptive-and-agentic/corrective-retrieval-with-web-fallback.ipynb) | CRAG | Robust answers |
| 26 | Agentic | [adaptive-routing-by-question-class](recipes/06-adaptive-and-agentic/adaptive-routing-by-question-class.ipynb) | Adaptive RAG | Mixed query types |
| 27 | Agentic | [speculative-rag-drafter-verifier](recipes/06-adaptive-and-agentic/speculative-rag-drafter-verifier.ipynb) | Small drafter, big verifier | Latency-bound apps |
| 28 | Agentic | [langgraph-agentic-rag](recipes/06-adaptive-and-agentic/langgraph-agentic-rag.ipynb) | Stateful cyclic graph | Tool-using agents |
| 29 | Agentic | [mcp-tool-retrieval](recipes/06-adaptive-and-agentic/mcp-tool-retrieval.ipynb) | Retrieval over MCP servers | Modular tools |
| 30 | Agentic | [dspy-compiled-rag](recipes/06-adaptive-and-agentic/dspy-compiled-rag.ipynb) | Optimize, don't prompt-engineer | Repeatable pipelines |
| 31 | Graph | [microsoft-graphrag-pipeline](recipes/07-graph-and-memory/microsoft-graphrag-pipeline.ipynb) | Communities + global/local | Global questions |
| 32 | Graph | [lightrag-dual-level](recipes/07-graph-and-memory/lightrag-dual-level.ipynb) | LightRAG | Cheap incremental KG |
| 33 | Graph | [hipporag-pagerank-memory](recipes/07-graph-and-memory/hipporag-pagerank-memory.ipynb) | PageRank over KG | Multi-hop facts |
| 34 | Graph | [mem0-long-term-memory](recipes/07-graph-and-memory/mem0-long-term-memory.ipynb) | Persistent agent memory | Chat assistants |
| 35 | Multimodal | [colpali-page-as-image](recipes/08-multimodal/colpali-page-as-image.ipynb) | VLM patch embeddings of PDFs | No-OCR pipelines |
| 36 | Multimodal | [vlm-synthesis-over-pages](recipes/08-multimodal/vlm-synthesis-over-pages.ipynb) | ColPali → Qwen-VL | Visual answers |
| 37 | Eval | [ragas-triad-eval](recipes/09-evaluation-and-production/ragas-triad-eval.ipynb) | Faithfulness / relevance / precision | Default metrics |
| 38 | Eval | [deepeval-pytest-ci](recipes/09-evaluation-and-production/deepeval-pytest-ci.ipynb) | Metrics in CI | Regression catch |
| 39 | Eval | [phoenix-tracing-debugging](recipes/09-evaluation-and-production/phoenix-tracing-debugging.ipynb) | End-to-end OTel traces | Debugging |
| 40 | Eval | [hallucination-guardrails-lynx](recipes/09-evaluation-and-production/hallucination-guardrails-lynx.ipynb) | Patronus Lynx | Production safety |

## Stack

| Layer | Default | Notes |
|---|---|---|
| LLM provider | **Nebius AI Studio** | Switchable to OpenAI, Anthropic, Groq, OpenRouter, Together via one env var |
| Provider abstraction | **LiteLLM** | Single `LLMClient` wraps chat / embed / rerank |
| Embeddings | voyage-3-large, BGE-M3, Qwen3-Embedding-8B, Nebius | Compared head-to-head in recipe 3 |
| Rerankers | Cohere Rerank 4, bge-reranker-v2-m3, jina-reranker-v3 | Recipe 22 |
| Vector store | **Qdrant** in-memory by default; LanceDB and Chroma also wrapped | Native hybrid + multi-vector |
| Graph store | NetworkX (Neo4j optional) | Recipes 31–34 |
| Agent frameworks | LangGraph, LlamaIndex Workflows, DSPy | One recipe each |
| Tracing | **Arize Phoenix** (local) + LangSmith (agentic) | One-line `init_tracing()` |
| Evaluation | RAGAS + DeepEval | Plus Patronus Lynx guardrails |
| Documentation | MkDocs Material + mkdocs-jupyter | Served at `mkdocs serve` |

## Corpus

Four fresh, public-domain document sets — chosen to exercise every technique without overlapping any other RAG tutorial:

1. A recent arXiv paper on state-space models (not "Attention Is All You Need")
2. A Wikipedia subset on superconductors (~50 markdown pages)
3. Palantir's 2024 SEC 10-K (text extracted, public filing)
4. A curated chapter set from *The Rust Programming Language* (CC-BY)

Fetch reproducibly with `uv run python scripts/fetch_corpus.py --verify`.

## Notebook structure

Every recipe follows the same six sections:

1. **What & Why** — original paper / blog, when this technique helps, when it doesn't
2. **Setup** — provider switch, tracing, corpus load (one cell)
3. **Build the Pipeline** — the implementation, broken into small reviewable cells
4. **Run on Corpus** — execute against one of the four corpora
5. **Discuss the Output** — markdown analysis of what came out, failure modes, observations
6. **Evaluate** — RAGAS / DeepEval metrics plus one custom assertion

## Contributing

Issues and pull requests welcome. New techniques are added under the appropriate category as `<technique-slug>.ipynb`, following the six-section template above. See [`docs/theory/`](docs/theory) for the conceptual framing you should match.

## Citing

If this cookbook helps your research or teaching, please cite it via the [`CITATION.cff`](CITATION.cff) file.

## License

[Apache-2.0](LICENSE) © 2026 Fareed Khan.
