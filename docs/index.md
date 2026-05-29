# The RAG Cookbook 2026

Welcome. This site renders every recipe in this repository as a readable page, with the source notebooks just a click away.

If you are new here, start with the [cookbook tour](../recipes/01-foundations/cookbook-tour.ipynb) and then the [vanilla pipeline](../recipes/01-foundations/vanilla-pipeline.ipynb). After that, the [Theory](theory/retrieval-foundations.md) section gives the conceptual map and the [Recipes](../recipes) section is the cookbook itself.

## What you will learn

- How to chunk documents in ways that beat naive fixed-window splitting (recipes 5–12)
- How to rewrite queries so retrieval finds what the user actually meant (recipes 13–17)
- How to combine dense, sparse, and late-interaction retrievers (recipes 18–21)
- How to rerank top-k with cross-encoders and LLMs (recipes 22–23)
- How to build adaptive and agentic pipelines that decide for themselves whether and how to retrieve (recipes 24–30)
- How to ground retrieval in knowledge graphs and long-term memory (recipes 31–34)
- How to retrieve from PDFs as images, no OCR (recipes 35–36)
- How to evaluate, trace, and harden a pipeline for production (recipes 37–40)

## Conventions

- The default provider is **Nebius AI Studio**. Every recipe begins with one cell that lets you flip to OpenAI, Anthropic, Groq, OpenRouter, or Together.
- The default vector store is **Qdrant in-memory**. Recipes that benchmark stores use Qdrant, LanceDB, and Chroma side by side.
- All recipes use the four-corpus mix described on the [Recipes index](../README.md#corpus).

Happy retrieving.
