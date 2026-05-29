# Corpus

Four small, fresh, public-domain document sets — chosen so every technique in the cookbook has something distinct to chew on.

| Folder / file | Bytes (approx) | Source | License |
|---|---|---|---|
| `arxiv-2403-mamba.pdf` | ~3 MB | arXiv 2403.16371 | arXiv non-exclusive |
| `wikipedia-superconductors/*.md` | ~1 MB | Wikipedia REST API | CC BY-SA 4.0 |
| `sec-10k-PLTR-2024.txt` | ~2 MB | SEC EDGAR | Public filing |
| `tech-docs-rust-book/*.md` | ~2 MB | rust-lang/book | CC BY 4.0 |
| `eval/ragas-questions.jsonl` | ~30 KB | hand-authored | Apache-2.0 (this repo) |

Reproduce with:

```bash
uv run python scripts/fetch_corpus.py
uv run python scripts/fetch_corpus.py --verify
```

Nothing in `corpus/_downloads/` is checked into git; everything else here is the canonical material every recipe loads.
