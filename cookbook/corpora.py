"""Loaders for the four cookbook corpora.

Every recipe pulls its documents through one of the loaders here. The actual
files live under `corpus/` and are downloaded reproducibly by
`scripts/fetch_corpus.py`. The loaders return iterables of `Document` dicts
with the schema below so every vector-store backend, chunker, and evaluator
speaks the same language.

Schema:
    {
        "doc_id": str,           # unique within the corpus
        "source": str,           # human-readable origin
        "text": str,             # plain text content
        "metadata": dict,        # arbitrary; commonly {section, page, url}
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"


@dataclass(frozen=True)
class Document:
    """One unit of corpus material — page, section, or whole file."""

    doc_id: str
    source: str
    text: str
    metadata: dict


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------
def load_arxiv_mamba() -> Iterator[Document]:
    """A recent arXiv survey of state-space sequence models.

    Returns one Document per page. Used by recipes that want a dense, technical
    long document to chunk and reason over.
    """
    pdf = CORPUS_DIR / "arxiv-2403-mamba.pdf"
    yield from _yield_pdf_pages(pdf, source="arxiv:2403-mamba-survey")


def load_wikipedia_superconductors() -> Iterator[Document]:
    """About 50 markdown pages on superconductivity from Wikipedia.

    One Document per markdown file. Good for graph-RAG and multi-hop queries.
    """
    folder = CORPUS_DIR / "wikipedia-superconductors"
    for md in sorted(folder.glob("*.md")):
        yield Document(
            doc_id=f"wiki:{md.stem}",
            source="wikipedia:superconductors",
            text=md.read_text(encoding="utf-8"),
            metadata={"title": md.stem.replace("_", " ")},
        )


def load_sec_10k() -> Iterator[Document]:
    """Palantir Technologies 2024 SEC 10-K, plain text extract.

    Yields one Document per section heading. Good for hybrid search and
    sub-question decomposition (the document is full of numbered items and
    tables).
    """
    txt = CORPUS_DIR / "sec-10k-PLTR-2024.txt"
    yield from _yield_text_sections(txt, source="sec:PLTR-2024-10K")


def load_rust_book() -> Iterator[Document]:
    """A curated subset of *The Rust Programming Language* (CC-BY).

    One Document per chapter, with the chapter heading in metadata. Good for
    code-aware chunking and step-back reasoning.
    """
    folder = CORPUS_DIR / "tech-docs-rust-book"
    for md in sorted(folder.glob("*.md")):
        title_line = md.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
        yield Document(
            doc_id=f"rust-book:{md.stem}",
            source="cc-by:rust-book",
            text=md.read_text(encoding="utf-8"),
            metadata={"chapter": md.stem, "title": title_line},
        )


def load_eval_questions() -> list[dict]:
    """The ~80-pair hand-curated evaluation set used by recipes 37–40."""
    qa = CORPUS_DIR / "eval" / "ragas-questions.jsonl"
    return [json.loads(line) for line in qa.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _yield_pdf_pages(pdf: Path, *, source: str) -> Iterator[Document]:
    import fitz  # pymupdf

    doc = fitz.open(pdf)
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if not text.strip():
                continue
            yield Document(
                doc_id=f"{source}#p{i}",
                source=source,
                text=text,
                metadata={"page": i, "n_pages": len(doc)},
            )
    finally:
        doc.close()


def _yield_text_sections(path: Path, *, source: str) -> Iterator[Document]:
    raw = path.read_text(encoding="utf-8")
    # Section split on lines that look like "Item 1." or "PART I" — common in 10-Ks.
    import re

    pattern = re.compile(r"(?m)^(?:Item\s+\d+[A-Z]?\.|PART\s+[IVX]+|^[A-Z][A-Z\s]{6,}$)")
    matches = list(pattern.finditer(raw))
    if not matches:
        yield Document(
            doc_id=f"{source}#all",
            source=source,
            text=raw,
            metadata={},
        )
        return
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        heading = m.group().strip()
        body = raw[start:end].strip()
        yield Document(
            doc_id=f"{source}#sec{idx:03d}",
            source=source,
            text=body,
            metadata={"heading": heading, "section_index": idx},
        )


__all__ = [
    "Document",
    "load_arxiv_mamba",
    "load_wikipedia_superconductors",
    "load_sec_10k",
    "load_rust_book",
    "load_eval_questions",
]
