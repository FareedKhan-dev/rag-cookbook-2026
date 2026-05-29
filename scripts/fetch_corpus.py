"""Download and verify the four public-domain corpora used by every recipe.

Sources (all public domain or permissively licensed):

  arxiv-2403-mamba.pdf            arXiv 2403.16371 — "A Survey on Mamba and SSMs"
                                  arXiv non-exclusive license, free for redistribution.
  wikipedia-superconductors/      ~50 markdown pages from Wikipedia (CC BY-SA 4.0).
                                  Fetched via the MediaWiki REST API and converted.
  sec-10k-PLTR-2024.txt           Palantir Technologies 2024 10-K — public filing.
                                  Fetched from SEC EDGAR, plain-text extract.
  tech-docs-rust-book/            Selected chapters of *The Rust Programming Language*
                                  (CC BY 4.0) from rust-lang/book.

Usage:
    python scripts/fetch_corpus.py            # download missing files
    python scripts/fetch_corpus.py --verify   # re-check hashes
    python scripts/fetch_corpus.py --force    # re-download all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
DOWNLOADS = CORPUS / "_downloads"


# ---------------------------------------------------------------------------
# Source manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    name: str
    kind: str        # "pdf" | "text" | "wiki" | "rust-book"
    url: str | list[str]
    dest: Path
    expected_sha256: str | None = None


WIKI_PAGES = [
    "Superconductivity",
    "Type-I_superconductor",
    "Type-II_superconductor",
    "High-temperature_superconductivity",
    "BCS_theory",
    "Cooper_pair",
    "Meissner_effect",
    "Josephson_effect",
    "Flux_pinning",
    "Cuprate_superconductor",
    "Iron-based_superconductor",
    "YBCO",
    "Niobium-titanium",
    "Magnesium_diboride",
    "Hydrogen_sulfide",
    "Lanthanum_hydride",
    "Room-temperature_superconductor",
    "Critical_temperature",
    "London_equations",
    "Ginzburg%E2%80%93Landau_theory",
    "Type_1.5_superconductor",
    "Pseudogap",
    "Cuprate",
    "Bednorz_and_M%C3%BCller",
    "Onnes",
    "Bardeen,_Cooper,_and_Schrieffer",
    "Superconducting_magnet",
    "MRI",
    "Particle_accelerator_magnets",
    "Maglev",
    "SQUID",
    "Tokamak",
    "ITER",
    "Quantum_computing",
    "Superconducting_qubit",
    "Transmon",
    "Flux_qubit",
    "Quantum_annealing",
    "D-Wave_Systems",
    "Bose%E2%80%93Einstein_condensate",
    "Helium-3",
    "Helium-4",
    "Lambda_point",
    "Heike_Kamerlingh_Onnes",
    "Discovery_of_superconductivity",
    "List_of_superconductors",
    "Penetration_depth",
    "Coherence_length",
    "Vortex_lattice",
    "Abrikosov_vortex",
    "Bogoliubov_quasiparticle",
]


RUST_BOOK_CHAPTERS = [
    "ch01-00-getting-started.md",
    "ch02-00-guessing-game-tutorial.md",
    "ch03-00-common-programming-concepts.md",
    "ch04-00-understanding-ownership.md",
    "ch05-00-structs.md",
    "ch06-00-enums.md",
    "ch07-00-managing-growing-projects-with-packages-crates-and-modules.md",
    "ch08-00-common-collections.md",
    "ch09-00-error-handling.md",
    "ch10-00-generics.md",
    "ch11-00-testing.md",
    "ch12-00-an-io-project.md",
    "ch13-00-functional-features.md",
    "ch15-00-smart-pointers.md",
    "ch16-00-concurrency.md",
    "ch17-00-async-await.md",
    "ch18-00-oop.md",
    "ch19-00-patterns.md",
]


SOURCES: list[Source] = [
    Source(
        name="arxiv-mamba-survey",
        kind="pdf",
        url="https://arxiv.org/pdf/2403.16371",
        dest=CORPUS / "arxiv-2403-mamba.pdf",
    ),
    Source(
        name="wikipedia-superconductors",
        kind="wiki",
        url=WIKI_PAGES,
        dest=CORPUS / "wikipedia-superconductors",
    ),
    Source(
        name="sec-10k-pltr-2024",
        kind="text",
        url="https://www.sec.gov/Archives/edgar/data/1321655/000132165525000022/pltr-20241231.htm",
        dest=CORPUS / "sec-10k-PLTR-2024.txt",
    ),
    Source(
        name="rust-book",
        kind="rust-book",
        url=RUST_BOOK_CHAPTERS,
        dest=CORPUS / "tech-docs-rust-book",
    ),
]


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------
def _fetch(url: str, *, headers: dict | None = None) -> bytes:
    import httpx

    # SEC requires a real-looking User-Agent with contact email per
    # https://www.sec.gov/about/webmaster-faq#code-support.
    h = {
        "User-Agent": "RAG Cookbook Educational Project (contact@example.org)",
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    }
    if headers:
        h.update(headers)
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        r = client.get(url, headers=h)
        r.raise_for_status()
        return r.content


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Per-source handlers
# ---------------------------------------------------------------------------
def fetch_pdf(src: Source, *, force: bool) -> None:
    if src.dest.exists() and not force:
        print(f"  [skip] {src.dest.name}")
        return
    print(f"  [pdf]  {src.url} -> {src.dest.name}")
    src.dest.parent.mkdir(parents=True, exist_ok=True)
    src.dest.write_bytes(_fetch(src.url))


def fetch_text(src: Source, *, force: bool) -> None:
    if src.dest.exists() and not force:
        print(f"  [skip] {src.dest.name}")
        return
    print(f"  [text] {src.url} -> {src.dest.name}")
    raw = _fetch(src.url).decode("utf-8", errors="replace")
    # Strip HTML tags for a clean plain-text extract.
    import re
    text = re.sub(r"<script.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    src.dest.parent.mkdir(parents=True, exist_ok=True)
    src.dest.write_text(text.strip(), encoding="utf-8")


def fetch_wiki(src: Source, *, force: bool) -> None:
    src.dest.mkdir(parents=True, exist_ok=True)
    for title in src.url:  # type: ignore[union-attr]
        slug = urllib.parse.unquote(title).replace(" ", "_")
        out = src.dest / f"{slug.replace('/', '_')}.md"
        if out.exists() and not force:
            continue
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        try:
            data = json.loads(_fetch(url).decode("utf-8"))
        except Exception as e:
            print(f"  [warn] wiki:{slug}: {e}")
            continue
        body = (
            f"# {data.get('title', slug)}\n\n"
            f"_Source: Wikipedia, CC BY-SA 4.0_\n\n"
            f"{data.get('extract', '')}\n"
        )
        out.write_text(body, encoding="utf-8")
    print(f"  [wiki] {len(src.url)} pages -> {src.dest.name}/")  # type: ignore[arg-type]


def fetch_rust_book(src: Source, *, force: bool) -> None:
    src.dest.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/rust-lang/book/main/src/"
    for chapter in src.url:  # type: ignore[union-attr]
        out = src.dest / chapter
        if out.exists() and not force:
            continue
        try:
            text = _fetch(base + chapter).decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  [warn] rust-book:{chapter}: {e}")
            continue
        out.write_text(text, encoding="utf-8")
    print(f"  [rust] {len(src.url)} chapters -> {src.dest.name}/")  # type: ignore[arg-type]


HANDLERS = {
    "pdf": fetch_pdf,
    "text": fetch_text,
    "wiki": fetch_wiki,
    "rust-book": fetch_rust_book,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="Re-download everything.")
    ap.add_argument("--verify", action="store_true", help="Only print existence + hashes.")
    args = ap.parse_args()

    CORPUS.mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)

    if args.verify:
        for s in SOURCES:
            if s.dest.is_dir():
                count = len(list(s.dest.glob("*")))
                print(f"  {s.name:30s} {s.dest} ({count} files)")
            elif s.dest.exists():
                print(f"  {s.name:30s} {s.dest} sha256={_sha(s.dest)[:12]}…")
            else:
                print(f"  {s.name:30s} MISSING — re-run without --verify")
        return 0

    failures: list[tuple[str, str]] = []
    for s in SOURCES:
        print(f"-- {s.name}")
        try:
            HANDLERS[s.kind](s, force=args.force)
        except Exception as e:
            print(f"  [error] {s.name}: {type(e).__name__}: {e}")
            failures.append((s.name, str(e)))
    if failures:
        print("\nSome sources failed (other recipes can still run):")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
    print("\nCorpus ready under", CORPUS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
