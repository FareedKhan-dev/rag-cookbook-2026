"""Recipe data shapes used by the renderer.

A `Recipe` is the full specification of one notebook. The renderer walks its
fields in a fixed order to emit cells; the validator checks that the
required depth is met before writing.

Design philosophy:
    - Markdown is `list[str]`, joined with `\n` on emit. Lets recipes
      write multi-paragraph theory as a Python list of lines.
    - Code is `list[str]`, joined with `\n` on emit. Same convention.
    - One `CodeStep` per *logical* step. Each step is narrated BEFORE the
      code (via `lead_md`) and may be followed by `expected_output_md`
      explaining what just printed.
    - Theory has 5 named subsections so every notebook reads the same way:
      problem / origin / landscape / when-to-use / intuition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ReferenceKind = Literal["paper", "blog", "repo", "docs", "video"]


@dataclass(frozen=True)
class Reference:
    """One bibliography entry — paper, blog post, repo, official docs page."""

    title: str
    url: str
    kind: ReferenceKind = "paper"
    note: str = ""


@dataclass
class CodeStep:
    """One narrated step in the recipe.

    The renderer emits:
        1. A markdown cell from `lead_md`
        2. A code cell from `code`
        3. (optionally) A markdown cell from `expected_output_md`

    `tag` is a short label (`load`, `chunk`, `embed`, `index`, `retrieve`,
    `generate`, `inspect`) used by the quality report.
    """

    lead_md: list[str]
    code: list[str]
    expected_output_md: list[str] = field(default_factory=list)
    tag: str = ""


@dataclass
class Recipe:
    """Full specification of one notebook."""

    # --- identity ---
    path: str
    title: str
    category: str

    # --- theory subsections (each is a list of paragraphs) ---
    theory_problem: list[str]
    theory_origin: list[str]
    theory_landscape: list[str]
    theory_when_to_use: list[str]
    theory_intuition: list[str]

    # --- diagram + references ---
    architecture_mermaid: str
    references: list[Reference]

    # --- granular execution ---
    cells: list[CodeStep]
    inspection_cells: list[CodeStep] = field(default_factory=list)

    # --- run + comparison + tuning + discussion ---
    run_md: list[str] = field(default_factory=list)
    run_code: list[str] = field(default_factory=list)
    comparison_md: list[str] = field(default_factory=list)
    comparison_code: list[str] = field(default_factory=list)
    tuning_md: list[str] = field(default_factory=list)
    discussion_md: list[str] = field(default_factory=list)

    # --- evaluation ---
    corpus_filter: str | None = None

    # --- knobs ---
    extra_setup_lines: list[str] = field(default_factory=list)
    skip_sections: set[str] = field(default_factory=set)
    needs_multimodal: bool = False


__all__ = ["Recipe", "CodeStep", "Reference", "ReferenceKind"]
