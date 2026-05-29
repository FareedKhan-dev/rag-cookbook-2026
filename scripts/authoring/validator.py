"""Depth validator. Fails loudly when a recipe drops below bar.

The thresholds match the v2 depth plan. Loosen them only with very good
reason — they exist so authoring drift across 40 notebooks is detected
mechanically, not by reading every diff.
"""
from __future__ import annotations

from .schema import Recipe


class RecipeValidationError(ValueError):
    """Raised when a recipe fails the depth check."""


def _wordcount(lines: list[str]) -> int:
    return sum(len(line.split()) for line in lines)


def validate(r: Recipe) -> list[str]:
    """Return a list of problems (empty on success).

    The build script raises if the list is non-empty.
    """
    problems: list[str] = []

    # theory subsections must each have at least ~50 words
    theory_sections = {
        "theory_problem": r.theory_problem,
        "theory_origin": r.theory_origin,
        "theory_landscape": r.theory_landscape,
        "theory_when_to_use": r.theory_when_to_use,
        "theory_intuition": r.theory_intuition,
    }
    for name, content in theory_sections.items():
        wc = _wordcount(content)
        if wc < 40:
            problems.append(f"{r.path}: {name} is only {wc} words (need ≥ 40)")

    total_theory_wc = sum(_wordcount(c) for c in theory_sections.values())
    if total_theory_wc < 420:
        problems.append(
            f"{r.path}: theory total is {total_theory_wc} words (need ≥ 420)"
        )

    if "diagram" not in r.skip_sections and not r.architecture_mermaid.strip():
        problems.append(f"{r.path}: architecture_mermaid is empty")

    if "references" not in r.skip_sections and len(r.references) < 5:
        problems.append(
            f"{r.path}: only {len(r.references)} references (need ≥ 5)"
        )

    if len(r.cells) < 5:
        problems.append(f"{r.path}: only {len(r.cells)} build steps (need ≥ 5)")

    if "inspection" not in r.skip_sections and len(r.inspection_cells) < 3:
        problems.append(
            f"{r.path}: only {len(r.inspection_cells)} inspection cells (need ≥ 3)"
        )

    if "comparison" not in r.skip_sections and not r.comparison_code:
        problems.append(f"{r.path}: comparison_code is empty")

    if "tuning" not in r.skip_sections and _wordcount(r.tuning_md) < 100:
        problems.append(
            f"{r.path}: tuning_md is only {_wordcount(r.tuning_md)} words (need ≥ 100)"
        )

    if _wordcount(r.discussion_md) < 80:
        problems.append(
            f"{r.path}: discussion_md is only {_wordcount(r.discussion_md)} words (need ≥ 80)"
        )

    return problems


__all__ = ["validate", "RecipeValidationError"]
