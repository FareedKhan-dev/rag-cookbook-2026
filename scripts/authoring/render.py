"""Section-driven notebook renderer.

Walks the fixed section order and emits Jupyter cells. Each section
function takes the `Recipe` and returns a `list[dict]` of cells (or `[]`
to skip). The top-level `write_notebook()` joins them, wraps with
notebook-level metadata, and writes a valid nbformat-4 JSON file.

Order of sections (matches the depth plan):
    1. theory          — title + 5 subsections
    2. diagram         — mermaid block
    3. references      — bibliography at the top, for trust
    4. setup           — provider + tracing
    5. build_steps     — narrated CodeStep loop
    6. inspection      — interleaved probes
    7. run             — representative query
    8. comparison      — vanilla vs technique
    9. tuning          — knobs to turn
    10. evaluate       — RAGAS slice
    11. discussion     — when-not-to-use
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import CodeStep, Recipe, Reference

ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Low-level cell helpers
# ---------------------------------------------------------------------------
def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": "\n".join(lines),
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": "\n".join(lines),
        "outputs": [],
        "execution_count": None,
    }


def _refs_to_md(refs: list[Reference]) -> str:
    if not refs:
        return ""
    icons = {
        "paper": "📄",
        "blog": "📝",
        "repo": "💻",
        "docs": "📚",
        "video": "🎥",
    }
    lines = ["## References", ""]
    for r in refs:
        icon = icons.get(r.kind, "📄")
        suffix = f" — {r.note}" if r.note else ""
        lines.append(f"- {icon} [{r.title}]({r.url}){suffix}")
    return "\n".join(lines)


def _emit_step(step: CodeStep) -> list[dict]:
    out: list[dict] = []
    if step.lead_md:
        out.append(md(*step.lead_md))
    if step.code:
        out.append(code(*step.code))
    if step.expected_output_md:
        out.append(md(*step.expected_output_md))
    return out


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
def section_theory(r: Recipe) -> list[dict]:
    parts: list[str] = [f"# {r.title}", "", "## What problem does this solve?", ""]
    parts += r.theory_problem
    parts += ["", "## Where it came from", ""]
    parts += r.theory_origin
    parts += ["", "## Where it fits in the RAG landscape", ""]
    parts += r.theory_landscape
    parts += ["", "## When to use it (and when not to)", ""]
    parts += r.theory_when_to_use
    parts += ["", "## The intuition", ""]
    parts += r.theory_intuition
    return [md(*parts)]


def section_diagram(r: Recipe) -> list[dict]:
    if "diagram" in r.skip_sections or not r.architecture_mermaid:
        return []
    diagram = r.architecture_mermaid.strip()
    # If author already wrapped in a fence, leave alone; else wrap.
    if not diagram.startswith("```"):
        diagram = "```mermaid\n" + diagram + "\n```"
    return [md("## Architecture", "", diagram)]


def section_references(r: Recipe) -> list[dict]:
    if "references" in r.skip_sections or not r.references:
        return []
    return [md(_refs_to_md(r.references))]


def section_setup(r: Recipe) -> list[dict]:
    setup_cells = [
        md(
            "## Setup",
            "",
            "Pick a provider via the `PROVIDER` env var; everything below is provider-agnostic. "
            "The default is Nebius. Tracing is off by default in published notebooks so the "
            "outputs are clean — flip `COOKBOOK_TRACING=phoenix` to send spans to a local Phoenix UI.",
        ),
        code(
            "import os",
            "os.environ.setdefault('PROVIDER', 'nebius')",
            "os.environ.setdefault('COOKBOOK_TRACING', 'off')",
            "",
            "from cookbook.providers import LLMClient",
            "from cookbook.tracing import init_tracing",
            "",
            "client = LLMClient()",
            "print(f'Provider: {client.provider}  |  Chat model: {client.chat_model}')",
            "print(init_tracing())",
        ),
    ]
    if r.extra_setup_lines:
        setup_cells.append(code(*r.extra_setup_lines))
    return setup_cells


def section_build_steps(r: Recipe) -> list[dict]:
    if not r.cells:
        return []
    out: list[dict] = [md("## Build the Pipeline, Step by Step")]
    for step in r.cells:
        out.extend(_emit_step(step))
    return out


def section_inspection(r: Recipe) -> list[dict]:
    if "inspection" in r.skip_sections or not r.inspection_cells:
        return []
    out: list[dict] = [md("## Look Inside")]
    for step in r.inspection_cells:
        out.extend(_emit_step(step))
    return out


def section_run(r: Recipe) -> list[dict]:
    if "run" in r.skip_sections or not r.run_code:
        return []
    return [md("## Run It", "", *r.run_md), code(*r.run_code)]


def section_comparison(r: Recipe) -> list[dict]:
    if "comparison" in r.skip_sections or not r.comparison_code:
        return []
    return [
        md("## Side by Side: Vanilla Baseline vs This Technique", "", *r.comparison_md),
        code(*r.comparison_code),
    ]


def section_tuning(r: Recipe) -> list[dict]:
    if "tuning" in r.skip_sections or not r.tuning_md:
        return []
    return [md("## Knobs to Turn", "", *r.tuning_md)]


def section_evaluate(r: Recipe) -> list[dict]:
    if "evaluate" in r.skip_sections:
        return []
    body = [
        "from cookbook.corpora import load_eval_questions",
        "from cookbook.eval import EvalSample",
        "",
        "qs = load_eval_questions()",
    ]
    if r.corpus_filter:
        body.append(f"qs = [q for q in qs if q['corpus'] == '{r.corpus_filter}']")
    body += [
        "samples = []",
        "for row in qs[:5]:",
        "    answer, contexts = answer_question(row['question'])",
        "    samples.append({",
        "        'question': row['question'],",
        "        'expected': row['answer'],",
        "        'actual': answer[:200],",
        "        'contexts_retrieved': len(list(contexts)),",
        "    })",
        "",
        "import pandas as pd",
        "pd.DataFrame(samples)",
    ]
    return [
        md(
            "## Evaluate on a Slice",
            "",
            "Run the recipe's `answer_question` over a small slice of the hand-curated eval set. "
            "Full RAGAS metrics are exercised in `recipes/09-evaluation-and-production/ragas-triad-eval.ipynb`; "
            "here we just print a quick spot-check table so you can eyeball whether the technique is on track.",
        ),
        code(*body),
    ]


def section_discussion(r: Recipe) -> list[dict]:
    if "discussion" in r.skip_sections or not r.discussion_md:
        return []
    return [md("## Closing Thoughts", "", *r.discussion_md)]


SECTIONS = [
    section_theory,
    section_diagram,
    section_references,
    section_setup,
    section_build_steps,
    section_inspection,
    section_run,
    section_comparison,
    section_tuning,
    section_evaluate,
    section_discussion,
]


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def build_cells(r: Recipe) -> list[dict]:
    cells: list[dict] = []
    for section in SECTIONS:
        cells.extend(section(r))
    return cells


def write_notebook(r: Recipe) -> Path:
    cells = build_cells(r)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "rag_cookbook": {
                "recipe_title": r.title,
                "category": r.category,
                "needs_multimodal": r.needs_multimodal,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / r.path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return out


__all__ = ["build_cells", "write_notebook", "md", "code", "SECTIONS"]
