"""Authoring infrastructure for cookbook recipes.

The published artifacts are the `.ipynb` files under `recipes/`. This package
exists to author them at scale and uniformly: one rich `Recipe` dataclass per
notebook, a section-driven renderer, and a validator that fails loudly when a
recipe drops below the depth bar.

Layout:
    schema.py     — Reference, CodeStep, Recipe dataclasses
    render.py     — Cell helpers + section renderers + write_notebook()
    validator.py  — Depth checks (≥5 references, ≥5 cells, etc.)
    standard.py   — Shared section content (setup, evaluate, references-block)

Recipes themselves live in `scripts/recipes/0X_<category>.py`, each exposing a
top-level `RECIPES: list[Recipe]`.
"""
from .schema import CodeStep, Recipe, Reference
from .render import write_notebook
from .validator import validate

__all__ = ["CodeStep", "Recipe", "Reference", "write_notebook", "validate"]
