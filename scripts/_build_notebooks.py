"""Build cookbook notebooks from category recipe specs.

This is the v2 builder: thin loader. Per-category recipes live in
`scripts/recipes/0X_<category>.py`, each exposing `RECIPES: list[Recipe]`.
The renderer and validator live in `scripts/authoring/`.

Usage:
    py scripts/_build_notebooks.py            # build all categories
    py scripts/_build_notebooks.py 01 02      # build only listed categories
    py scripts/_build_notebooks.py --check    # validate without writing
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = ROOT / "scripts" / "recipes"
sys.path.insert(0, str(ROOT / "scripts"))

from authoring import Recipe, validate, write_notebook  # noqa: E402
from authoring.validator import RecipeValidationError  # noqa: E402


def discover_recipe_modules(filters: list[str] | None) -> list[Path]:
    """Return per-category recipe spec files, optionally filtered."""
    if not RECIPES_DIR.exists():
        return []
    files = sorted(RECIPES_DIR.glob("[0-9]*.py"))
    if filters:
        return [f for f in files if any(f.name.startswith(p) for p in filters)]
    return files


def load_recipes(module_path: Path) -> list[Recipe]:
    spec = importlib.util.spec_from_file_location(
        f"recipes.{module_path.stem}", module_path
    )
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "RECIPES", [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "filters",
        nargs="*",
        help="Category prefixes to build (e.g. '01', '06'). Empty = all.",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Run validators only; don't write notebooks.",
    )
    args = ap.parse_args()

    modules = discover_recipe_modules(args.filters)
    if not modules:
        print(f"No recipe modules found under {RECIPES_DIR}.")
        print("Add files like scripts/recipes/01_foundations.py with `RECIPES = [...]`.")
        return 1

    all_recipes: list[Recipe] = []
    for m in modules:
        rs = load_recipes(m)
        print(f"[{m.name}] loaded {len(rs)} recipes")
        all_recipes.extend(rs)

    problems: list[str] = []
    for r in all_recipes:
        problems.extend(validate(r))

    if problems:
        print("\nVALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        if args.check:
            return 2
        raise RecipeValidationError(f"{len(problems)} problem(s); see above")

    if args.check:
        print(f"\n{len(all_recipes)} recipes validated cleanly.")
        return 0

    for r in all_recipes:
        out = write_notebook(r)
        rel = out.relative_to(ROOT)
        print(f"wrote {rel}")

    print(f"\nGenerated {len(all_recipes)} notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
