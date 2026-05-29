"""Print a per-notebook quality report for the cookbook.

Reports: cell count, markdown wordcount, code wordcount, executed-yes/no,
file size. Run with: `python scripts/recipe_quality_report.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

ROOT = Path(__file__).resolve().parent.parent
RECIPES = ROOT / "recipes"


def analyse(path: Path) -> dict:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    md = [c for c in cells if c["cell_type"] == "markdown"]
    code = [c for c in cells if c["cell_type"] == "code"]
    md_words = 0
    for c in md:
        src = c["source"]
        if isinstance(src, list):
            src = "".join(src)
        md_words += len(src.split())
    executed = sum(1 for c in code if c.get("outputs"))
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "cells": len(cells),
        "md": len(md),
        "code": len(code),
        "md_words": md_words,
        "executed_cells": executed,
        "kb": path.stat().st_size // 1024,
    }


def main() -> int:
    rows = [analyse(p) for p in sorted(RECIPES.rglob("*.ipynb"))]
    print(f"{'recipe':75s} {'cells':>5s} {'md':>4s} {'code':>5s} {'words':>6s} {'exec':>5s} {'KB':>4s}")
    print("-" * 110)
    for r in rows:
        print(
            f"{r['path']:75s} {r['cells']:>5d} {r['md']:>4d} {r['code']:>5d} "
            f"{r['md_words']:>6d} {r['executed_cells']:>5d} {r['kb']:>4d}"
        )
    print()
    print(f"Total: {len(rows)} notebooks")
    total_md = sum(r["md_words"] for r in rows)
    total_cells = sum(r["cells"] for r in rows)
    total_executed = sum(r["executed_cells"] for r in rows)
    print(f"  cells:      {total_cells:,}")
    print(f"  md words:   {total_md:,}")
    print(f"  exec cells: {total_executed:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
