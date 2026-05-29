"""Execute notebooks in-place against the configured provider.

For each notebook passed (or matched by `--batch`), runs every cell via
`nbclient` and writes the executed `.ipynb` back with real outputs visible.
This is how the published repo gets to look like every recipe has been run.

Usage:
    py scripts/execute_notebooks.py recipes/01-foundations/vanilla-pipeline.ipynb
    py scripts/execute_notebooks.py --batch 01-foundations
    py scripts/execute_notebooks.py --all                  # everything except needs-multimodal
    py scripts/execute_notebooks.py --all --include-multimodal

Caching is on by default (see `cookbook._cache`), so re-runs over the same
recipe are nearly free.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows consoles so progress symbols don't crash printing.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

ROOT = Path(__file__).resolve().parent.parent
RECIPES_ROOT = ROOT / "recipes"


def discover(batch: str | None, paths: list[str]) -> list[Path]:
    if paths:
        return [Path(p) if Path(p).is_absolute() else ROOT / p for p in paths]
    if batch:
        return sorted((RECIPES_ROOT / batch).glob("*.ipynb"))
    return sorted(RECIPES_ROOT.rglob("*.ipynb"))


def is_multimodal(path: Path) -> bool:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))["metadata"]
        return bool(meta.get("rag_cookbook", {}).get("needs_multimodal", False))
    except Exception:
        return False


def execute_one(path: Path, *, timeout_sec: int) -> tuple[bool, str]:
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError
    except ImportError as e:
        return False, f"missing dep: {e}"

    rel = path.relative_to(ROOT)
    print(f"  >> {rel}", flush=True)
    nb = nbformat.read(str(path), as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout_sec,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    t0 = time.time()
    try:
        client.execute()
    except CellExecutionError as e:
        return False, f"cell error: {e}"[:4000]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:4000]
    nbformat.write(nb, str(path))
    return True, f"ok ({time.time()-t0:.1f}s)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="Specific notebook paths.")
    ap.add_argument(
        "--batch", help="Category directory name, e.g. '01-foundations'."
    )
    ap.add_argument(
        "--all", action="store_true", help="Execute every recipe under recipes/."
    )
    ap.add_argument(
        "--include-multimodal",
        action="store_true",
        help="Include notebooks tagged as needing multimodal extras.",
    )
    ap.add_argument(
        "--timeout", type=int, default=900,
        help="Per-cell timeout in seconds (default 900).",
    )
    args = ap.parse_args()

    if not (args.paths or args.batch or args.all):
        ap.error("Specify paths, --batch, or --all.")

    notebooks = discover(args.batch, args.paths)
    if not args.include_multimodal:
        skipped_mm = [p for p in notebooks if is_multimodal(p)]
        notebooks = [p for p in notebooks if not is_multimodal(p)]
        for p in skipped_mm:
            print(f"  skip (multimodal) {p.relative_to(ROOT)}")

    if not notebooks:
        print("No notebooks to execute.")
        return 0

    print(f"Executing {len(notebooks)} notebook(s)...")
    ok, failed = 0, 0
    failures: list[tuple[Path, str]] = []
    for nb in notebooks:
        success, msg = execute_one(nb, timeout_sec=args.timeout)
        print(f"     {msg}")
        if success:
            ok += 1
        else:
            failed += 1
            failures.append((nb, msg))

    print(f"\nResult: {ok} ok, {failed} failed.")
    if failures:
        print("Failures:")
        for nb, msg in failures:
            print(f"  - {nb.relative_to(ROOT)}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
