"""SHA256-keyed disk cache for LLM and embedding calls.

Goal: every recipe in the cookbook should be cheap to *re-execute* after the
first pass. Authors and CI runs alike benefit. We don't try to be a real
cache — no eviction, no TTL, no versioning. Just hash the inputs, write the
output, read it back next time.

Enable / disable with the `COOKBOOK_CACHE` env var:
    COOKBOOK_CACHE=1      # default — on
    COOKBOOK_CACHE=0      # bypass; always hit the network

Cache files live under `<repo>/.cache/providers/<sha256>.json`. To wipe the
cache, delete the directory.

This module is intentionally dependency-free (`hashlib`, `json`, `pathlib`)
so it imports cleanly even on a half-installed environment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _REPO_ROOT / ".cache" / "providers"


def _enabled() -> bool:
    return os.getenv("COOKBOOK_CACHE", "1") != "0"


def _key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.json"


def get(payload: dict[str, Any]) -> Any | None:
    """Return the cached value for `payload`, or None on miss."""
    if not _enabled():
        return None
    p = _path(_key(payload))
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))["value"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def put(payload: dict[str, Any], value: Any) -> None:
    """Write `value` into the cache under the hash of `payload`."""
    if not _enabled():
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(_key(payload))
    try:
        p.write_text(
            json.dumps({"payload": payload, "value": value}, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        # Cache failure is never fatal — let the caller continue with the live value.
        return


def stats() -> dict[str, int]:
    """Tiny helper for the cookbook-tour recipe."""
    if not _CACHE_DIR.exists():
        return {"entries": 0}
    return {"entries": sum(1 for _ in _CACHE_DIR.glob("*.json"))}


__all__ = ["get", "put", "stats"]
