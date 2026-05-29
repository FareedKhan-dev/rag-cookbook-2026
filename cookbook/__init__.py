"""The RAG Cookbook 2026 — installable utility package.

Every recipe in `recipes/` imports from this package. The goal is to keep
notebooks readable: the boring plumbing (provider switching, corpus loading,
vector-store wrappers, tracing, evaluation) lives here, and each notebook
spends its cells on the technique that gives it its name.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rag-cookbook")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = [
    "__version__",
]
