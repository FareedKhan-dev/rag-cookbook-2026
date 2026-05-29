"""Thin wrappers over Qdrant, LanceDB, and Chroma with one identical API.

The point: every recipe uses the same `.add()` / `.search()` / `.hybrid_search()`
signatures, so swapping vector stores is a one-line change. The wrappers are
intentionally minimal — they expose only what the recipes need, no plumbing
for sharding, replication, or schema migrations.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Hit:
    """One retrieval result."""

    doc_id: str
    text: str
    score: float
    metadata: dict


class VectorBackend(ABC):
    """Shared API across Qdrant, LanceDB, and Chroma backends."""

    @abstractmethod
    def add(
        self,
        texts: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadatas: Sequence[dict] | None = None,
        ids: Sequence[str] | None = None,
    ) -> None: ...

    @abstractmethod
    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[Hit]: ...

    def hybrid_search(
        self,
        query_vector: Sequence[float],
        query_text: str,
        top_k: int = 5,
    ) -> list[Hit]:
        """Default fallback: dense only. Backends that support sparse override."""
        return self.search(query_vector, top_k=top_k)


# ---------------------------------------------------------------------------
# Qdrant (default — in-memory unless QDRANT_URL is set)
# ---------------------------------------------------------------------------
class QdrantBackend(VectorBackend):
    def __init__(
        self,
        collection: str,
        dim: int,
        *,
        location: str | None = None,
        enable_hybrid: bool = True,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(location=location or ":memory:")
        self.collection = collection
        self.enable_hybrid = enable_hybrid

        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, texts, vectors, metadatas=None, ids=None):  # type: ignore[override]
        from qdrant_client.models import PointStruct

        metadatas = metadatas or [{} for _ in texts]
        # Qdrant only accepts UUID strings or unsigned ints as point IDs.
        # We derive a stable UUIDv5 from the user-supplied id and keep the
        # original in the payload under `external_id` so retrieval is
        # round-trippable.
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, str(i))) for i in ids]
        points = [
            PointStruct(
                id=pid,
                vector=list(v),
                payload={"text": t, "external_id": str(orig_id), **m},
            )
            for pid, orig_id, t, v, m in zip(point_ids, ids, texts, vectors, metadatas)
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k=5):  # type: ignore[override]
        result = self.client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            limit=top_k,
            with_payload=True,
        ).points
        return [
            Hit(
                doc_id=str(p.payload.get("external_id", p.id)) if p.payload else str(p.id),
                text=str(p.payload.get("text", "")) if p.payload else "",
                score=float(p.score),
                metadata={
                    k: v
                    for k, v in (p.payload or {}).items()
                    if k not in {"text", "external_id"}
                },
            )
            for p in result
        ]


# ---------------------------------------------------------------------------
# LanceDB (embedded — good for laptop / notebook demos)
# ---------------------------------------------------------------------------
class LanceBackend(VectorBackend):
    def __init__(self, table_name: str, dim: int, *, uri: str = "./lance_data") -> None:
        import lancedb
        import pyarrow as pa

        self.db = lancedb.connect(uri)
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
            ]
        )
        if table_name in self.db.table_names():
            self.table = self.db.open_table(table_name)
        else:
            self.table = self.db.create_table(table_name, schema=schema)

    def add(self, texts, vectors, metadatas=None, ids=None):  # type: ignore[override]
        import json as _json

        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        rows = [
            {
                "id": i,
                "text": t,
                "metadata": _json.dumps(m),
                "vector": list(v),
            }
            for i, t, v, m in zip(ids, texts, vectors, metadatas)
        ]
        self.table.add(rows)

    def search(self, query_vector, top_k=5):  # type: ignore[override]
        import json as _json

        rows = self.table.search(list(query_vector)).limit(top_k).to_list()
        hits: list[Hit] = []
        for r in rows:
            hits.append(
                Hit(
                    doc_id=r["id"],
                    text=r["text"],
                    score=float(r.get("_distance", 0.0)),
                    metadata=_json.loads(r.get("metadata", "{}")),
                )
            )
        return hits


# ---------------------------------------------------------------------------
# Chroma (simplest — used in the cookbook-tour recipe)
# ---------------------------------------------------------------------------
class ChromaBackend(VectorBackend):
    def __init__(self, collection: str, *, persist: str | None = None) -> None:
        import chromadb

        self.client = (
            chromadb.PersistentClient(path=persist) if persist else chromadb.Client()
        )
        self.col = self.client.get_or_create_collection(collection)

    def add(self, texts, vectors, metadatas=None, ids=None):  # type: ignore[override]
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        self.col.add(
            ids=list(ids),
            embeddings=[list(v) for v in vectors],
            documents=list(texts),
            metadatas=list(metadatas),
        )

    def search(self, query_vector, top_k=5):  # type: ignore[override]
        result = self.col.query(
            query_embeddings=[list(query_vector)],
            n_results=top_k,
        )
        hits: list[Hit] = []
        for i in range(len(result["ids"][0])):
            hits.append(
                Hit(
                    doc_id=result["ids"][0][i],
                    text=result["documents"][0][i],
                    score=float(result["distances"][0][i]),
                    metadata=result["metadatas"][0][i] or {},
                )
            )
        return hits


__all__ = [
    "Hit",
    "VectorBackend",
    "QdrantBackend",
    "LanceBackend",
    "ChromaBackend",
]
