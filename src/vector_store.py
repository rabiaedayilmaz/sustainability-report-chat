"""Qdrant vector store wrapper used by the RAG pipeline."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .chunker import Chunk
from .config import Settings, get_settings
from .utils.log import logger


@dataclass(frozen=True)
class SearchHit:
    """A single retrieval result, denormalised for callers."""

    score: float
    text: str
    source: str
    year: str
    page_num: int
    chunk_id: str
    payload: dict


class QdrantVectorStore:
    """Thin wrapper around qdrant-client with a single fixed collection.

    Cosine distance is used with normalised vectors (matches our embedder).
    Year and source are indexed so that callers can pre-filter cheaply.
    """

    def __init__(self, embedding_dim: int, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dim = int(embedding_dim)
        self.collection = self.settings.qdrant_collection
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=60,
        )

    # ------------------------------------------------------------- collection
    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection (and supporting indexes) if needed.

        Raises ``ValueError`` when the existing collection's vector dimension
        doesn't match the embedder. This is the most common silent corruption
        mode after switching embedding models — better to fail loud than to
        upsert vectors of the wrong size and get garbage retrievals.
        """
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            logger.info("Recreating collection '%s'", self.collection)
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            self._assert_dim_matches()
            return

        logger.info("Creating collection '%s' (dim=%d)", self.collection, self.dim)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
        )
        for field, schema in (
            ("year", qm.PayloadSchemaType.KEYWORD),
            ("source", qm.PayloadSchemaType.KEYWORD),
        ):
            try:
                self.client.create_payload_index(
                    self.collection, field_name=field, field_schema=schema
                )
            except Exception as exc:  # index may already exist
                logger.debug("create_payload_index(%s) skipped: %s", field, exc)

    def _assert_dim_matches(self) -> None:
        """Raise if the on-disk collection's vector size differs from this embedder."""
        try:
            info = self.client.get_collection(self.collection)
            actual = int(info.config.params.vectors.size)
        except Exception as exc:
            logger.warning("Could not introspect collection '%s' for dim check: %s", self.collection, exc)
            return
        if actual != self.dim:
            raise ValueError(
                f"Vector dimension mismatch on collection '{self.collection}': "
                f"on-disk={actual}, embedder={self.dim}. "
                f"This usually means the embedding model changed. "
                f"Re-ingest with --recreate."
            )

    # ----------------------------------------------------------------- writes
    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        batch_size: int = 128,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch"
            )
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vecs = vectors[start : start + batch_size]
            points = [
                qm.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                    vector=v.tolist(),
                    payload=c.to_payload(),
                )
                for c, v in zip(batch_chunks, batch_vecs)
            ]
            self.client.upsert(collection_name=self.collection, points=points)

    # ----------------------------------------------------------------- search
    def search(
        self,
        vector: np.ndarray,
        top_k: int = 6,
        year: str | None = None,
        source: str | None = None,
        score_threshold: float | None = None,
    ) -> List[SearchHit]:
        # Fail-soft: an unindexed collection should return [], not raise — callers
        # already handle empty hits gracefully (RAGAnswer fallback).
        if not self.collection_exists():
            logger.warning(
                "Search against missing collection '%s' — run scripts/index_pdfs.py first.",
                self.collection,
            )
            return []

        must: List[qm.FieldCondition] = []
        if year:
            must.append(qm.FieldCondition(key="year", match=qm.MatchValue(value=year)))
        if source:
            must.append(qm.FieldCondition(key="source", match=qm.MatchValue(value=source)))
        flt = qm.Filter(must=must) if must else None

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector.tolist(),
                limit=top_k,
                query_filter=flt,
                score_threshold=score_threshold,
                with_payload=True,
            )
            results = response.points
        except Exception as exc:
            logger.error("Qdrant query_points failed on '%s': %s", self.collection, exc)
            return []
        hits: List[SearchHit] = []
        for r in results:
            payload = r.payload or {}
            hits.append(
                SearchHit(
                    score=float(r.score),
                    text=str(payload.get("text", "")),
                    source=str(payload.get("source", "")),
                    year=str(payload.get("year", "")),
                    page_num=int(payload.get("page_num", -1)),
                    chunk_id=str(payload.get("chunk_id", r.id)),
                    payload=payload,
                )
            )
        return hits

    # ---------------------------------------------------------------- inspect
    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def collection_exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection))
