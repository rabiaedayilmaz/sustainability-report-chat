"""Prometheus instrumentation for the RAG pipeline.

We expose two layers of metrics:
  1. **HTTP layer** — registered on the FastAPI app via
     :class:`prometheus_fastapi_instrumentator.Instrumentator`. Gives you
     ``http_requests_total`` and ``http_request_duration_seconds`` for free.
  2. **RAG layer** — defined here, observed by the pipeline. Tracks retrieval,
     LLM, embedding latency and a counter of /ask outcomes.

Use the :class:`Timer` context manager for low-overhead histogram observation:

    from src.utils.metrics import Timer, RETRIEVAL_LATENCY
    with Timer(RETRIEVAL_LATENCY):
        hits = store.search(...)
"""
from __future__ import annotations

import time
from contextlib import AbstractContextManager
from typing import Any, Optional

from prometheus_client import Counter, Gauge, Histogram

# Latency buckets sized for our actual operations:
#   * retrieval: ~30-300 ms (Qdrant Cloud roundtrip)
#   * llm:       ~5-90 s (CPU Ollama)
#   * embed:     ~5-50 ms (single query) / longer for batches
_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
    1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
)

ASK_TOTAL = Counter(
    "rag_ask_total",
    "Total /ask requests, broken down by outcome.",
    labelnames=("status",),  # success | no_hits | error
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_seconds",
    "Wall-clock time to embed a query and pull top-k from Qdrant.",
    buckets=_LATENCY_BUCKETS,
)

LLM_LATENCY = Histogram(
    "rag_llm_seconds",
    "Wall-clock time for the LLM generation step (Ollama /api/generate).",
    buckets=_LATENCY_BUCKETS,
)

EMBED_LATENCY = Histogram(
    "rag_embed_seconds",
    "Wall-clock time per embedding call.",
    labelnames=("kind",),  # query | passage
    buckets=_LATENCY_BUCKETS,
)

HITS_PER_QUERY = Histogram(
    "rag_hits_returned",
    "Number of chunks returned per /ask call (after filters / threshold).",
    buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50),
)

CHUNKS_INDEXED = Counter(
    "rag_chunks_indexed_total",
    "Lifetime number of chunks upserted into Qdrant.",
)

INDEX_INFO = Gauge(
    "rag_index_info",
    "Static labels describing the current index — always 1.",
    labelnames=("embedding_model", "embedding_dim", "schema_version", "fingerprint"),
)


class Timer(AbstractContextManager):
    """Cheap context manager that records elapsed seconds into ``histogram``.

    Pass ``labels=...`` for histograms that have label names.
    """

    __slots__ = ("histogram", "labels", "_start")

    def __init__(self, histogram: Histogram, labels: Optional[dict[str, str]] = None):
        self.histogram = histogram
        self.labels = labels
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        elapsed = time.perf_counter() - self._start
        target = self.histogram.labels(**self.labels) if self.labels else self.histogram
        target.observe(elapsed)


def set_index_info(
    embedding_model: str,
    embedding_dim: int,
    schema_version: int,
    fingerprint: str,
) -> None:
    """Stamp the current index identity onto the ``rag_index_info`` gauge."""
    INDEX_INFO.labels(
        embedding_model=embedding_model,
        embedding_dim=str(embedding_dim),
        schema_version=str(schema_version),
        fingerprint=fingerprint,
    ).set(1)
