"""End-to-end RAG pipeline: ingest, retrieve, generate."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Sequence

import httpx

from .chunker import Chunk, TextChunker
from .config import Settings, get_settings
from .embedder import E5Embedder
from .manifest import IngestManifest
from .pdf_processor import PDFProcessor
from .query_parser import YearExtractor
from .utils.log import logger
from .vector_store import QdrantVectorStore, SearchHit
from .version import CODE_VERSION, SCHEMA_VERSION, IndexVersion


SYSTEM_PROMPT = (
    "You are a precise sustainability analyst answering questions about NTT DATA "
    "Business Solutions sustainability reports.\n\n"
    "Rules:\n"
    "- Use ONLY the provided context to answer.\n"
    "- If the answer is not in the context, reply exactly: "
    '"I cannot find this in the provided reports."\n'
    "- Cite sources inline using the bracketed reference numbers from the context, "
    "e.g. [1] or [2, 3]. EVERY factual claim must carry at least one citation.\n"
    "- Prefer numbers, units, and years from the reports over generic statements.\n"
    "- Be concise; do not invent figures, targets, or commitments."
)

USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Context (top-{k} retrieved chunks, ordered by similarity). "
    "Cite by the [N] number shown in each block:\n"
    "{context}\n\n"
    "Answer the question following all rules. Use [N] inline citations."
)

# Snippet returned in the API/CLI sources list. Long enough that a reader can
# verify a citation without re-fetching the chunk, short enough to keep
# responses lightweight.
SOURCE_SNIPPET_LEN = 500


@dataclass
class RAGAnswer:
    """Structured response returned by :meth:`RAGPipeline.ask`."""

    answer: str
    sources: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class RAGPipeline:
    """Composition root that wires processor -> chunker -> embedder -> store -> LLM."""

    INGEST_BATCH = 256
    _NO_HITS_ANSWER = "I could not retrieve any relevant context from the reports."

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.processor = PDFProcessor(self.settings)
        self.chunker = TextChunker(self.settings)
        self.embedder = E5Embedder(self.settings)
        self.store = QdrantVectorStore(self.embedder.dim, self.settings)
        self.year_extractor = YearExtractor.from_data_dir(self.settings.data_dir)
        self.version = IndexVersion(
            code_version=CODE_VERSION,
            schema_version=SCHEMA_VERSION,
            embedding_model=self.settings.embedding_model,
            embedding_dim=self.embedder.dim,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    # ================================================================ ingest
    def ingest(self, data_dir: Path | str | None = None, recreate: bool = False) -> dict:
        """Process every PDF under ``data_dir`` and upsert into Qdrant."""
        data_dir = Path(data_dir or self.settings.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

        self.store.ensure_collection(recreate=recreate)

        n_pages = 0
        n_chunks = 0
        n_failed_batches = 0
        buffer: List[Chunk] = []

        def _try_flush() -> None:
            nonlocal n_chunks, n_failed_batches
            try:
                self._flush(buffer)
                n_chunks += len(buffer)
            except Exception as exc:  # one bad batch must not kill the whole ingest
                n_failed_batches += 1
                logger.exception(
                    "Flush failed for %d chunks (batch %d): %s — continuing.",
                    len(buffer),
                    n_failed_batches,
                    exc,
                )
            finally:
                buffer.clear()

        for page in self.processor.process_directory(data_dir):
            n_pages += 1
            buffer.extend(self.chunker.chunk_page(page))
            if len(buffer) >= self.INGEST_BATCH:
                _try_flush()
        if buffer:
            _try_flush()

        total = self.store.count()
        logger.info(
            "Ingest complete. pages=%d chunks=%d failed_batches=%d collection_total=%d",
            n_pages,
            n_chunks,
            n_failed_batches,
            total,
        )

        # --- versioning: persist a manifest of what was just indexed -----------
        pdf_count = sum(1 for _ in Path(data_dir).glob("**/*.pdf"))
        manifest = IngestManifest.build(
            collection=self.store.collection,
            version=self.version,
            pdf_count=pdf_count,
            page_count=n_pages,
            chunk_count=n_chunks,
            failed_batches=n_failed_batches,
        )
        manifest.write(data_dir)

        return {
            "pages": n_pages,
            "chunks": n_chunks,
            "failed_batches": n_failed_batches,
            "collection_total": total,
            "fingerprint": self.version.fingerprint(),
        }

    def _flush(self, chunks: List[Chunk]) -> None:
        vectors = self.embedder.embed_passages([c.text for c in chunks])
        self.store.upsert(chunks, vectors)

    # ============================================================== retrieve
    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        year: str | None = None,
        source: str | None = None,
    ) -> List[SearchHit]:
        qv = self.embedder.embed_query(question)
        return self.store.search(
            vector=qv,
            top_k=top_k or self.settings.retrieval_top_k,
            year=year,
            source=source,
            score_threshold=self.settings.retrieval_score_threshold,
        )

    # ============================================================== generate
    @staticmethod
    def _format_context(hits: Sequence[SearchHit]) -> str:
        blocks: List[str] = []
        for i, h in enumerate(hits, start=1):
            header = (
                f"[{i}] source={h.source} | year={h.year} | p.{h.page_num} "
                f"| score={h.score:.3f}"
            )
            blocks.append(f"{header}\n{h.text}")
        return "\n\n".join(blocks)

    def _build_payload(self, question: str, hits: Sequence[SearchHit]) -> dict:
        context = self._format_context(hits)
        prompt = USER_TEMPLATE.format(question=question, k=len(hits), context=context)
        return {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": self.settings.ollama_temperature,
                "num_ctx": self.settings.ollama_num_ctx,
            },
        }

    def _ollama_url(self) -> str:
        return f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"

    def _generate(self, question: str, hits: Sequence[SearchHit]) -> str:
        try:
            r = httpx.post(
                self._ollama_url(),
                json=self._build_payload(question, hits),
                timeout=self.settings.ollama_timeout,
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama call failed ({exc}). Is the daemon running at "
                f"{self.settings.ollama_base_url} and is model "
                f"'{self.settings.ollama_model}' pulled?"
            ) from exc
        return (r.json().get("response") or "").strip()

    async def _agenerate(self, question: str, hits: Sequence[SearchHit]) -> str:
        try:
            async with httpx.AsyncClient(timeout=self.settings.ollama_timeout) as client:
                r = await client.post(self._ollama_url(), json=self._build_payload(question, hits))
                r.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama call failed ({exc}). Is the daemon running at "
                f"{self.settings.ollama_base_url} and is model "
                f"'{self.settings.ollama_model}' pulled?"
            ) from exc
        return (r.json().get("response") or "").strip()

    # ================================================================== ask
    @staticmethod
    def _hits_to_sources(hits: Sequence[SearchHit]) -> List[dict]:
        out: List[dict] = []
        for i, h in enumerate(hits, start=1):
            snippet = h.text[:SOURCE_SNIPPET_LEN]
            if len(h.text) > SOURCE_SNIPPET_LEN:
                snippet += "..."
            out.append(
                {
                    "ref": i,  # matches the [N] used in the LLM answer
                    "source": h.source,
                    "year": h.year,
                    "page_num": h.page_num,
                    "chunk_id": h.chunk_id,
                    "score": round(h.score, 4),
                    "snippet": snippet,
                }
            )
        return out

    @staticmethod
    def _validate_question(question: str) -> str:
        if not isinstance(question, str):
            raise ValueError("question must be a string")
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("question must not be empty or whitespace")
        if len(cleaned) > 4000:
            raise ValueError("question is too long (>4000 chars)")
        return cleaned

    def _resolve_year(self, question: str, explicit: str | None) -> str | None:
        """Honour an explicit ``year`` argument; otherwise auto-extract one."""
        if explicit is not None:
            return explicit
        inferred = self.year_extractor.extract(question)
        if inferred:
            logger.info("Auto-detected year filter: %s (from question)", inferred)
        return inferred

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        year: str | None = None,
        source: str | None = None,
    ) -> RAGAnswer:
        question = self._validate_question(question)
        year = self._resolve_year(question, year)
        hits = self.retrieve(question, top_k=top_k, year=year, source=source)
        if not hits:
            return RAGAnswer(answer=self._NO_HITS_ANSWER, sources=[])
        answer = self._generate(question, hits)
        return RAGAnswer(answer=answer, sources=self._hits_to_sources(hits))

    async def aask(
        self,
        question: str,
        top_k: int | None = None,
        year: str | None = None,
        source: str | None = None,
    ) -> RAGAnswer:
        question = self._validate_question(question)
        year = self._resolve_year(question, year)
        hits = self.retrieve(question, top_k=top_k, year=year, source=source)
        if not hits:
            return RAGAnswer(answer=self._NO_HITS_ANSWER, sources=[])
        answer = await self._agenerate(question, hits)
        return RAGAnswer(answer=answer, sources=self._hits_to_sources(hits))

    # =============================================================== health
    HEALTH_TIMEOUT = 5.0  # seconds; bound any single dependency probe

    def health(self) -> dict:
        """Per-component health checks with latency measurements.

        Returns a dict shaped like ``{"<component>": {"status": "up"|"down", ...}}``
        suitable for direct serialisation by the API layer.
        """
        return {
            "qdrant": self._check_qdrant(),
            "ollama": self._check_ollama(),
            "embedder": self._check_embedder(),
        }

    @staticmethod
    def _ms(start: float) -> float:
        import time as _t
        return round((_t.perf_counter() - start) * 1000.0, 2)

    def _check_qdrant(self) -> dict:
        import time as _t
        start = _t.perf_counter()
        try:
            exists = self.store.collection_exists()
            count = self.store.count() if exists else 0
            return {
                "status": "up",
                "latency_ms": self._ms(start),
                "collection": self.store.collection,
                "collection_exists": exists,
                "indexed_chunks": count,
            }
        except Exception as exc:
            return {
                "status": "down",
                "latency_ms": self._ms(start),
                "collection": self.store.collection,
                "error": str(exc),
            }

    def _check_ollama(self) -> dict:
        import time as _t
        start = _t.perf_counter()
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/tags"
        wanted = self.settings.ollama_model
        try:
            r = httpx.get(url, timeout=self.HEALTH_TIMEOUT)
            r.raise_for_status()
            return {
                "status": "up",
                "latency_ms": self._ms(start),
                "model": wanted,
                "model_available": wanted,
            }
        except Exception as exc:
            return {
                "status": "down",
                "latency_ms": self._ms(start),
                "model": wanted,
                "error": str(exc),
            }

    def _check_embedder(self) -> dict:
        """The embedder is loaded in-process — if pipeline init succeeded it's up."""
        return {
            "status": "up",
            "model": self.settings.embedding_model,
            "device": self.settings.embedding_device,
            "dim": self.embedder.dim,
        }
