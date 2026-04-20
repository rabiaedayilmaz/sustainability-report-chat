"""E5-family embeddings via sentence-transformers."""
from __future__ import annotations

from typing import List, Protocol, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings, get_settings
from .utils.log import logger

E5_DEFAULT_MODEL = "intfloat/multilingual-e5-large"
HARRIER_DEFAULT_MODEL = "harrier-oss-v1-270m"


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...


class _BaseSTEmbedder:
    """Common sentence-transformers wrapper with optional query/passage prefixes."""

    QUERY_PREFIX = ""
    PASSAGE_PREFIX = ""

    def __init__(self, settings: Settings | None = None, model_name: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.embedding_model
        logger.info(
            "Loading embedding model %s on %s (backend=%s)",
            self.model_name,
            self.settings.embedding_device,
            self.settings.embedding_backend,
        )
        self.model = SentenceTransformer(
            self.model_name,
            device=self.settings.embedding_device,
        )
        dim = self.model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(
                f"Could not determine embedding dimension for {self.model_name}"
            )
        self._dim: int = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    # ---------------------------------------------------------------- helpers
    def _prefixed(self, texts: Sequence[str], is_query: bool) -> List[str]:
        prefix = self.QUERY_PREFIX if is_query else self.PASSAGE_PREFIX
        return [prefix + (t or "") for t in texts]

    # ----------------------------------------------------------------- public
    def embed(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        """Encode a batch of texts, normalised for cosine similarity."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vectors = self.model.encode(
            self._prefixed(texts, is_query),
            batch_size=self.settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """One-shot helper that returns a single 1-D vector."""
        return self.embed([text], is_query=True)[0]

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed(texts, is_query=False)


class E5Embedder(_BaseSTEmbedder):
    """Wraps a sentence-transformers model with the required E5 prefixes.

    The E5 family expects ``"query: ..."`` for search-time inputs and
    ``"passage: ..."`` for documents being indexed. Mixing them up silently
    degrades retrieval quality, so this class enforces the distinction.
    """

    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

class HarrierEmbedder(_BaseSTEmbedder):
    """Harrier embeddings (no E5 query/passage prefixes)."""


def resolve_embedding_model(settings: Settings) -> str:
    """Return the effective model for the selected backend.

    This keeps common defaults user-friendly: switching backend to Harrier while
    leaving the old E5 default model now auto-picks Harrier's default model.
    """
    if settings.embedding_backend == "harrier" and settings.embedding_model == E5_DEFAULT_MODEL:
        logger.info(
            "embedding_backend=harrier but model is E5 default (%s); switching to %s",
            E5_DEFAULT_MODEL,
            HARRIER_DEFAULT_MODEL,
        )
        return HARRIER_DEFAULT_MODEL
    if settings.embedding_backend == "e5" and settings.embedding_model == HARRIER_DEFAULT_MODEL:
        logger.info(
            "embedding_backend=e5 but model is Harrier default (%s); switching to %s",
            HARRIER_DEFAULT_MODEL,
            E5_DEFAULT_MODEL,
        )
        return E5_DEFAULT_MODEL
    return settings.embedding_model


def create_embedder(settings: Settings | None = None) -> Embedder:
    cfg = settings or get_settings()
    model_name = resolve_embedding_model(cfg)
    cfg.embedding_model = model_name
    if cfg.embedding_backend == "harrier":
        return HarrierEmbedder(cfg, model_name=model_name)
    return E5Embedder(cfg, model_name=model_name)
