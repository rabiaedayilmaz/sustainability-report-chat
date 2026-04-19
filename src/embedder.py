"""E5-family embeddings via sentence-transformers."""
from __future__ import annotations

from typing import List, Protocol, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings, get_settings
from .utils.log import logger

E5_DEFAULT_MODEL = "intfloat/multilingual-e5-large"
HARRIER_DEFAULT_MODEL = "microsoft/harrier-oss-v1-0.6b"


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

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        logger.info(
            "Loading embedding model %s on %s (backend=%s)",
            self.settings.embedding_model,
            self.settings.embedding_device,
            self.settings.embedding_backend,
        )
        self.model = SentenceTransformer(
            self.settings.embedding_model,
            device=self.settings.embedding_device,
        )
        dim = self.model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(
                f"Could not determine embedding dimension for {self.settings.embedding_model}"
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


def create_embedder(settings: Settings | None = None) -> Embedder:
    cfg = settings or get_settings()
    if cfg.embedding_backend == "harrier":
        return HarrierEmbedder(cfg)
    return E5Embedder(cfg)
