"""Version metadata for the indexed corpus.

Bump :data:`SCHEMA_VERSION` when the chunk payload shape changes; that forces
the fingerprint (and thus the on-disk manifest) to differ from the previous
build, signalling that re-indexing is required.

The fingerprint deliberately encodes everything that, if changed, would make
existing vectors stale:
    * embedding model name
    * embedding dimension
    * payload schema version
    * chunk size + overlap

If you tweak only the LLM (Ollama model), the fingerprint does NOT change —
generation params do not invalidate the index.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

CODE_VERSION = "1.0.0"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexVersion:
    """Identity of an index build, suitable for fingerprinting + manifest."""

    code_version: str
    schema_version: int
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int

    def fingerprint(self) -> str:
        """Short stable hash uniquely identifying this index configuration."""
        material = "|".join(
            (
                self.embedding_model,
                str(self.embedding_dim),
                f"schema={self.schema_version}",
                f"chunk={self.chunk_size}",
                f"overlap={self.chunk_overlap}",
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "code_version": self.code_version,
            "schema_version": self.schema_version,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "fingerprint": self.fingerprint(),
        }
