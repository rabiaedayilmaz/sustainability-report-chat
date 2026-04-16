"""Application settings loaded from environment variables (.env)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed configuration for the RAG pipeline.

    All values can be overridden via environment variables or a ``.env`` file
    sitting next to the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----- Qdrant -----
    qdrant_url: str = Field(..., description="Qdrant Cloud REST endpoint.")
    qdrant_api_key: str = Field(..., description="Qdrant Cloud API key.")
    qdrant_cluster: str = Field(
        "rag-projects",
        description="Logical cluster label (recorded in payload, not the collection name).",
    )
    qdrant_collection: str = Field(
        "nttdata_sustainability",
        description="Vector collection name; one collection per ingest run.",
    )

    # ----- Embedding -----
    embedding_model: str = Field("intfloat/multilingual-e5-large")
    embedding_device: Literal["cpu", "cuda", "mps"] = Field("cpu")
    embedding_batch_size: int = Field(32, ge=1)

    # ----- Chunking (character-based; ~4 chars per English token) -----
    chunk_size: int = Field(800, ge=64, description="Max characters per chunk.")
    chunk_overlap: int = Field(120, ge=0, description="Chars of overlap between adjacent chunks.")

    # ----- PDF / OCR (PaddleOCR backend) -----
    enable_ocr: bool = Field(True)
    ocr_language: str = Field(
        "en",
        description="PaddleOCR language code: 'en', 'ch', 'tr', 'german', 'french', 'japan', 'korean', ...",
    )
    ocr_dpi: int = Field(200, ge=72, description="DPI used when rendering PDF pages to images for OCR.")
    ocr_use_angle_cls: bool = Field(
        True,
        description="Detect & rotate upside-down / sideways text before recognition.",
    )
    min_chars_for_ocr_skip: int = Field(
        50,
        description="If a page yields fewer chars than this, fall back to OCR.",
    )

    # ----- LLM (Ollama) -----
    ollama_base_url: str = Field("http://localhost:11434")
    ollama_model: str = Field("qwen3:8b")
    ollama_temperature: float = Field(0.2, ge=0.0, le=2.0)
    ollama_num_ctx: int = Field(8192, ge=512)
    ollama_timeout: float = Field(120.0, gt=0)

    # ----- Retrieval -----
    retrieval_top_k: int = Field(6, ge=1)
    retrieval_score_threshold: float | None = Field(
        None,
        description="Drop hits below this cosine similarity (None = keep all).",
    )

    # ----- Paths -----
    data_dir: Path = Field(Path("data"), description="Root directory containing data/<year>/*.pdf.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()  # type: ignore[call-arg]
