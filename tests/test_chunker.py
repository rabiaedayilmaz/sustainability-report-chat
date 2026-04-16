"""Pure-Python tests for TextChunker — no Qdrant / no embedder needed."""
from __future__ import annotations

from src.chunker import Chunk, TextChunker
from src.pdf_processor import Page


def _make_chunker(size: int = 200, overlap: int = 30) -> TextChunker:
    """Build a chunker without going through pydantic Settings / .env."""
    c = TextChunker.__new__(TextChunker)
    c.chunk_size = size
    c.chunk_overlap = overlap
    c.separators = TextChunker.DEFAULT_SEPARATORS
    return c


def test_split_respects_chunk_size() -> None:
    # Note: the merge phase keeps chunks <= chunk_size, but the overlap phase
    # prepends the tail of the previous chunk. So the documented bound is
    # chunk_size + chunk_overlap (matches LangChain's RecursiveCharacterTextSplitter).
    c = _make_chunker(size=120, overlap=20)
    text = ("Sentence one. " * 30).strip()
    out = c.split_text(text)
    assert all(len(s) <= c.chunk_size + c.chunk_overlap for s in out), [len(s) for s in out]
    assert len(out) > 1, "expected the text to fragment"


def test_split_returns_empty_for_empty_input() -> None:
    assert _make_chunker().split_text("") == []


def test_overlap_visible_between_chunks() -> None:
    c = _make_chunker(size=100, overlap=20)
    text = "A" * 200 + "\n\n" + "B" * 200
    out = c.split_text(text)
    assert len(out) >= 2
    # The second chunk should start with the overlap tail of the first.
    tail = out[0][-20:]
    assert out[1].startswith(tail) or tail in out[1][:40]


def test_chunk_page_propagates_metadata() -> None:
    c = _make_chunker(size=200, overlap=20)
    page = Page(source="sr2024.pdf", year="2024", page_num=7,
                text="Some content. " * 40, extracted_via="text")
    chunks = c.chunk_page(page)
    assert chunks, "expected at least one chunk"
    for ch in chunks:
        assert isinstance(ch, Chunk)
        assert ch.source == "sr2024.pdf"
        assert ch.year == "2024"
        assert ch.page_num == 7
        assert ch.extracted_via == "text"
        assert ch.chunk_id.startswith("sr2024.pdf::p7::c")
        # Payload must include all keys the vector store relies on.
        keys = set(ch.to_payload())
        assert {"chunk_id", "text", "source", "year", "page_num"}.issubset(keys)
