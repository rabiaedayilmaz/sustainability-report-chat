"""Recursive character text splitter that prefers semantic boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .config import Settings, get_settings
from .pdf_processor import Page


@dataclass(frozen=True)
class Chunk:
    """A retrievable chunk of text plus the provenance needed for citation."""

    chunk_id: str
    text: str
    source: str
    year: str
    page_num: int
    extracted_via: str
    chunk_index: int

    def to_payload(self) -> dict:
        """Serialize as the Qdrant point payload (one row per chunk)."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "year": self.year,
            "page_num": self.page_num,
            "extracted_via": self.extracted_via,
            "chunk_index": self.chunk_index,
        }


class TextChunker:
    """Recursive character splitter with overlap.

    Tries paragraph -> line -> sentence -> word -> character separators in
    order, only descending when the current piece is too large. Mirrors the
    behaviour of LangChain's RecursiveCharacterTextSplitter without the
    dependency.
    """

    DEFAULT_SEPARATORS: tuple[str, ...] = (
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
        "",
    )

    def __init__(
        self,
        settings: Settings | None = None,
        separators: Sequence[str] | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self.chunk_size = cfg.chunk_size
        self.chunk_overlap = cfg.chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.separators: tuple[str, ...] = (
            tuple(separators) if separators is not None else self.DEFAULT_SEPARATORS
        )

    # --------------------------------------------------------------- helpers
    def _hard_split(self, text: str) -> List[str]:
        """Last-resort character split when no separator works."""
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        if separator == "":
            return self._hard_split(text)
        return [s for s in text.split(separator) if s]

    def _recursive_split(self, text: str, separators: Sequence[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        for i, sep in enumerate(separators):
            pieces = self._split_by_separator(text, sep)
            if len(pieces) > 1:
                out: List[str] = []
                for piece in pieces:
                    if len(piece) <= self.chunk_size:
                        out.append(piece)
                    else:
                        out.extend(self._recursive_split(piece, separators[i + 1 :]))
                return out
        return self._hard_split(text)

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """Greedy pack pieces up to ``chunk_size``, then add tail-overlap."""
        merged: List[str] = []
        buf = ""
        for piece in pieces:
            candidate = (buf + "\n" + piece).strip() if buf else piece
            if len(candidate) <= self.chunk_size:
                buf = candidate
            else:
                if buf:
                    merged.append(buf)
                # piece itself may exceed chunk_size if hard-splitter ran out of options
                if len(piece) <= self.chunk_size:
                    buf = piece
                else:
                    merged.extend(self._hard_split(piece))
                    buf = ""
        if buf:
            merged.append(buf)

        if self.chunk_overlap == 0 or len(merged) <= 1:
            return merged

        with_overlap = [merged[0]]
        for prev, cur in zip(merged, merged[1:]):
            tail = prev[-self.chunk_overlap :]
            with_overlap.append((tail + "\n" + cur).strip())
        return with_overlap

    # ---------------------------------------------------------------- public
    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        pieces = self._recursive_split(text, self.separators)
        return self._merge_with_overlap(pieces)

    def chunk_page(self, page: Page) -> List[Chunk]:
        return [
            Chunk(
                chunk_id=f"{page.source}::p{page.page_num}::c{i}",
                text=t,
                source=page.source,
                year=page.year,
                page_num=page.page_num,
                extracted_via=page.extracted_via,
                chunk_index=i,
            )
            for i, t in enumerate(self.split_text(page.text))
        ]

    def chunk_pages(self, pages: Iterable[Page]) -> List[Chunk]:
        out: List[Chunk] = []
        for p in pages:
            out.extend(self.chunk_page(p))
        return out
