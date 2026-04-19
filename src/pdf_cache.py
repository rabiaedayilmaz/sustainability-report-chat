"""On-disk cache of extracted PDF pages, one JSONL per source PDF.

Separating OCR (which writes the cache) from embedding (which reads it) keeps
PaddleOCR and the sentence-transformers model from coexisting in RAM — the
combination is what pushes low-RAM machines into swap. As a side benefit the
extract phase becomes resumable: a PDF whose JSONL is already present is
skipped on subsequent runs.

Layout::

    <data_dir>/.cache/pages/<year>/<pdf_stem>.jsonl

Each line is one Page serialised as JSON::

    {"page_num": int, "text": str, "extracted_via": "text"|"ocr"}

Writes go through a ``*.tmp`` sibling and ``os.replace`` so a killed extract
never leaves a partial JSONL that :func:`is_cached` would falsely accept.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

from .pdf_processor import Page
from .utils.log import logger

CACHE_SUBDIR = ".cache/pages"


def cache_root(data_dir: Path | str) -> Path:
    return Path(data_dir) / CACHE_SUBDIR


def cache_path(data_dir: Path | str, pdf_path: Path | str) -> Path:
    """Return the JSONL cache path that mirrors ``pdf_path`` under ``data_dir``."""
    pdf_path = Path(pdf_path)
    year = pdf_path.parent.name
    return cache_root(data_dir) / year / f"{pdf_path.stem}.jsonl"


def is_cached(data_dir: Path | str, pdf_path: Path | str) -> bool:
    return cache_path(data_dir, pdf_path).is_file()


def write_pages(
    data_dir: Path | str,
    pdf_path: Path | str,
    pages: Iterable[Page],
) -> int:
    """Write ``pages`` as JSONL atomically; return the number of lines written."""
    dest = cache_path(data_dir, pdf_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    count = 0
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for p in pages:
                f.write(
                    json.dumps(
                        {
                            "page_num": p.page_num,
                            "text": p.text,
                            "extracted_via": p.extracted_via,
                        },
                        ensure_ascii=False,
                    )
                )
                f.write("\n")
                count += 1
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return count


def list_caches(data_dir: Path | str) -> list[Path]:
    """Return every JSONL cache path under ``data_dir``, sorted."""
    root = cache_root(data_dir)
    if not root.is_dir():
        return []
    return sorted(root.glob("*/*.jsonl"))


def iter_cached_pages(data_dir: Path | str) -> Iterator[Page]:
    """Stream every cached page under ``data_dir`` as a :class:`Page`.

    The source PDF does not need to exist anymore — ``source`` and ``year``
    are reconstructed from the cache path (``<year>/<stem>.jsonl`` → ``stem.pdf``).
    """
    for jsonl in list_caches(data_dir):
        year = jsonl.parent.name
        source = f"{jsonl.stem}.pdf"
        try:
            fh = jsonl.open("r", encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not open cache %s: %s", jsonl, exc)
            continue
        with fh as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line in %s: %s", jsonl, exc)
                    continue
                yield Page(
                    source=source,
                    year=year,
                    page_num=int(row["page_num"]),
                    text=row.get("text", "") or "",
                    extracted_via=row.get("extracted_via", "text") or "text",
                )


def discover_pdfs(data_dir: Path | str) -> list[Path]:
    """List every ``*.pdf`` under ``data_dir`` excluding the cache directory."""
    data_dir = Path(data_dir)
    return [
        p
        for p in sorted(data_dir.glob("**/*.pdf"))
        if ".cache" not in p.parts
    ]
