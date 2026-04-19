"""On-disk cache of extracted PDF pages, one ``.txt`` per page.

Separating OCR (which writes the cache) from embedding (which reads and then
deletes it) keeps PaddleOCR and the sentence-transformers model from
coexisting in RAM — the combination is what pushes low-RAM machines into
swap.

Layout::

    <data_dir>/.cache/pages/<year>/<pdf_stem>/
        p0001.txt          # page 1, extracted via native text
        p0002.ocr.txt      # page 2, extracted via OCR fallback
        ...

The ``.ocr`` infix records how the page was extracted so the embed phase can
preserve it in the Qdrant payload. The text file itself holds the cleaned
page text as UTF-8, nothing else.

A PDF's cache is written atomically: pages land in a ``*.tmp`` sibling
directory first and the final rename happens only after the last page is
flushed. ``is_cached`` therefore never returns True for a half-written PDF.

After a PDF's pages have been embedded and upserted, the embed phase can
delete its cache directory to keep peak disk usage bounded to roughly one
PDF's worth of text at a time.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Iterator

from .pdf_processor import Page
from .utils.log import logger

CACHE_SUBDIR = ".cache/pages"
_OCR_SUFFIX = ".ocr.txt"
_PLAIN_SUFFIX = ".txt"
_PAGE_NAME = re.compile(r"^p(\d{4,})(\.ocr)?\.txt$")


def cache_root(data_dir: Path | str) -> Path:
    return Path(data_dir) / CACHE_SUBDIR


def cache_dir_for(data_dir: Path | str, pdf_path: Path | str) -> Path:
    """Directory under which a single PDF's page files live."""
    pdf_path = Path(pdf_path)
    return cache_root(data_dir) / pdf_path.parent.name / pdf_path.stem


def is_cached(data_dir: Path | str, pdf_path: Path | str) -> bool:
    """True iff a completed cache directory exists for this PDF."""
    return cache_dir_for(data_dir, pdf_path).is_dir()


def _page_filename(page: Page) -> str:
    suffix = _OCR_SUFFIX if page.extracted_via == "ocr" else _PLAIN_SUFFIX
    return f"p{page.page_num:04d}{suffix}"


def write_pages(
    data_dir: Path | str,
    pdf_path: Path | str,
    pages: Iterator[Page],
) -> int:
    """Write ``pages`` as one file each, atomically; return the number written.

    Pages land in a ``<dest>.tmp`` directory; after the iterator is exhausted
    the tmp dir is renamed into place. A crash leaves only the tmp dir, which
    ``is_cached`` ignores.
    """
    dest = cache_dir_for(data_dir, pdf_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    count = 0
    try:
        for p in pages:
            (tmp / _page_filename(p)).write_text(p.text, encoding="utf-8")
            count += 1
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return count


def list_cached_dirs(data_dir: Path | str) -> list[Path]:
    """Return every completed PDF cache directory, sorted."""
    root = cache_root(data_dir)
    if not root.is_dir():
        return []
    out: list[Path] = []
    for year_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for stem_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
            if stem_dir.name.endswith(".tmp"):
                continue  # half-written; skip
            out.append(stem_dir)
    return out


def iter_pages_in_dir(cache_dir: Path) -> Iterator[Page]:
    """Stream every page from a single PDF's cache directory in page order."""
    year = cache_dir.parent.name
    source = f"{cache_dir.name}.pdf"

    entries: list[tuple[int, Path, str]] = []
    for entry in cache_dir.iterdir():
        if not entry.is_file():
            continue
        m = _PAGE_NAME.match(entry.name)
        if not m:
            continue
        page_num = int(m.group(1))
        via = "ocr" if m.group(2) else "text"
        entries.append((page_num, entry, via))

    entries.sort(key=lambda t: t[0])
    for page_num, path, via in entries:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        yield Page(
            source=source,
            year=year,
            page_num=page_num,
            text=text,
            extracted_via=via,
        )


def remove_cache(cache_dir: Path) -> None:
    """Delete a single PDF's cache directory; log and swallow OS errors."""
    try:
        shutil.rmtree(cache_dir)
    except OSError as exc:
        logger.warning("Could not delete cache %s: %s", cache_dir, exc)


def discover_pdfs(data_dir: Path | str) -> list[Path]:
    """List every ``*.pdf`` under ``data_dir`` excluding the cache directory."""
    data_dir = Path(data_dir)
    return [
        p
        for p in sorted(data_dir.glob("**/*.pdf"))
        if ".cache" not in p.parts
    ]
