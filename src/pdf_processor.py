"""PDF text extraction with a PaddleOCR fallback for scanned pages."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import fitz  # PyMuPDF
import numpy as np

from .config import Settings, get_settings
from .utils.log import logger


@dataclass(frozen=True)
class Page:
    """A single processed page from a PDF."""

    source: str
    year: str
    page_num: int  # 1-indexed
    text: str
    extracted_via: str  # "text" | "ocr"


class PDFProcessor:
    """Extracts cleaned text from PDFs, falling back to OCR for image-only pages.

    Text extraction is done with PyMuPDF; pages that come back nearly empty are
    re-rendered to a pixmap (still via PyMuPDF — no poppler dependency) and run
    through PaddleOCR. PaddleOCR is loaded lazily on first use because model
    download + initialization is expensive.
    """

    _MULTI_NEWLINE = re.compile(r"\n{3,}")
    _MULTI_SPACE = re.compile(r"[ \t]+")
    _PAGE_NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
    _TRAIL_PIPE_NUM = re.compile(r"\s*\|\s*\d+\s*$", re.MULTILINE)
    _SOFT_HYPHEN = "\u00ad"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._ocr_available = self._probe_ocr()
        self._ocr_engine: Any = None  # lazy init

    # ------------------------------------------------------------- internals
    @staticmethod
    def _probe_ocr() -> bool:
        try:
            import paddleocr  # noqa: F401

            return True
        except ImportError:
            logger.warning(
                "paddleocr is not installed — OCR fallback disabled. "
                "Install paddleocr + paddlepaddle to enable scanned-page extraction."
            )
            return False

    def _ensure_engine(self) -> Any:
        """Lazily build a single PaddleOCR engine; disable OCR for the run if it can't init.        """
        if not self._ocr_available:
            return None
        if self._ocr_engine is not None:
            return self._ocr_engine

        from paddleocr import PaddleOCR

        lang = self.settings.ocr_language
        angle = self.settings.ocr_use_angle_cls
        attempts: list[dict] = [
            {"lang": lang, "use_textline_orientation": angle},  # 3.x
            {"lang": lang, "use_angle_cls": angle, "show_log": False},  # 2.x verbose-off
            {"lang": lang, "use_angle_cls": angle},  # 2.x basic
            {"lang": lang},  # bare minimum
        ]
        last_exc: BaseException | None = None
        for kwargs in attempts:
            try:
                logger.info("Initialising PaddleOCR with %s", kwargs)
                self._ocr_engine = PaddleOCR(**kwargs)
                return self._ocr_engine
            except BaseException as exc:  # PaddleX raises bare SystemExit/RuntimeError too
                last_exc = exc
                logger.debug("PaddleOCR init failed with %s: %s", kwargs, exc)

        logger.warning(
            "Could not initialise PaddleOCR with any known signature (last error: %s). "
            "OCR fallback disabled for the rest of this run.",
            last_exc,
        )
        self._ocr_available = False
        return None

    def _clean(self, text: str) -> str:
        """Normalise whitespace and strip common boilerplate."""
        if not text:
            return ""
        text = text.replace("\x00", "").replace(self._SOFT_HYPHEN, "")
        # Re-join words split across line breaks: ``sustain-\nability`` -> ``sustainability``
        text = re.sub(r"-\n(\w)", r"\1", text)
        text = self._PAGE_NUM_LINE.sub("", text)
        text = self._TRAIL_PIPE_NUM.sub("", text)
        text = self._MULTI_SPACE.sub(" ", text)
        text = self._MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()

    def _render_page(self, doc: fitz.Document, page_num: int) -> np.ndarray | None:
        """Render a single PDF page to an HxWx3 BGR ``np.uint8`` array."""
        try:
            page = doc.load_page(page_num - 1)
            zoom = self.settings.ocr_dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        except Exception as exc:
            logger.warning("Render failed for p%d: %s", page_num, exc)
            return None

        # ``pix.samples`` is RGB-packed; PaddleOCR expects BGR (it uses cv2 internally).
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return np.ascontiguousarray(rgb[..., ::-1])

    @staticmethod
    def _flatten_paddle_result(result: Any) -> str:
        """Walk PaddleOCR's nested output and return joined text lines."""
        if not result:
            return ""

        lines: list[str] = []
        for page in result:
            if not page:
                continue

            # 3.x dict-shaped page result.
            if isinstance(page, dict):
                texts = page.get("rec_texts") or page.get("text") or page.get("texts")
                if isinstance(texts, (list, tuple)):
                    lines.extend(str(t) for t in texts if t)
                continue

            # 2.x list-of-items page result.
            if isinstance(page, (list, tuple)):
                for item in page:
                    text = ""
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        info = item[1]
                        if isinstance(info, (list, tuple)) and info:
                            text = str(info[0])
                        elif isinstance(info, dict):
                            text = str(info.get("text", ""))
                    elif isinstance(item, dict):
                        text = str(item.get("text", ""))
                    if text:
                        lines.append(text)
        return "\n".join(lines)

    def _run_paddle(self, engine: Any, image: np.ndarray) -> Any:
        """Call PaddleOCR using whichever method the installed version exposes."""
        # 3.x-style ``predict`` returns the new dict format.
        if hasattr(engine, "predict"):
            try:
                return engine.predict(image)
            except TypeError:
                pass  # fall through to legacy
        # 2.x ``ocr(img, cls=...)`` then ``ocr(img)``.
        try:
            return engine.ocr(image, cls=self.settings.ocr_use_angle_cls)
        except TypeError:
            return engine.ocr(image)

    def _ocr_page(self, doc: fitz.Document, page_num: int) -> str:
        engine = self._ensure_engine()
        if engine is None:
            return ""
        image = self._render_page(doc, page_num)
        if image is None:
            return ""
        try:
            result = self._run_paddle(engine, image)
        except Exception as exc:
            logger.warning("OCR failed for p%d: %s", page_num, exc)
            return ""
        return self._flatten_paddle_result(result)

    # ---------------------------------------------------------------- public
    def process(self, pdf_path: Path | str, year: str | None = None) -> Iterator[Page]:
        """Yield cleaned pages from a single PDF.

        ``year`` is inferred from the parent directory name when not given,
        matching the layout produced by ``scripts/download_pdfs.py``.
        """
        pdf_path = Path(pdf_path)
        if year is None:
            year = pdf_path.parent.name

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.error("Could not open %s: %s", pdf_path, exc)
            return

        try:
            for idx in range(1, doc.page_count + 1):
                try:
                    page = doc.load_page(idx - 1)
                    raw = page.get_text("text") or ""
                    via = "text"

                    if (
                        self.settings.enable_ocr
                        and len(raw.strip()) < self.settings.min_chars_for_ocr_skip
                    ):
                        ocr_text = self._ocr_page(doc, idx)
                        if ocr_text:
                            raw = ocr_text
                            via = "ocr"

                    cleaned = self._clean(raw)
                except Exception as exc:  # malformed page — skip, never abort the file
                    logger.warning("Skipping %s p%d: %s", pdf_path.name, idx, exc)
                    continue

                if not cleaned:
                    continue

                yield Page(
                    source=pdf_path.name,
                    year=year,
                    page_num=idx,
                    text=cleaned,
                    extracted_via=via,
                )
        finally:
            doc.close()

    def process_directory(self, root: Path | str) -> Iterator[Page]:
        """Walk ``root/<year>/*.pdf`` recursively and yield every cleaned page."""
        root = Path(root)
        pdfs = sorted(root.glob("**/*.pdf"))
        logger.info("Discovered %d PDFs under %s", len(pdfs), root)
        for pdf_path in pdfs:
            yield from self.process(pdf_path)
