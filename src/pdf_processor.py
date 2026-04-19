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


# Cap on any side of a rendered page image. At 3 B/pixel this keeps a single
# page under ~30 MB even if ocr_dpi combined with a huge media box would blow up.
_MAX_PIXELS_PER_SIDE = 3000


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

    Text extraction uses PyMuPDF; pages that come back nearly empty are rendered
    to a pixmap and run through PaddleOCR. Pages are processed one at a time and
    large intermediates (pixmap, numpy image, raw text) are released before the
    next page is touched, so memory stays flat per-page instead of growing with
    the PDF.
    """

    _MULTI_NEWLINE = re.compile(r"\n{3,}")
    _MULTI_SPACE = re.compile(r"[ \t]+")
    _PAGE_NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
    _TRAIL_PIPE_NUM = re.compile(r"\s*\|\s*\d+\s*$", re.MULTILINE)
    _HYPHEN_WRAP = re.compile(r"-\n(\w)")
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
        """Lazily build a single PaddleOCR engine; disable OCR for the run if it can't init."""
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
        # Re-join words split across line breaks: ``sustain-\nability`` -> ``sustainability``.
        text = self._HYPHEN_WRAP.sub(r"\1", text)
        text = self._PAGE_NUM_LINE.sub("", text)
        text = self._TRAIL_PIPE_NUM.sub("", text)
        text = self._MULTI_SPACE.sub(" ", text)
        text = self._MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()

    def _zoom_for(self, page: fitz.Page) -> float:
        """Zoom factor honouring ``ocr_dpi`` but capped so no side exceeds the limit."""
        zoom = self.settings.ocr_dpi / 72.0
        rect = page.rect
        max_side = max(rect.width, rect.height) * zoom
        if max_side > _MAX_PIXELS_PER_SIDE:
            zoom *= _MAX_PIXELS_PER_SIDE / max_side
        return zoom

    def _render_page(self, page: fitz.Page) -> np.ndarray | None:
        """Render ``page`` to a contiguous HxWx3 BGR ``np.uint8`` array.

        The pixmap is released as soon as its pixels have been copied into an
        owned numpy buffer, so peak memory is ~one page image rather than two.
        """
        try:
            zoom = self._zoom_for(page)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        except Exception as exc:
            logger.warning("Render failed for p%d: %s", page.number + 1, exc)
            return None

        try:
            h, w = pix.height, pix.width
            # ``samples_mv`` is a zero-copy memoryview into the pixmap; fall back
            # to ``samples`` (a bytes copy) on older PyMuPDF.
            buf = getattr(pix, "samples_mv", None) or pix.samples
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()
        finally:
            pix = None  # release the MuPDF pixmap immediately

        # In-place RGB -> BGR (PaddleOCR uses cv2 internally). Uses a H*W temp
        # — a third of a full image copy.
        tmp = arr[:, :, 0].copy()
        arr[:, :, 0] = arr[:, :, 2]
        arr[:, :, 2] = tmp
        return arr

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
        if hasattr(engine, "predict"):
            try:
                return engine.predict(image)
            except TypeError:
                pass  # fall through to legacy
        try:
            return engine.ocr(image, cls=self.settings.ocr_use_angle_cls)
        except TypeError:
            return engine.ocr(image)

    def _ocr_page(self, page: fitz.Page) -> str:
        engine = self._ensure_engine()
        if engine is None:
            return ""
        image = self._render_page(page)
        if image is None:
            return ""
        try:
            result = self._run_paddle(engine, image)
        except Exception as exc:
            logger.warning("OCR failed for p%d: %s", page.number + 1, exc)
            return ""
        finally:
            image = None  # drop the big array before walking the (small) result
        return self._flatten_paddle_result(result)

    # ---------------------------------------------------------------- public
    def process(self, pdf_path: Path | str, year: str | None = None) -> Iterator[Page]:
        """Yield cleaned pages from a single PDF, one page at a time.

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
                via = "text"
                cleaned = ""
                try:
                    page = doc.load_page(idx - 1)
                    raw = page.get_text("text") or ""

                    if (
                        self.settings.enable_ocr
                        and len(raw.strip()) < self.settings.min_chars_for_ocr_skip
                    ):
                        ocr_text = self._ocr_page(page)
                        if ocr_text:
                            raw = ocr_text
                            via = "ocr"
                        ocr_text = ""  # release

                    cleaned = self._clean(raw)
                    raw = ""  # release before the generator suspends on yield
                    page = None
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
