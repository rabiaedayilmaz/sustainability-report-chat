"""Extract structured filters (currently just year) from a free-text question.

Kept dependency-free on purpose: spaCy / dateparser would pull >100 MB for
something we can solve precisely with a handful of regexes against a known
vocabulary (the report years actually present on disk).
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Set

from .utils.log import logger

# Most-specific pattern first; the third matches bare 4-digit years anywhere.
_YEAR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bFY\s*(20\d{2})\b", re.IGNORECASE),
    re.compile(r"\bfiscal\s+(?:year\s+)?(20\d{2})\b", re.IGNORECASE),
    re.compile(r"\b(19\d{2}|20\d{2})\b"),
)

_LATEST_RE = re.compile(
    r"\b(?:latest|most[\s-]+recent|newest|current)\b(?:\s+(?:report|year|data|figures|results))?",
    re.IGNORECASE,
)
_LAST_YEAR_RE = re.compile(r"\b(?:last|previous|prior)\s+year\b", re.IGNORECASE)
_THIS_YEAR_RE = re.compile(r"\b(?:this|current)\s+year\b", re.IGNORECASE)


class YearExtractor:
    """Pulls a single ``year`` filter from a question, if and only if unambiguous.

    Rules (in order):
      1. **0 valid years** found → ``None`` (no filter; semantic search alone).
      2. **Exactly 1** in-range year explicitly mentioned → filter by it.
      3. **2+ distinct** in-range years mentioned → ``None`` (user is comparing).
      4. **"latest" / "most recent"** phrasing → max indexed year.
      5. **"last year" / "this year"** → calendar arithmetic, clamped to indexed range.

    "In-range" means the year actually exists in the indexed corpus, so phrases
    like "net-zero by 2050" do not accidentally constrain retrieval to the (non-
    existent) 2050 report.
    """

    def __init__(self, available_years: Iterable[str]) -> None:
        self._years: Set[str] = {str(y) for y in available_years if str(y).isdigit()}
        self._max_year: Optional[str] = max(self._years) if self._years else None

    # ------------------------------------------------------------- factories
    @classmethod
    def from_data_dir(cls, data_dir: Path | str) -> "YearExtractor":
        """Discover indexed years from ``<data_dir>/<YYYY>/`` directory layout.

        Returns an empty extractor (filter never applied) if the directory does
        not exist — safe default for containers without the data volume mounted.
        """
        years: list[str] = []
        try:
            for child in Path(data_dir).iterdir():
                if child.is_dir() and child.name.isdigit() and len(child.name) == 4:
                    years.append(child.name)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            logger.info("YearExtractor: data dir '%s' unavailable; auto-filter disabled.", data_dir)
        return cls(years)

    # -------------------------------------------------------------- exposed
    @property
    def available_years(self) -> Set[str]:
        return set(self._years)

    def extract(self, question: str) -> Optional[str]:
        """Return the inferred year filter, or ``None`` if nothing safe to apply."""
        if not self._years or not question:
            return None

        text = question.strip()

        # 1. Explicit year mentions (any pattern).
        mentioned: Set[str] = set()
        for pat in _YEAR_PATTERNS:
            for m in pat.finditer(text):
                mentioned.add(m.group(1))
        in_range = mentioned & self._years

        if len(in_range) == 1:
            return next(iter(in_range))
        if len(in_range) >= 2:
            return None  # comparison query — let semantic ranker decide

        # 2. Relative phrases (only when no in-range year was already chosen).
        if _LATEST_RE.search(text):
            return self._max_year

        today = date.today().year
        if _THIS_YEAR_RE.search(text):
            candidate = str(today)
            return candidate if candidate in self._years else self._max_year
        if _LAST_YEAR_RE.search(text):
            candidate = str(today - 1)
            return candidate if candidate in self._years else self._max_year

        return None
