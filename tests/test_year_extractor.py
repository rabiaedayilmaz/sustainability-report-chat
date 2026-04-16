"""YearExtractor — tests the rule table without touching the filesystem."""
from __future__ import annotations

from datetime import date

from src.query_parser import YearExtractor

YEARS = [str(y) for y in range(2009, 2026)]  # 2009..2025 like our actual corpus


def make() -> YearExtractor:
    return YearExtractor(YEARS)


def test_explicit_year_in_range() -> None:
    assert make().extract("What are the 2024 emissions targets?") == "2024"


def test_fy_pattern() -> None:
    assert make().extract("FY2023 indicators") == "2023"


def test_fiscal_year_pattern() -> None:
    assert make().extract("fiscal year 2022 results") == "2022"


def test_two_in_range_years_returns_none() -> None:
    # Comparison query — let semantic ranker decide rather than over-filter.
    assert make().extract("Compare 2020 and 2024 emissions") is None


def test_out_of_range_year_ignored() -> None:
    # 2050 is a future-target mention, not a report year — must not filter.
    assert make().extract("What is the net-zero target by 2050?") is None


def test_no_year_returns_none() -> None:
    assert make().extract("Tell me about sustainability initiatives") is None


def test_latest_resolves_to_max() -> None:
    assert make().extract("What's in the latest report?") == "2025"
    assert make().extract("most-recent diversity policy") == "2025"


def test_filename_mention_does_not_match() -> None:
    # "sr2024.pdf" — the leading 'r' is a word char so \b doesn't fire.
    assert make().extract("sr2024.pdf summary") is None


def test_one_in_range_among_out_of_range_wins() -> None:
    assert make().extract("targets for 2024 to be achieved by 2030") == "2024"


def test_relative_dates_clamped_to_range() -> None:
    today = date.today().year
    ye = make()
    last = str(today - 1)
    if last in YEARS:
        assert ye.extract("last year emissions") == last
    else:
        assert ye.extract("last year emissions") == "2025"  # max-year fallback


def test_empty_extractor_never_filters() -> None:
    # Container/test environments without the data dir mounted.
    ye = YearExtractor([])
    assert ye.extract("2024 emissions") is None
