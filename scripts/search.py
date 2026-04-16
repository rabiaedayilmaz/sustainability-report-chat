"""One-shot CLI search against the indexed Qdrant collection.

The year filter is inferred from the question itself (e.g. "2024", "FY2023",
"latest report"). Add ``--source FILE.pdf`` to constrain to a single document,
or ``--top-k N`` to widen retrieval.

Examples:
    python scripts/search.py "What are the 2024 emissions targets?"
    python scripts/search.py "Compare 2020 and 2024 emissions"   # 2 years -> no filter
    python scripts/search.py "renewable energy share in the latest report" --top-k 8
    python scripts/search.py "diversity policy" --json > out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import RAGPipeline  # noqa: E402
from src.utils.log import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ask a single question against the index. "
            "Year filtering is inferred from the question; no --year flag."
        )
    )
    parser.add_argument("question", help="The natural-language question.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval depth.")
    parser.add_argument("--source", type=str, default=None, help="Filter by exact PDF filename.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def render_text(result_dict: dict) -> None:
    print("\n=== Answer ===")
    print(result_dict["answer"])
    print("\n=== Sources ===")
    for s in result_dict["sources"]:
        ref = s.get("ref", "?")
        print(
            f"[{ref}] {s['source']} — page {s['page_num']} "
            f"(year={s['year']}, score={s['score']:.3f}, chunk={s['chunk_id']})"
        )
        # Indent snippet so it visually nests under its citation.
        for line in s["snippet"].splitlines():
            print(f"    {line}")
        print()


def main() -> int:
    args = parse_args()
    # CLI default: only WARNING+ on the console (we render answer/sources via print).
    setup_logging(level="DEBUG" if args.verbose else "WARNING")

    pipeline = RAGPipeline()
    # year=None → pipeline auto-extracts from the question via YearExtractor.
    result = pipeline.ask(
        args.question,
        top_k=args.top_k,
        source=args.source,
    )

    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        render_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
