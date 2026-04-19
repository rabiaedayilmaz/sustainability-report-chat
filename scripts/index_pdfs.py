"""Index every sustainability report PDF under ``data/`` into Qdrant.

Usage:
    python scripts/index_pdfs.py            # incremental upsert
    python scripts/index_pdfs.py --recreate # drop + rebuild collection
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402
from src.embedder import E5_DEFAULT_MODEL, HARRIER_DEFAULT_MODEL  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.utils.log import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index PDFs into Qdrant.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the data directory (default: settings.data_dir).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the collection before re-ingesting (use after schema changes).",
    )
    parser.add_argument(
        "--embedder",
        choices=["e5", "harrier"],
        default=None,
        help="Embedding backend override (defaults to EMBEDDING_BACKEND/.env).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level="DEBUG" if args.verbose else None)

    settings = get_settings()
    if args.embedder is not None:
        settings.embedding_backend = args.embedder
        # Convenience default: if user picks Harrier but left the E5 default model,
        # switch to the expected Harrier model automatically.
        if args.embedder == "harrier" and settings.embedding_model == E5_DEFAULT_MODEL:
            settings.embedding_model = HARRIER_DEFAULT_MODEL
    data_dir = args.data_dir or settings.data_dir

    print(f"Indexing PDFs from '{data_dir}' into collection '{settings.qdrant_collection}'")
    print(
        f"Embedding backend/model: {settings.embedding_backend} / "
        f"{settings.embedding_model} ({settings.embedding_device})"
    )
    if args.recreate:
        print("(Existing collection will be dropped first.)")

    pipeline = RAGPipeline(settings)
    stats = pipeline.ingest(data_dir=data_dir, recreate=args.recreate)
    print(
        f"Done. pages={stats['pages']} chunks={stats['chunks']} "
        f"collection_total={stats['collection_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
