"""Index every sustainability report PDF under ``data/`` into Qdrant.

Two-phase pipeline — run them as separate invocations when memory is tight so
PaddleOCR and the embedding model never live in the same process::

    python scripts/index_pdfs.py --phase extract         # OCR -> JSONL cache
    python scripts/index_pdfs.py --phase embed           # cache -> Qdrant
    python scripts/index_pdfs.py --phase embed --recreate  # drop + rebuild

    python scripts/index_pdfs.py                         # both, in one process
    python scripts/index_pdfs.py --recreate              # both, drop collection
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
        "--phase",
        choices=["extract", "embed", "all"],
        default="all",
        help=(
            "Which phase(s) to run. 'extract' writes JSONL caches via OCR; "
            "'embed' reads caches and upserts to Qdrant; 'all' does both."
        ),
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the Qdrant collection before embedding (use after schema changes).",
    )
    parser.add_argument(
        "--recreate-cache",
        action="store_true",
        help="Re-extract PDFs even if a cached page directory already exists.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not delete a PDF's cached pages after they have been embedded.",
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

    print(f"Phase: {args.phase} | data='{data_dir}' | collection='{settings.qdrant_collection}'")
    if args.phase in ("embed", "all"):
        print(
            f"Embedding backend/model: {settings.embedding_backend} / "
            f"{settings.embedding_model} ({settings.embedding_device})"
        )
    if args.recreate and args.phase in ("embed", "all"):
        print("(Existing collection will be dropped first.)")
    if args.recreate_cache and args.phase in ("extract", "all"):
        print("(Existing JSONL caches will be overwritten.)")

    pipeline = RAGPipeline(settings)

    if args.phase == "extract":
        stats = pipeline.extract_to_cache(
            data_dir=data_dir,
            recreate_cache=args.recreate_cache,
        )
        print(
            f"Extract done. pdfs_written={stats['pdfs_written']} "
            f"pdfs_cached={stats['pdfs_cached']} pages={stats['pages']}"
        )
    elif args.phase == "embed":
        stats = pipeline.embed_from_cache(
            data_dir=data_dir,
            recreate=args.recreate,
            keep_cache=args.keep_cache,
        )
        print(
            f"Embed done. pages={stats['pages']} chunks={stats['chunks']} "
            f"collection_total={stats['collection_total']}"
        )
    else:  # all
        stats = pipeline.ingest(
            data_dir=data_dir,
            recreate=args.recreate,
            recreate_cache=args.recreate_cache,
            keep_cache=args.keep_cache,
        )
        print(
            f"Done. pages={stats['pages']} chunks={stats['chunks']} "
            f"collection_total={stats['collection_total']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
