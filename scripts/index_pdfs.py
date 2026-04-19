"""Index every sustainability report PDF under ``data/`` into Qdrant.

Two-phase pipeline. ``--phase all`` (the default) runs the extract phase in
a **child process** and embeds in the parent, so PaddleOCR's native
allocations are fully reclaimed by the OS before the sentence-transformers
model is loaded. That keeps peak RAM at max(OCR, embed) instead of
(OCR + embed).

Usage::

    python scripts/index_pdfs.py                         # extract (subproc) + embed
    python scripts/index_pdfs.py --recreate              # same, drop collection first
    python scripts/index_pdfs.py --phase extract         # only OCR -> data/ocr/*.txt
    python scripts/index_pdfs.py --phase embed           # only cache -> Qdrant
    python scripts/index_pdfs.py --phase embed --recreate  # drop + rebuild
"""
from __future__ import annotations

import argparse
import subprocess
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
            "Which phase(s) to run. 'extract' writes page .txt files via OCR; "
            "'embed' reads them and upserts to Qdrant; 'all' runs extract in a "
            "child process then embeds in the parent."
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
        "--cleanup",
        action="store_true",
        help="Delete each PDF's cached .txt pages after they have been embedded.",
    )
    parser.add_argument(
        "--embedder",
        choices=["e5", "harrier"],
        default=None,
        help="Embedding backend override (defaults to EMBEDDING_BACKEND/.env).",
    )
    parser.add_argument(
        "--no-subprocess",
        action="store_true",
        help=(
            "Force --phase all to stay in the same process. Uses less latency "
            "but pays the 'OCR + embed in RAM together' cost."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def _run_extract_subprocess(args: argparse.Namespace) -> int:
    """Invoke this same script with --phase extract in a fresh Python process.

    On exit the OS reclaims everything Paddle allocated, including native
    C++ buffers that ``del engine; gc.collect()`` cannot return to the heap.
    """
    argv = [sys.executable, str(Path(__file__).resolve()), "--phase", "extract"]
    if args.data_dir is not None:
        argv += ["--data-dir", str(args.data_dir)]
    if args.recreate_cache:
        argv += ["--recreate-cache"]
    if args.verbose:
        argv += ["-v"]
    print(f"[all] spawning extract subprocess: {' '.join(argv)}")
    return subprocess.call(argv)


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

    # For --phase all: run extract as a child process, then embed in the parent.
    # This is the path that actually keeps peak RAM low — native Paddle memory
    # only goes back to the OS when that subprocess exits.
    if args.phase == "all" and not args.no_subprocess:
        rc = _run_extract_subprocess(args)
        if rc != 0:
            print(f"Extract subprocess exited with {rc}; skipping embed.")
            return rc
        args.phase = "embed"

    print(f"Phase: {args.phase} | data='{data_dir}' | collection='{settings.qdrant_collection}'")
    if args.phase in ("embed", "all"):
        print(
            f"Embedding backend/model: {settings.embedding_backend} / "
            f"{settings.embedding_model} ({settings.embedding_device})"
        )
    if args.recreate and args.phase in ("embed", "all"):
        print("(Existing collection will be dropped first.)")
    if args.recreate_cache and args.phase in ("extract", "all"):
        print("(Existing caches will be overwritten.)")

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
            cleanup=args.cleanup,
        )
        print(
            f"Embed done. pages={stats['pages']} chunks={stats['chunks']} "
            f"collection_total={stats['collection_total']}"
        )
    else:  # all with --no-subprocess
        stats = pipeline.ingest(
            data_dir=data_dir,
            recreate=args.recreate,
            recreate_cache=args.recreate_cache,
            cleanup=args.cleanup,
        )
        print(
            f"Done. pages={stats['pages']} chunks={stats['chunks']} "
            f"collection_total={stats['collection_total']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
