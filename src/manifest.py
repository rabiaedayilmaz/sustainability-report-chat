"""Per-ingest manifest — written after every successful index run.

Why: months later, when retrieval quality drops or you wonder whether you need
to re-index, the manifest tells you exactly what is in the collection: which
embedding model, what schema version, when, on which git SHA, how many chunks.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .utils.log import logger
from .version import IndexVersion

MANIFEST_FILENAME = ".last_ingest_manifest.json"


def _git_sha() -> Optional[str]:
    """Return short git SHA if we're in a git repo, else None."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return None


@dataclass
class IngestManifest:
    """Snapshot of an ingest run — written next to ``data/`` for traceability."""

    timestamp: str
    collection: str
    version: dict  # IndexVersion.to_dict()
    pdf_count: int
    page_count: int
    chunk_count: int
    failed_batches: int
    git_sha: Optional[str] = None
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")

    @classmethod
    def build(
        cls,
        *,
        collection: str,
        version: IndexVersion,
        pdf_count: int,
        page_count: int,
        chunk_count: int,
        failed_batches: int,
    ) -> "IngestManifest":
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            collection=collection,
            version=version.to_dict(),
            pdf_count=pdf_count,
            page_count=page_count,
            chunk_count=chunk_count,
            failed_batches=failed_batches,
            git_sha=_git_sha(),
        )

    # ----------------------------------------------------------------- I/O
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def write(self, dest_dir: Path | str) -> Path:
        dest = Path(dest_dir) / MANIFEST_FILENAME
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(self.to_json(), encoding="utf-8")
            logger.info("Wrote ingest manifest to %s", dest)
        except OSError as exc:
            # Non-fatal: ingest still succeeded; we just couldn't persist the manifest.
            logger.warning("Could not write manifest to %s: %s", dest, exc)
        return dest

    @classmethod
    def load(cls, src_dir: Path | str) -> Optional["IngestManifest"]:
        src = Path(src_dir) / MANIFEST_FILENAME
        if not src.exists():
            return None
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
            return cls(**data)
        except Exception as exc:
            logger.warning("Could not read manifest %s: %s", src, exc)
            return None
