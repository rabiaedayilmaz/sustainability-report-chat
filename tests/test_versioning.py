"""IndexVersion fingerprint + IngestManifest round-trip."""
from __future__ import annotations

import json
from pathlib import Path

from src.manifest import IngestManifest
from src.version import CODE_VERSION, SCHEMA_VERSION, IndexVersion


def _v(**overrides) -> IndexVersion:
    base = dict(
        code_version="1.0.0",
        schema_version=1,
        embedding_model="intfloat/multilingual-e5-large",
        embedding_dim=1024,
        chunk_size=800,
        chunk_overlap=120,
    )
    base.update(overrides)
    return IndexVersion(**base)


def test_fingerprint_is_stable() -> None:
    assert _v().fingerprint() == _v().fingerprint()


def test_fingerprint_changes_with_embedding_model() -> None:
    assert _v().fingerprint() != _v(embedding_model="intfloat/e5-large-v2").fingerprint()


def test_fingerprint_changes_with_dim() -> None:
    assert _v().fingerprint() != _v(embedding_dim=768).fingerprint()


def test_fingerprint_changes_with_schema() -> None:
    assert _v().fingerprint() != _v(schema_version=2).fingerprint()


def test_fingerprint_changes_with_chunk_config() -> None:
    assert _v().fingerprint() != _v(chunk_size=400).fingerprint()
    assert _v().fingerprint() != _v(chunk_overlap=60).fingerprint()


def test_fingerprint_short_and_hex() -> None:
    fp = _v().fingerprint()
    assert len(fp) == 12
    assert all(c in "0123456789abcdef" for c in fp)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    v = _v()
    m = IngestManifest.build(
        collection="testcoll",
        version=v,
        pdf_count=22,
        page_count=1234,
        chunk_count=8397,
        failed_batches=0,
    )
    m.write(tmp_path)
    loaded = IngestManifest.load(tmp_path)
    assert loaded is not None
    assert loaded.collection == "testcoll"
    assert loaded.chunk_count == 8397
    assert loaded.version["fingerprint"] == v.fingerprint()
    # Manifest is human-readable JSON.
    raw = (tmp_path / ".last_ingest_manifest.json").read_text()
    parsed = json.loads(raw)
    assert parsed["collection"] == "testcoll"


def test_constants_in_sync_with_dataclass() -> None:
    v = _v(code_version=CODE_VERSION, schema_version=SCHEMA_VERSION)
    assert v.code_version == CODE_VERSION
    assert v.schema_version == SCHEMA_VERSION
