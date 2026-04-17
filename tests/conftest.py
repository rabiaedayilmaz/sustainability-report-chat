"""Shared test setup.

Seed dummy values for the two required Settings fields so tests that import
``app`` (and therefore trigger Settings validation) run in any environment —
no real Qdrant credentials needed. These are only consumed by the real
clients, which the test suite never constructs.
"""
from __future__ import annotations

import os

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-key")
