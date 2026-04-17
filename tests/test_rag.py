"""Tests for the /ask endpoint and the RAGPipeline logic that backs it.

No live Qdrant, Ollama, or embedding-model is needed. Instead the suite
exercises four layers:

  * RAGPipeline's pure helpers (_validate_question, _format_context,
    _hits_to_sources, _resolve_year) — constructed via __new__ so no
    external client is ever opened.
  * The Ollama integration (_build_payload, _generate, _agenerate,
    _check_ollama) with httpx monkey-patched at the module level, so we
    verify payload shape, response parsing, and error translation
    without reaching the network.
  * The Qdrant health probe (_check_qdrant) with a stubbed vector store,
    covering both the up and down branches.
  * The /ask FastAPI route wired against a stub pipeline — verifies the
    endpoint's contract (status codes, response shape, error mapping,
    filter pass-through) end-to-end.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

import pytest
from fastapi.testclient import TestClient

import app as app_module
from src import pipeline as pipeline_module
from src.pipeline import RAGPipeline, RAGAnswer
from src.query_parser import YearExtractor
from src.vector_store import SearchHit


# ---------------------------------------------------------------- helpers


def _make_hit(
    *,
    text: str = "NTT DATA reported FY2024 Scope 1 emissions of 48,218 t-CO2e.",
    source: str = "sr2025db_all_en.pdf",
    year: str = "2025",
    page_num: int = 42,
    score: float = 0.91,
    chunk_id: Optional[str] = None,
) -> SearchHit:
    cid = chunk_id or f"{source}::p{page_num}::c0"
    return SearchHit(
        score=score,
        text=text,
        source=source,
        year=year,
        page_num=page_num,
        chunk_id=cid,
        payload={
            "source": source, "year": year, "page_num": page_num,
            "text": text, "chunk_id": cid,
        },
    )


class _StubSettings:
    """Mirrors the fields RAGPipeline reads on its ``settings`` attribute."""
    ollama_base_url = "http://ollama.test:11434"
    ollama_model = "qwen3:8b"
    ollama_temperature = 0.1
    ollama_num_ctx = 4096
    ollama_timeout = 5.0
    qdrant_collection = "test-collection"


def _pipeline_stub(store=None) -> RAGPipeline:
    """Build a RAGPipeline shell without opening any network or model handles."""
    p = RAGPipeline.__new__(RAGPipeline)
    p.year_extractor = YearExtractor(["2023", "2024", "2025"])
    p.settings = _StubSettings()
    if store is not None:
        p.store = store
    return p


class _FakeResponse:
    """Minimal httpx.Response stand-in covering only what the pipeline uses."""

    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(
                "http error", request=None, response=None,  # type: ignore[arg-type]
            )

    def json(self):
        return self._payload


# ==================================================== _validate_question


def test_validate_question_strips_whitespace() -> None:
    assert RAGPipeline._validate_question("  hi there  ") == "hi there"


def test_validate_question_rejects_empty() -> None:
    with pytest.raises(ValueError):
        RAGPipeline._validate_question("")


def test_validate_question_rejects_whitespace_only() -> None:
    with pytest.raises(ValueError):
        RAGPipeline._validate_question("   \n\t ")


def test_validate_question_rejects_too_long() -> None:
    with pytest.raises(ValueError):
        RAGPipeline._validate_question("x" * 4001)


def test_validate_question_accepts_boundary_length() -> None:
    q = "x" * 4000
    assert RAGPipeline._validate_question(q) == q


def test_validate_question_rejects_non_string() -> None:
    with pytest.raises(ValueError):
        RAGPipeline._validate_question(None)  # type: ignore[arg-type]


# ======================================================= _format_context


def test_format_context_numbers_blocks_one_indexed() -> None:
    hits = [_make_hit(text="alpha body"), _make_hit(text="beta body", page_num=43)]
    out = RAGPipeline._format_context(hits)
    assert out.startswith("[1] source=")
    assert "\n\n[2] source=" in out
    assert "alpha body" in out and "beta body" in out


def test_format_context_includes_all_metadata() -> None:
    h = _make_hit(score=0.873, source="x.pdf", year="2024", page_num=7)
    out = RAGPipeline._format_context([h])
    assert "source=x.pdf" in out
    assert "year=2024" in out
    assert "p.7" in out
    assert "0.873" in out  # score rounded to 3 decimals


def test_format_context_empty_hits() -> None:
    assert RAGPipeline._format_context([]) == ""


# ======================================================= _hits_to_sources


def test_hits_to_sources_assigns_sequential_refs() -> None:
    sources = RAGPipeline._hits_to_sources([
        _make_hit(text="a"), _make_hit(text="b"), _make_hit(text="c"),
    ])
    assert [s["ref"] for s in sources] == [1, 2, 3]


def test_hits_to_sources_truncates_long_snippets() -> None:
    long_text = "X" * 2000
    [s] = RAGPipeline._hits_to_sources([_make_hit(text=long_text)])
    # Snippet cap is 500 chars; an ellipsis is appended when truncation happens.
    assert s["snippet"].endswith("...")
    assert len(s["snippet"]) == 503


def test_hits_to_sources_keeps_short_snippet_verbatim() -> None:
    [s] = RAGPipeline._hits_to_sources([_make_hit(text="short enough")])
    assert s["snippet"] == "short enough"
    assert not s["snippet"].endswith("...")


def test_hits_to_sources_preserves_metadata() -> None:
    hit = _make_hit(source="X.pdf", year="2024", page_num=9, score=0.4242)
    [s] = RAGPipeline._hits_to_sources([hit])
    assert s["source"] == "X.pdf"
    assert s["year"] == "2024"
    assert s["page_num"] == 9
    assert s["score"] == 0.4242
    assert s["chunk_id"] == hit.chunk_id


# ========================================================== _resolve_year


def test_resolve_year_explicit_wins_over_question() -> None:
    p = _pipeline_stub()
    # Explicit argument must override any auto-detection from the text.
    assert p._resolve_year("what happened in 2025?", explicit="2024") == "2024"


def test_resolve_year_auto_extracts_from_question() -> None:
    p = _pipeline_stub()
    assert p._resolve_year("FY2024 emissions", explicit=None) == "2024"


def test_resolve_year_returns_none_when_nothing_matches() -> None:
    p = _pipeline_stub()
    assert p._resolve_year("tell me about sustainability", explicit=None) is None


# ============================================== Ollama integration (mocked)


def test_build_payload_includes_model_prompt_system_and_options() -> None:
    p = _pipeline_stub()
    hits = [_make_hit(text="chunk-one"), _make_hit(text="chunk-two", page_num=44)]
    payload = p._build_payload("How much CO2?", hits)

    # Routing fields the Ollama /api/generate endpoint requires.
    assert payload["model"] == "qwen3:8b"
    assert payload["stream"] is False
    # System prompt is injected — it's what enforces citation + anti-hallucination.
    assert "sustainability analyst" in payload["system"].lower()
    # The user prompt must contain the question and the numbered chunks.
    assert "How much CO2?" in payload["prompt"]
    assert "[1]" in payload["prompt"] and "[2]" in payload["prompt"]
    assert "chunk-one" in payload["prompt"] and "chunk-two" in payload["prompt"]
    # LLM sampling options are forwarded from settings.
    assert payload["options"] == {"temperature": 0.1, "num_ctx": 4096}


def test_build_payload_renders_k_in_template() -> None:
    p = _pipeline_stub()
    payload = p._build_payload("q", [_make_hit(), _make_hit(), _make_hit()])
    assert "top-3 retrieved chunks" in payload["prompt"]


def test_ollama_url_trims_trailing_slashes() -> None:
    p = _pipeline_stub()
    p.settings = _StubSettings()
    p.settings.ollama_base_url = "http://ollama.test:11434///"
    assert p._ollama_url() == "http://ollama.test:11434/api/generate"


def test_generate_parses_response_field(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"response": "  the answer  "})

    monkeypatch.setattr(pipeline_module.httpx, "post", fake_post)

    p = _pipeline_stub()
    out = p._generate("q", [_make_hit(text="ctx")])

    assert out == "the answer"  # .strip() applied
    assert captured["url"] == "http://ollama.test:11434/api/generate"
    assert captured["timeout"] == 5.0
    assert captured["json"]["model"] == "qwen3:8b"


def test_generate_returns_empty_string_when_response_missing(monkeypatch) -> None:
    # Ollama sometimes returns an error body without a ``response`` key.
    monkeypatch.setattr(
        pipeline_module.httpx, "post",
        lambda *a, **kw: _FakeResponse({"error": "model not pulled"}),
    )
    assert _pipeline_stub()._generate("q", [_make_hit()]) == ""


def test_generate_wraps_httpx_error_as_runtime_error(monkeypatch) -> None:
    import httpx

    def boom(*_a, **_kw):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(pipeline_module.httpx, "post", boom)
    with pytest.raises(RuntimeError, match="Ollama call failed"):
        _pipeline_stub()._generate("q", [_make_hit()])


def test_agenerate_parses_response_field(monkeypatch) -> None:
    """Async variant must use AsyncClient and still .strip() the response."""
    captured = {}

    class _FakeAsyncClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse({"response": " async answer "})

    monkeypatch.setattr(pipeline_module.httpx, "AsyncClient", _FakeAsyncClient)

    out = asyncio.run(_pipeline_stub()._agenerate("q", [_make_hit()]))
    assert out == "async answer"
    assert captured["url"] == "http://ollama.test:11434/api/generate"
    assert captured["timeout"] == 5.0


def test_agenerate_wraps_httpx_error_as_runtime_error(monkeypatch) -> None:
    import httpx

    class _ExplodingClient:
        def __init__(self, timeout): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False
        async def post(self, *_a, **_kw):
            raise httpx.ConnectError("nope")

    monkeypatch.setattr(pipeline_module.httpx, "AsyncClient", _ExplodingClient)

    with pytest.raises(RuntimeError, match="Ollama call failed"):
        asyncio.run(_pipeline_stub()._agenerate("q", [_make_hit()]))


def test_check_ollama_reports_up_on_200(monkeypatch) -> None:
    captured = {}

    def fake_get(url, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResponse({"models": [{"name": "qwen3:8b"}]})

    monkeypatch.setattr(pipeline_module.httpx, "get", fake_get)

    out = _pipeline_stub()._check_ollama()
    assert out["status"] == "up"
    assert out["model"] == "qwen3:8b"
    assert captured["url"] == "http://ollama.test:11434/api/tags"
    # Health probe must be bounded — uses HEALTH_TIMEOUT, not ollama_timeout.
    assert captured["timeout"] == RAGPipeline.HEALTH_TIMEOUT
    assert isinstance(out["latency_ms"], float)


def test_check_ollama_reports_down_on_connection_error(monkeypatch) -> None:
    import httpx

    def boom(*_a, **_kw):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(pipeline_module.httpx, "get", boom)

    out = _pipeline_stub()._check_ollama()
    assert out["status"] == "down"
    assert "connection refused" in out["error"]
    assert out["model"] == "qwen3:8b"
    assert isinstance(out["latency_ms"], float)


# ================================================== Qdrant health probe


class _FakeStore:
    """Simulates :class:`QdrantVectorStore` for health-check tests only."""

    def __init__(self, *, exists: bool = True, count: int = 123,
                 raise_on: Optional[str] = None) -> None:
        self.collection = "test-collection"
        self._exists = exists
        self._count = count
        self._raise_on = raise_on  # which method should raise

    def collection_exists(self) -> bool:
        if self._raise_on == "collection_exists":
            raise ConnectionError("qdrant unreachable")
        return self._exists

    def count(self) -> int:
        if self._raise_on == "count":
            raise ConnectionError("qdrant unreachable")
        return self._count


def test_check_qdrant_reports_up_with_chunk_count() -> None:
    p = _pipeline_stub(store=_FakeStore(exists=True, count=9999))
    out = p._check_qdrant()
    assert out["status"] == "up"
    assert out["collection_exists"] is True
    assert out["indexed_chunks"] == 9999
    assert out["collection"] == "test-collection"


def test_check_qdrant_skips_count_when_collection_missing() -> None:
    # If the collection doesn't exist yet, ``count()`` must not be called
    # (would raise in the real client). Probe still reports up.
    p = _pipeline_stub(store=_FakeStore(exists=False))
    out = p._check_qdrant()
    assert out["status"] == "up"
    assert out["collection_exists"] is False
    assert out["indexed_chunks"] == 0


def test_check_qdrant_reports_down_on_connection_error() -> None:
    p = _pipeline_stub(store=_FakeStore(raise_on="collection_exists"))
    out = p._check_qdrant()
    assert out["status"] == "down"
    assert "qdrant unreachable" in out["error"]
    assert out["collection"] == "test-collection"


# ================================================= /ask endpoint (stubbed)


class StubPipeline:
    """Minimal pipeline stub exposing exactly the surface app.py consumes.

    - ``aask`` records its last call (so filter pass-through can be asserted)
      and returns a canned answer + stubbed sources, mirroring the real
      pipeline's RAGAnswer contract.
    """

    class _Settings:
        qdrant_collection = "test"
        embedding_model = "stub"
        ollama_model = "stub"

    def __init__(self) -> None:
        self.settings = self._Settings()
        self._hits: List[SearchHit] = []
        self._answer: str = "stubbed answer [1]"
        self._health = {
            "qdrant":   {"status": "up"},
            "ollama":   {"status": "up"},
            "embedder": {"status": "up"},
        }
        self.last_call: Optional[dict] = None
        self.raise_on_ask: Optional[BaseException] = None

    async def aask(self, question, top_k=None, year=None, source=None):
        self.last_call = dict(question=question, top_k=top_k, year=year, source=source)
        if self.raise_on_ask is not None:
            raise self.raise_on_ask
        if not self._hits:
            return RAGAnswer(
                answer="I could not retrieve any relevant context from the reports.",
                sources=[],
            )
        return RAGAnswer(
            answer=self._answer,
            sources=RAGPipeline._hits_to_sources(self._hits),
        )

    def health(self) -> dict:
        return self._health


@pytest.fixture
def stub_pipeline():
    """Install a stub pipeline on app._State.pipeline for one test."""
    prev = app_module._State.pipeline
    stub = StubPipeline()
    app_module._State.pipeline = stub  # type: ignore[assignment]
    try:
        yield stub
    finally:
        app_module._State.pipeline = prev


@pytest.fixture
def client(stub_pipeline):
    # Build TestClient without entering its context manager: the real lifespan
    # would instantiate a live RAGPipeline (which needs Qdrant). Our stub is
    # already wired in, so requests go straight to the route handlers.
    return TestClient(app_module.app)


# -------------------------------------------------- /ask happy path


def test_ask_returns_answer_and_sources(client, stub_pipeline) -> None:
    stub_pipeline._hits = [_make_hit(text="alpha body")]
    stub_pipeline._answer = "Scope 1 is 48,218 t-CO2e [1]"

    r = client.post("/ask", json={"question": "what is scope 1?"})
    assert r.status_code == 200

    body = r.json()
    assert body["answer"] == "Scope 1 is 48,218 t-CO2e [1]"
    assert len(body["sources"]) == 1
    [s] = body["sources"]
    assert s["ref"] == 1
    assert s["source"] == "sr2025db_all_en.pdf"
    assert s["snippet"] == "alpha body"
    assert s["chunk_id"].startswith("sr2025db_all_en.pdf::p")


def test_ask_no_hits_returns_canonical_message(client, stub_pipeline) -> None:
    # No retrieval hits → the pipeline short-circuits with a fixed message.
    stub_pipeline._hits = []

    r = client.post("/ask", json={"question": "unknown question"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "I could not retrieve any relevant context from the reports."
    assert body["sources"] == []


def test_ask_forwards_top_k_year_and_source(client, stub_pipeline) -> None:
    r = client.post("/ask", json={
        "question": "FY2024 waste?",
        "top_k": 3,
        "year": "2025",
        "source": "sr2025db_all_en.pdf",
    })
    assert r.status_code == 200
    assert stub_pipeline.last_call == {
        "question": "FY2024 waste?",
        "top_k": 3,
        "year": "2025",
        "source": "sr2025db_all_en.pdf",
    }


def test_ask_defaults_optional_fields_to_none(client, stub_pipeline) -> None:
    r = client.post("/ask", json={"question": "hello"})
    assert r.status_code == 200
    assert stub_pipeline.last_call == {
        "question": "hello", "top_k": None, "year": None, "source": None,
    }


# -------------------------------------------------- /ask request validation


def test_ask_empty_question_422(client) -> None:
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_ask_missing_question_field_422(client) -> None:
    r = client.post("/ask", json={"top_k": 3})
    assert r.status_code == 422


def test_ask_top_k_below_min_422(client) -> None:
    r = client.post("/ask", json={"question": "hi", "top_k": 0})
    assert r.status_code == 422


def test_ask_top_k_above_max_422(client) -> None:
    r = client.post("/ask", json={"question": "hi", "top_k": 1000})
    assert r.status_code == 422


def test_ask_question_too_long_422(client) -> None:
    r = client.post("/ask", json={"question": "x" * 4001})
    assert r.status_code == 422


# -------------------------------------------------- /ask error mapping


def test_ask_value_error_maps_to_400(client, stub_pipeline) -> None:
    # Pipeline-raised ValueError (e.g. downstream validation) → HTTP 400.
    stub_pipeline.raise_on_ask = ValueError("bad input")
    r = client.post("/ask", json={"question": "anything"})
    assert r.status_code == 400
    assert "bad input" in r.json()["detail"]


def test_ask_runtime_error_maps_to_502(client, stub_pipeline) -> None:
    # Upstream Ollama / Qdrant unreachable → HTTP 502 Bad Gateway.
    stub_pipeline.raise_on_ask = RuntimeError("ollama down")
    r = client.post("/ask", json={"question": "anything"})
    assert r.status_code == 502
    assert "ollama down" in r.json()["detail"]


def test_ask_returns_503_when_pipeline_not_initialised() -> None:
    # Simulate a request that arrives before lifespan finished loading.
    prev = app_module._State.pipeline
    app_module._State.pipeline = None
    try:
        tc = TestClient(app_module.app)
        r = tc.post("/ask", json={"question": "hi"})
        assert r.status_code == 503
        assert "not initialised" in r.json()["detail"].lower()
    finally:
        app_module._State.pipeline = prev
