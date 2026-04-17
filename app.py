"""FastAPI service exposing the sustainability-report RAG pipeline.

Endpoints:
    POST /ask      → {question, top_k?, year?, source?} → {answer, sources}
    GET  /health   → pipeline + dependency status

Run locally:
    uvicorn app:app --reload --port 8000
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent import RAGAgent
from src.config import get_settings
from src.pipeline import RAGPipeline
from src.utils.log import logger, setup_logging

SERVICE_NAME = "sustainability-rag"
SERVICE_VERSION = "1.0.0"


# =============================================================== DTOs
class AskRequest(BaseModel):
    """Body of ``POST /ask``."""

    question: str = Field(..., min_length=1, max_length=4000, description="Natural-language question.")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Override retrieval depth.")
    year: Optional[str] = Field(None, description="Filter by report year (e.g. '2024').")
    source: Optional[str] = Field(None, description="Filter by exact PDF filename.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What are NTT DATA's 2024 emissions targets?",
                "top_k": 6,
                "year": "2024",
            }
        }
    }


class SourceItem(BaseModel):
    """One retrieved chunk surfaced alongside the answer for verification."""

    ref: int = Field(..., description="Reference number cited inline in the answer as [N].")
    source: str
    year: str
    page_num: int
    chunk_id: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


# ----- /agent ---------------------------------------------------------------
class AgentRequest(BaseModel):
    """Body of ``POST /agent`` — the tool-using variant of /ask."""

    question: str = Field(..., min_length=1, max_length=4000)
    max_iterations: Optional[int] = Field(
        None, ge=1, le=10,
        description="Cap the agent loop. Default 5; raise for complex multi-hop questions.",
    )


class AgentStep(BaseModel):
    name: str
    arguments: Dict[str, Any]
    result_summary: str


class AgentResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    steps: List[AgentStep] = Field(default_factory=list, description="Tools the agent called, in order.")
    iterations: int
    stopped_reason: Literal["final_answer", "max_iterations", "error"]


class ComponentCheck(BaseModel):
    """Status of a single dependency. ``details`` carries component-specific fields."""

    status: Literal["up", "down"]
    latency_ms: Optional[float] = Field(None, description="Wall-clock latency of the probe.")
    error: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Verbose health snapshot — includes per-component checks and metadata."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Aggregate verdict over all checks."
    )
    service: str = SERVICE_NAME
    version: str = SERVICE_VERSION
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of this probe.")
    uptime_seconds: float = Field(..., description="Seconds since the API process came up.")
    checks: Dict[str, ComponentCheck]


# =========================================================== lifespan
class _State:
    """Module-level singleton container for the loaded pipeline + uptime tracking."""

    pipeline: Optional[RAGPipeline] = None
    start_time: float = 0.0  # set at lifespan start


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    _State.start_time = time.time()
    logger.info("Loading RAG pipeline (downloads embedding model on first run)...")
    _State.pipeline = RAGPipeline()
    s = _State.pipeline.settings
    logger.info(
        "Pipeline ready. collection=%s | embedder=%s | llm=%s",
        s.qdrant_collection,
        s.embedding_model,
        s.ollama_model,
    )
    try:
        yield
    finally:
        logger.info("Shutting down.")
        _State.pipeline = None


# ================================================================ app
settings = get_settings()
app = FastAPI(
    title="NTT DATA Sustainability RAG",
    description="Retrieval-augmented Q&A over NTT DATA Business Solutions sustainability reports.",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_pipeline() -> RAGPipeline:
    if _State.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not initialised yet")
    return _State.pipeline


# =========================================================== endpoints
def _uptime() -> float:
    return round(time.time() - _State.start_time, 2)


def _classify(checks: Dict[str, Dict[str, Any]]) -> str:
    """Aggregate per-component statuses into a single verdict."""
    statuses = [c.get("status", "down") for c in checks.values()]
    if all(s == "up" for s in statuses):
        return "healthy"
    if any(s == "up" for s in statuses):
        return "degraded"
    return "unhealthy"


def _to_component_check(raw: Dict[str, Any]) -> ComponentCheck:
    """Pull canonical fields out of the raw pipeline.health() dict."""
    return ComponentCheck(
        status=raw.get("status", "down"),
        latency_ms=raw.get("latency_ms"),
        error=raw.get("error"),
        details={k: v for k, v in raw.items() if k not in {"status", "latency_ms", "error"}},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Verbose health snapshot (per-component checks + metadata)",
    responses={
        200: {"description": "All critical components are up (status=healthy/degraded)."},
        503: {"description": "One or more critical components are down."},
    },
)
async def health(response: Response) -> HealthResponse:
    """Verbose health probe with per-dependency status and latency.

    Returns ``HTTP 503`` when no component is up (``status=unhealthy``); otherwise
    ``HTTP 200`` even in ``degraded`` state so load balancers can keep routing
    to a partially functional instance.
    """
    pipeline = _require_pipeline()
    raw = pipeline.health()
    checks = {name: _to_component_check(c) for name, c in raw.items()}
    verdict = _classify(raw)

    if verdict == "unhealthy":
        response.status_code = 503

    return HealthResponse(
        status=verdict,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        uptime_seconds=_uptime(),
        checks=checks,
    )


@app.post("/ask", response_model=AskResponse, tags=["rag"])
async def ask(req: AskRequest) -> AskResponse:
    """Answer ``question`` from indexed reports and return cited sources."""
    pipeline = _require_pipeline()
    try:
        result = await pipeline.aask(
            req.question,
            top_k=req.top_k,
            year=req.year,
            source=req.source,
        )
    except ValueError as exc:
        # Validation-level error (empty question, too long, etc.).
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Upstream Ollama / Qdrant unreachable.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AskResponse(
        answer=result.answer,
        sources=[SourceItem(**s) for s in result.sources],
    )


@app.post("/agent", response_model=AgentResponse, tags=["rag"])
async def agent_endpoint(req: AgentRequest) -> AgentResponse:
    """Tool-using agent over the same RAG corpus.

    Use this instead of /ask when one round of retrieval isn't enough — e.g.
    comparison queries ("compare 2020 vs 2024 emissions"), exploratory
    questions ("what years do you cover?"), or multi-step reasoning. Returns
    the same shape as /ask plus a ``steps`` list so callers can audit what
    tools the agent invoked.
    """
    pipeline = _require_pipeline()
    agent = RAGAgent(pipeline, max_iterations=req.max_iterations)
    try:
        result = await agent.run(req.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AgentResponse(
        answer=result.answer,
        sources=[SourceItem(**s) for s in result.sources],
        steps=[AgentStep(name=s.name, arguments=s.arguments, result_summary=s.result_summary)
               for s in result.steps],
        iterations=result.iterations,
        stopped_reason=result.stopped_reason,
    )

# .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload
if "__name__" == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
