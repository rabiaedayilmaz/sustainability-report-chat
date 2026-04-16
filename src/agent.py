"""Tool-using agent built on the RAG pipeline + Ollama's native tool calling.

The agent is a small loop: the LLM picks which tool to call from a fixed
toolbox, we execute it, feed the result back, and repeat until it returns a
final answer (or hits the iteration cap).

Use :class:`RAGAgent` when single-shot retrieval isn't enough — comparison
queries ("how did X change from 2020 to 2024"), exploratory questions ("what
years do you cover?"), or anything benefitting from multi-step reasoning.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Tuple

import httpx

from .pipeline import RAGPipeline
from .utils.log import logger
from .utils.metrics import LLM_LATENCY, Timer

# =============================================================== prompt
AGENT_SYSTEM_PROMPT = (
    "You are a sustainability analyst answering questions about NTT DATA "
    "sustainability reports. You have a small toolbox to retrieve evidence:\n\n"
    "  - search_reports: dense retrieval over chunked reports, optionally "
    "filtered by year.\n"
    "  - list_available_years: shows which report years are indexed (use "
    "this when the user asks about coverage or scope).\n"
    "  - compare_years: runs N parallel retrievals for the same question "
    "across multiple years (use for comparison queries instead of multiple "
    "search_reports calls).\n\n"
    "Workflow:\n"
    "1. Decide if you need evidence (most factual questions: yes).\n"
    "2. Call the right tool(s). Don't call the same tool with identical "
    "arguments twice.\n"
    "3. Synthesise an answer in plain prose. Cite each factual claim "
    "inline as [source.pdf, p.X] using the metadata returned by the tools.\n"
    "4. If retrieved evidence does not answer the question, say so plainly "
    "rather than inventing facts.\n\n"
    "Be concise. Prefer numbers and years from the reports over generic "
    "statements."
)


# ============================================================ tool schema
def tool_definitions() -> list[dict]:
    """OpenAI/Ollama function-calling schema for the toolbox."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_reports",
                "description": (
                    "Retrieve relevant text chunks from indexed NTT DATA "
                    "sustainability reports. Returns chunks with source PDF, "
                    "year, page number, and similarity score. Use this for "
                    "any factual question about reports."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The semantic search query.",
                        },
                        "year": {
                            "type": "string",
                            "description": "Filter to one report year (e.g. '2024'). Omit if not applicable.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "How many chunks to fetch (default 6).",
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_available_years",
                "description": (
                    "Return the list of report years that have been indexed. "
                    "Use this when the user asks 'what years do you have?', "
                    "'what's the most recent report?', etc."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_years",
                "description": (
                    "Run the same retrieval question against multiple years "
                    "in one call. Use for comparison queries like 'how did X "
                    "change from 2020 to 2024'. Returns top-3 chunks per year."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "years": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Report years to compare, e.g. ['2020','2024'].",
                        },
                    },
                    "required": ["question", "years"],
                },
            },
        },
    ]


# =============================================================== records
@dataclass
class ToolCall:
    """Single tool invocation, kept for transparency in the API response."""

    name: str
    arguments: dict
    result_summary: str  # short — full result is fed back to the LLM


@dataclass
class AgentAnswer:
    answer: str
    sources: List[dict] = field(default_factory=list)
    steps: List[ToolCall] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = "final_answer"  # | "max_iterations" | "error"

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "steps": [asdict(s) for s in self.steps],
            "iterations": self.iterations,
            "stopped_reason": self.stopped_reason,
        }


# ================================================================ agent
class RAGAgent:
    """Tool-using agent on top of :class:`RAGPipeline`.

    Constructed cheaply (no model load) — share the pipeline across many agent
    instances if needed.
    """

    DEFAULT_MAX_ITERATIONS = 5

    def __init__(self, pipeline: RAGPipeline, max_iterations: int | None = None) -> None:
        self.pipeline = pipeline
        self.max_iterations = max_iterations or self.DEFAULT_MAX_ITERATIONS

    # ----------------------------------------------------------- ollama I/O
    def _chat_url(self) -> str:
        return f"{self.pipeline.settings.ollama_base_url.rstrip('/')}/api/chat"

    async def _chat(self, messages: list[dict]) -> dict:
        payload = {
            "model": self.pipeline.settings.ollama_model,
            "messages": messages,
            "tools": tool_definitions(),
            "stream": False,
            "options": {
                "temperature": self.pipeline.settings.ollama_temperature,
                "num_ctx": self.pipeline.settings.ollama_num_ctx,
            },
        }
        try:
            with Timer(LLM_LATENCY):
                async with httpx.AsyncClient(timeout=self.pipeline.settings.ollama_timeout) as client:
                    r = await client.post(self._chat_url(), json=payload)
                    r.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama /api/chat failed ({exc}). Is the daemon running and "
                f"does '{self.pipeline.settings.ollama_model}' support tool calls?"
            ) from exc
        return r.json()

    # ----------------------------------------------------------- dispatch
    def _exec_tool(self, name: str, args: dict) -> Tuple[Any, list[dict]]:
        """Execute a tool. Returns ``(payload_for_llm, sources_to_collect)``."""
        if name == "search_reports":
            hits = self.pipeline.retrieve(
                question=args["question"],
                year=args.get("year"),
                top_k=args.get("top_k"),
            )
            sources = self.pipeline._hits_to_sources(hits)
            payload = [
                {
                    "source": s["source"],
                    "year": s["year"],
                    "page": s["page_num"],
                    "score": s["score"],
                    "text": s["snippet"],
                }
                for s in sources
            ]
            return payload, sources

        if name == "list_available_years":
            years = sorted(self.pipeline.year_extractor.available_years)
            return {"years": years, "count": len(years)}, []

        if name == "compare_years":
            buckets: dict[str, list] = {}
            collected: list[dict] = []
            for year in args.get("years", []):
                hits = self.pipeline.retrieve(
                    question=args["question"], year=year, top_k=3,
                )
                src = self.pipeline._hits_to_sources(hits)
                collected.extend(src)
                buckets[year] = [
                    {
                        "source": s["source"],
                        "page": s["page_num"],
                        "score": s["score"],
                        "text": s["snippet"],
                    }
                    for s in src
                ]
            return buckets, collected

        return {"error": f"unknown tool '{name}'"}, []

    @staticmethod
    def _summarise(result: Any) -> str:
        if isinstance(result, list):
            return f"{len(result)} chunks"
        if isinstance(result, dict) and "years" in result:
            return f"{result.get('count', '?')} years"
        if isinstance(result, dict):
            return f"dict with {len(result)} keys"
        return type(result).__name__

    @staticmethod
    def _coerce_args(raw: Any) -> dict:
        """Some Ollama versions return arguments as a JSON string."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _dedup_sources(sources: list[dict]) -> list[dict]:
        """Dedupe by chunk_id while preserving order, then renumber refs."""
        seen: set[str] = set()
        out: list[dict] = []
        for s in sources:
            key = s.get("chunk_id") or f"{s.get('source')}::p{s.get('page_num')}"
            if key in seen:
                continue
            seen.add(key)
            out.append({**s, "ref": len(out) + 1})
        return out

    # ------------------------------------------------------------ main loop
    async def run(self, question: str) -> AgentAnswer:
        """Drive the conversation until the model produces a final answer."""
        question = self.pipeline._validate_question(question)
        messages: list[dict] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        steps: list[ToolCall] = []
        all_sources: list[dict] = []
        seen_calls: set[str] = set()

        for i in range(1, self.max_iterations + 1):
            response = await self._chat(messages)
            msg = response.get("message", {})
            messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # No tool requested → treat as final answer.
                return AgentAnswer(
                    answer=(msg.get("content") or "").strip(),
                    sources=self._dedup_sources(all_sources),
                    steps=steps,
                    iterations=i,
                    stopped_reason="final_answer",
                )

            # Execute each requested tool sequentially.
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                args = self._coerce_args(fn.get("arguments"))

                # Cheap loop guard: identical name+args means stuck agent.
                key = f"{name}:{json.dumps(args, sort_keys=True)}"
                if key in seen_calls:
                    logger.warning("Agent attempted duplicate tool call: %s", key)
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"error": "duplicate_call_skipped"}),
                    })
                    continue
                seen_calls.add(key)

                try:
                    raw, new_sources = self._exec_tool(name, args)
                    all_sources.extend(new_sources)
                except Exception as exc:
                    raw = {"error": str(exc)}
                    logger.exception("Tool '%s' raised: %s", name, exc)

                steps.append(ToolCall(
                    name=name,
                    arguments=args,
                    result_summary=self._summarise(raw),
                ))
                messages.append({
                    "role": "tool",
                    "content": json.dumps(raw, ensure_ascii=False),
                })

        # Hit the iteration cap.
        return AgentAnswer(
            answer="(agent did not converge — increase max_iterations or simplify the question)",
            sources=self._dedup_sources(all_sources),
            steps=steps,
            iterations=self.max_iterations,
            stopped_reason="max_iterations",
        )
