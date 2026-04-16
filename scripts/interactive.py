"""Interactive REPL for chatting with the indexed sustainability reports.

Run:
    python scripts/interactive.py

Inside the prompt:
    :year 2024     restrict retrieval to a specific year (or :year clear)
    :topk 8        change retrieval depth
    :sources       toggle source printing
    :health        print pipeline health
    :quit          leave
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import RAGPipeline  # noqa: E402
from src.utils.log import setup_logging  # noqa: E402

PROMPT = "\nask> "
HELP = """
Commands:
  :year <YYYY>   filter retrieval by report year (':year clear' removes filter)
  :topk <N>      change retrieval depth
  :sources       toggle source printing
  :health        print Qdrant + Ollama health
  :help          show this help
  :quit / :exit  leave the REPL
Anything else is treated as a question.
"""


def _print_sources(sources: list[dict]) -> None:
    if not sources:
        print("(no sources)")
        return
    for s in sources:
        ref = s.get("ref", "?")
        print(
            f"[{ref}] {s['source']} — page {s['page_num']} "
            f"(year={s['year']}, score={s['score']:.3f})"
        )


def main() -> int:
    # REPL: keep the console clean — only show warnings/errors below the prompt.
    setup_logging(level="WARNING")

    print("Loading pipeline (first run downloads the embedding model — be patient)...")
    pipeline = RAGPipeline()
    print(
        f"Ready. collection={pipeline.settings.qdrant_collection} "
        f"| embedder={pipeline.settings.embedding_model} "
        f"| llm={pipeline.settings.ollama_model}"
    )
    print("Type :help for commands, :quit to exit.")

    year: str | None = None
    top_k: int | None = None
    show_sources = True

    while True:
        try:
            question = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question in (":quit", ":exit", ":q"):
            break
        if question == ":help":
            print(HELP)
            continue
        if question == ":health":
            for k, v in pipeline.health().items():
                print(f"  {k}: {v}")
            continue
        if question.startswith(":year"):
            _, _, val = question.partition(" ")
            val = val.strip()
            year = None if val in ("", "clear") else val
            print(f"[year filter = {year}]")
            continue
        if question.startswith(":topk"):
            _, _, val = question.partition(" ")
            try:
                top_k = int(val) if val.strip() else None
            except ValueError:
                print("usage: :topk <int>")
                continue
            print(f"[top_k = {top_k}]")
            continue
        if question == ":sources":
            show_sources = not show_sources
            print(f"[show_sources = {show_sources}]")
            continue

        try:
            result = pipeline.ask(question, top_k=top_k, year=year)
        except RuntimeError as exc:
            print(f"[error] {exc}")
            continue

        print("\n--- Answer ---")
        print(result.answer)
        if show_sources:
            print("\n--- Sources ---")
            _print_sources(result.sources)

    print("bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
