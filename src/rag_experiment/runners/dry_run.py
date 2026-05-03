"""Dry-run RAG experiment runner."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_experiment.data.hotpotqa import HotpotExample, load_hotpot_jsonl
from rag_experiment.generation.prompts import format_retrieved_context, get_prompt
from rag_experiment.retrieval.factory import build_retriever
from rag_experiment.runners.artifacts import (
    error_record,
    example_record,
    jsonable_config,
    load_json,
    resolve_config_paths,
    resolve_path,
    retrieval_record,
)


def run_config(config_path: str | Path) -> Path:
    config_file = resolve_path(config_path)
    config = load_json(config_file)
    resolved_config = resolve_config_paths(config)

    if not resolved_config["generation"].get("dry_run", False):
        raise ValueError("This runner currently supports dry_run=true only.")

    examples = load_hotpot_jsonl(
        resolved_config["dataset"]["path"],
        limit=resolved_config["dataset"].get("limit"),
    )
    output_path = resolved_config["output"]["path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = get_prompt(resolved_config["prompt"]["id"])
    started_at = datetime.now(timezone.utc).isoformat()

    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            record = _build_dry_run_record(
                example=example,
                config=resolved_config,
                prompt=prompt,
                started_at=started_at,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def _build_dry_run_record(
    *,
    example: HotpotExample,
    config: dict[str, Any],
    prompt: Any,
    started_at: str,
) -> dict[str, Any]:
    try:
        top_k = int(config["retriever"]["top_k"])
        retriever = build_retriever(config["retriever"], example.passages())
        results = retriever.retrieve(example.question, top_k=top_k)
        context = format_retrieved_context(results)
        rendered_messages = prompt.render(question=example.question, context=context)

        return {
            "run_name": config["run_name"],
            "created_at": started_at,
            "dry_run": True,
            "config": jsonable_config(config),
            "example": example_record(example),
            "retrieved_passages": [retrieval_record(result) for result in results],
            "prompt": prompt.as_dict(),
            "rendered_messages": rendered_messages,
            "raw_model_answer": None,
            "parsed_answer": None,
            "cited_passage_ids": [],
            "error": None,
        }
    except Exception as exc:  # Preserve enough context for later failure analysis.
        return {
            "run_name": config["run_name"],
            "created_at": started_at,
            "dry_run": True,
            "config": jsonable_config(config),
            "example": example_record(example),
            "retrieved_passages": [],
            "prompt": prompt.as_dict(),
            "rendered_messages": [],
            "raw_model_answer": None,
            "parsed_answer": None,
            "cited_passage_ids": [],
            "error": error_record(exc),
        }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run a dry-run RAG artifact pass.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    output_path = run_config(args.config)
    print(f"wrote={output_path}")


if __name__ == "__main__":
    _main()
