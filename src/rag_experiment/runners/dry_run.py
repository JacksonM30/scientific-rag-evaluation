"""Dry-run RAG experiment runner."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_experiment.data.hotpotqa import HotpotExample, load_hotpot_jsonl
from rag_experiment.generation.prompts import format_retrieved_context, get_prompt
from rag_experiment.retrieval.bm25 import BM25Retriever, RetrievalResult


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_config(config_path: str | Path) -> Path:
    config_file = _resolve_path(config_path)
    config = _load_json(config_file)
    resolved_config = _resolve_config_paths(config)

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
        retriever = BM25Retriever(example.passages(), top_k=top_k)
        results = retriever.search(example.question, top_k=top_k)
        context = format_retrieved_context(results)
        rendered_messages = prompt.render(question=example.question, context=context)

        return {
            "run_name": config["run_name"],
            "created_at": started_at,
            "dry_run": True,
            "config": _jsonable_config(config),
            "example": _example_record(example),
            "retrieved_passages": [_retrieval_record(result) for result in results],
            "prompt": prompt.as_dict(),
            "rendered_messages": rendered_messages,
            "raw_model_answer": None,
            "parsed_answer": None,
            "error": None,
        }
    except Exception as exc:  # Preserve enough context for later failure analysis.
        return {
            "run_name": config["run_name"],
            "created_at": started_at,
            "dry_run": True,
            "config": _jsonable_config(config),
            "example": _example_record(example),
            "retrieved_passages": [],
            "prompt": prompt.as_dict(),
            "rendered_messages": [],
            "raw_model_answer": None,
            "parsed_answer": None,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


def _example_record(example: HotpotExample) -> dict[str, Any]:
    return {
        "id": example.id,
        "question": example.question,
        "gold_answer": example.answer,
        "type": example.type,
        "level": example.level,
        "supporting_facts": [
            {"title": title, "sentence_index": sentence_index}
            for title, sentence_index in example.supporting_facts
        ],
    }


def _retrieval_record(result: RetrievalResult) -> dict[str, Any]:
    passage = result.passage
    return {
        "rank": result.rank,
        "score": result.score,
        "passage_id": passage.id,
        "example_id": passage.example_id,
        "title": passage.title,
        "sentence_index": passage.sentence_index,
        "text": passage.text,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    resolved["dataset"]["path"] = _resolve_path(resolved["dataset"]["path"])
    resolved["output"]["path"] = _resolve_path(resolved["output"]["path"])
    return resolved


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _jsonable_config(config: Any) -> Any:
    if isinstance(config, dict):
        return {key: _jsonable_config(value) for key, value in config.items()}
    if isinstance(config, list):
        return [_jsonable_config(value) for value in config]
    if isinstance(config, Path):
        return str(config)
    return config


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run a dry-run RAG artifact pass.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    output_path = run_config(args.config)
    print(f"wrote={output_path}")


if __name__ == "__main__":
    _main()
