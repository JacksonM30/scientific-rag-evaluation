"""Shared helpers for runnable experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_experiment.data.hotpotqa import HotpotExample
from rag_experiment.retrieval.base import RetrievalResult


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def example_record(example: HotpotExample) -> dict[str, Any]:
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


def retrieval_record(result: RetrievalResult) -> dict[str, Any]:
    passage = result.passage
    return {
        "rank": result.rank,
        "score": result.score,
        "passage_id": passage.id,
        "example_id": passage.example_id,
        "title": passage.title,
        "sentence_index": passage.sentence_index,
        "text": passage.text,
        "metadata": result.metadata,
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    resolved["dataset"]["path"] = resolve_path(resolved["dataset"]["path"])
    resolved["output"]["path"] = resolve_path(resolved["output"]["path"])
    return resolved


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def jsonable_config(config: Any) -> Any:
    if isinstance(config, dict):
        return {key: jsonable_config(value) for key, value in config.items()}
    if isinstance(config, list):
        return [jsonable_config(value) for value in config]
    if isinstance(config, Path):
        return str(config)
    return config


def error_record(exc: Exception) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }
