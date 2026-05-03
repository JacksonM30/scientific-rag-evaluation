"""Parsing helpers for model generation outputs."""

from __future__ import annotations

import json
from typing import Any


def parse_answer_json(raw_answer: str) -> dict[str, Any]:
    """Parse the strict JSON answer format used by RAG generation prompts."""

    payload = json.loads(_strip_markdown_fence(raw_answer))
    if not isinstance(payload, dict):
        raise ValueError("model answer JSON must be an object")

    answer = payload.get("answer")
    if not isinstance(answer, str):
        raise ValueError("model answer JSON must contain string key 'answer'")

    cited_passage_ids = payload.get("cited_passage_ids")
    if not isinstance(cited_passage_ids, list) or not all(
        isinstance(passage_id, str) for passage_id in cited_passage_ids
    ):
        raise ValueError(
            "model answer JSON must contain list[str] key 'cited_passage_ids'"
        )

    return {
        "answer": answer,
        "cited_passage_ids": cited_passage_ids,
    }


def _strip_markdown_fence(raw_answer: str) -> str:
    stripped = raw_answer.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped
