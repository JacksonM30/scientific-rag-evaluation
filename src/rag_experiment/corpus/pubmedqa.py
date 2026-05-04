"""PubMedQA corpus helpers for pooled retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_experiment.data.hotpotqa import Passage


def load_pubmedqa_rows(*, cache_dir: Path) -> list[dict[str, Any]]:
    """Load all labeled PubMedQA rows that contain retrievable context text."""
    from datasets import load_dataset

    dataset = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split="train",
        cache_dir=str(cache_dir),
    )

    rows: list[dict[str, Any]] = []
    for row in dataset:
        context = row.get("context") or {}
        if context.get("contexts"):
            rows.append(dict(row))
    if not rows:
        raise ValueError("No PubMedQA rows with context passages were loaded")
    return rows


def select_pubmedqa_queries(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Select deterministic query rows from loaded PubMedQA rows."""
    selected = rows[:limit]
    if not selected:
        raise ValueError("No PubMedQA query rows selected")
    return selected


def select_pubmedqa_corpus_rows(
    rows: list[dict[str, Any]],
    *,
    corpus_limit: int,
    query_limit: int,
) -> list[dict[str, Any]]:
    """Select corpus rows; corpus_limit=0 means all loaded rows."""
    if corpus_limit < 0:
        raise ValueError("corpus_limit must be >= 0")
    if query_limit <= 0:
        raise ValueError("query_limit must be > 0")
    if corpus_limit != 0 and corpus_limit < query_limit:
        raise ValueError(
            "PubMedQA corpus_limit must be 0 or >= limit so every evaluated "
            "query has its gold context in the retrieval corpus"
        )
    selected = rows if corpus_limit == 0 else rows[:corpus_limit]
    if not selected:
        raise ValueError("No PubMedQA corpus rows selected")
    return selected


def build_pubmedqa_passages(rows: list[dict[str, Any]]) -> list[Passage]:
    """Convert PubMedQA context sections into retrievable passages.

    Only context text enters the passage body. Question, long_answer,
    final_decision, and prediction fields are intentionally excluded to avoid
    answer leakage.
    """
    passages: list[Passage] = []
    for row in rows:
        pubid = str(row["pubid"])
        context = row.get("context") or {}
        context_texts = context.get("contexts") or []
        context_labels = context.get("labels") or []
        for index, text in enumerate(context_texts):
            label = context_labels[index] if index < len(context_labels) else "CONTEXT"
            passages.append(
                Passage(
                    id=f"pubmedqa::{pubid}::{index}",
                    example_id=pubid,
                    title=str(label),
                    sentence_index=index,
                    text=str(text),
                )
            )
    if not passages:
        raise ValueError("PubMedQA passage pool is empty")
    return passages


def pubmedqa_gold_evidence(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return context-level gold evidence keys for a PubMedQA row."""
    pubid = str(row["pubid"])
    context = row.get("context") or {}
    context_texts = context.get("contexts") or []
    return [
        {"pubid": pubid, "context_idx": index}
        for index in range(len(context_texts))
    ]
