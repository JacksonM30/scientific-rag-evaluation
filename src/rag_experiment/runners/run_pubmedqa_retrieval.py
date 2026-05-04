"""Run pooled-corpus PubMedQA retrieval and write normalized v0.1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_experiment.data.hotpotqa import Passage
from rag_experiment.data.inspect_datasets import DEFAULT_HF_CACHE
from rag_experiment.retrieval.base import RetrievalResult, Retriever
from rag_experiment.runners.artifacts import PROJECT_ROOT, error_record
from rag_experiment.runners.pooled_retrieval import (
    RETRIEVER_CHOICES,
    build_pooled_retriever,
    pooled_output_path,
)


SCHEMA_VERSION = "v0.1"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs/retrieval/pubmedqa_bm25_pooled_v01.jsonl"


def run_pubmedqa_retrieval(
    *,
    retriever_name: str,
    limit: int,
    top_k: int,
    output_path: Path,
    cache_dir: Path,
) -> Path:
    """Build a pooled PubMedQA BM25 artifact from the first N labeled rows."""
    rows = _load_pubmedqa_rows(limit=limit, cache_dir=cache_dir)
    passages = _build_passage_pool(rows)
    retriever = build_pooled_retriever(retriever_name, passages, top_k=top_k)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = _build_record(
                row=row,
                retriever=retriever,
                retriever_name=retriever_name,
                top_k=top_k,
                limit=limit,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def _load_pubmedqa_rows(*, limit: int, cache_dir: Path) -> list[dict[str, Any]]:
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
        if len(rows) >= limit:
            break
    if not rows:
        raise ValueError("No PubMedQA rows with context passages were loaded")
    return rows


def _build_passage_pool(rows: list[dict[str, Any]]) -> list[Passage]:
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


def _build_record(
    *,
    row: dict[str, Any],
    retriever: Retriever,
    retriever_name: str,
    top_k: int,
    limit: int,
) -> dict[str, Any]:
    try:
        results = retriever.retrieve(str(row["question"]), top_k=top_k)
        return _record(
            row=row,
            retrieved_passages=[_retrieved_passage(result) for result in results],
            retriever_name=retriever_name,
            top_k=top_k,
            limit=limit,
            error=None,
        )
    except Exception as exc:
        return _record(
            row=row,
            retrieved_passages=[],
            retriever_name=retriever_name,
            top_k=top_k,
            limit=limit,
            error=error_record(exc),
        )


def _record(
    *,
    row: dict[str, Any],
    retrieved_passages: list[dict[str, Any]],
    retriever_name: str,
    top_k: int,
    limit: int,
    error: dict[str, str] | None,
) -> dict[str, Any]:
    pubid = str(row["pubid"])
    context = row.get("context") or {}
    context_texts = context.get("contexts") or []

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "pubmedqa",
        "run": {
            "name": f"pubmedqa_{retriever_name}_pooled",
            "retriever": retriever_name,
            "top_k": top_k,
            "model_profile": "gold_label_demo",
            "sample_limit": limit,
        },
        "example": {
            "id": pubid,
            "query": row["question"],
            "gold_answer": row["final_decision"],
            "gold_evidence": [
                {"pubid": pubid, "context_idx": index}
                for index in range(len(context_texts))
            ],
            "metadata": {
                "pubid": pubid,
                "long_answer": row.get("long_answer"),
                "meshes": context.get("meshes") or [],
            },
        },
        "retrieved_passages": retrieved_passages,
        "prediction": {
            "answer": row["final_decision"],
            "cited_passage_ids": [],
            "metadata": {"source": "gold_label_demo"},
        },
        "error": error,
    }


def _retrieved_passage(result: RetrievalResult) -> dict[str, Any]:
    passage = result.passage
    return {
        "passage_id": passage.id,
        "rank": result.rank,
        "score": result.score,
        "text": passage.text,
        "metadata": {
            "dataset": "pubmedqa",
            "pubid": passage.example_id,
            "context_idx": passage.sentence_index,
            "context_label": passage.title,
        },
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PubMedQA retrieval over a pooled passage corpus."
    )
    parser.add_argument("--retriever", choices=RETRIEVER_CHOICES, default="bm25")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--hf-cache-dir", type=Path, default=DEFAULT_HF_CACHE)
    args = parser.parse_args()

    output_path = args.output or pooled_output_path("pubmedqa", args.retriever)
    output_path = run_pubmedqa_retrieval(
        retriever_name=args.retriever,
        limit=args.limit,
        top_k=args.top_k,
        output_path=output_path,
        cache_dir=args.hf_cache_dir,
    )
    print(f"wrote={output_path}")


if __name__ == "__main__":
    _main()
