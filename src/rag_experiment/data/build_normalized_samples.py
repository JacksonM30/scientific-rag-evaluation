"""Build small normalized v0.1 artifacts from real PubMedQA and SciFact rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_experiment.data.inspect_datasets import (
    DEFAULT_HF_CACHE,
    DEFAULT_SCIFACT_DIR,
    _ensure_scifact_data,
    _read_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/normalized_samples"
SCHEMA_VERSION = "v0.1"


def build_pubmedqa_sample(
    *,
    limit: int,
    output_path: Path,
    cache_dir: Path,
) -> Path:
    """Write a tiny PubMedQA normalized artifact with oracle context passages."""
    from datasets import load_dataset

    dataset = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split="train",
        cache_dir=str(cache_dir),
    )

    records: list[dict[str, Any]] = []
    for row in dataset:
        context = row.get("context") or {}
        context_texts = context.get("contexts") or []
        if not context_texts:
            continue
        records.append(_pubmedqa_record(row))
        if len(records) >= limit:
            break

    _write_jsonl(output_path, records)
    return output_path


def build_scifact_sample(
    *,
    limit: int,
    output_path: Path,
    data_dir: Path,
) -> Path:
    """Write a tiny SciFact normalized artifact with oracle rationale passages."""
    dataset_dir = _ensure_scifact_data(data_dir)
    corpus = {
        int(row["doc_id"]): row for row in _read_jsonl(dataset_dir / "corpus.jsonl")
    }
    claims = _read_jsonl(dataset_dir / "claims_train.jsonl")

    records: list[dict[str, Any]] = []
    for claim in claims:
        record = _scifact_record(claim, corpus)
        if record is None:
            continue
        records.append(record)
        if len(records) >= limit:
            break

    _write_jsonl(output_path, records)
    return output_path


def _pubmedqa_record(row: dict[str, Any]) -> dict[str, Any]:
    pubid = str(row["pubid"])
    context = row.get("context") or {}
    context_texts = context.get("contexts") or []
    context_labels = context.get("labels") or []

    gold_evidence = [
        {"pubid": pubid, "context_idx": index}
        for index in range(len(context_texts))
    ]
    retrieved_passages = [
        {
            "passage_id": f"pubmedqa::{pubid}::{index}",
            "rank": index + 1,
            "text": text,
            "metadata": {
                "dataset": "pubmedqa",
                "pubid": pubid,
                "context_idx": index,
                "context_label": context_labels[index]
                if index < len(context_labels)
                else None,
            },
        }
        for index, text in enumerate(context_texts)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "pubmedqa",
        "run": {
            "name": "pubmedqa_mini_oracle_context",
            "retriever": "oracle_context",
            "top_k": len(retrieved_passages),
            "model_profile": "gold_label_demo",
        },
        "example": {
            "id": pubid,
            "query": row["question"],
            "gold_answer": row["final_decision"],
            "gold_evidence": gold_evidence,
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
        "error": None,
    }


def _scifact_record(
    claim: dict[str, Any],
    corpus: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    evidence = claim.get("evidence") or {}
    if not evidence:
        return None

    doc_id_text, evidence_items = next(iter(evidence.items()))
    if not evidence_items:
        return None
    evidence_item = evidence_items[0]
    label = evidence_item.get("label")
    sentence_indices = evidence_item.get("sentences") or []
    doc_id = int(doc_id_text)
    corpus_row = corpus.get(doc_id)
    if not label or not sentence_indices or corpus_row is None:
        return None

    abstract = corpus_row.get("abstract") or []
    gold_evidence = [
        {"doc_id": str(doc_id), "sentence_index": int(sentence_index)}
        for sentence_index in sentence_indices
        if int(sentence_index) < len(abstract)
    ]
    if not gold_evidence:
        return None

    retrieved_passages = [
        {
            "passage_id": f"scifact::{doc_id}::{item['sentence_index']}",
            "rank": rank,
            "text": abstract[item["sentence_index"]],
            "metadata": {
                "dataset": "scifact",
                "doc_id": str(doc_id),
                "sentence_index": item["sentence_index"],
                "title": corpus_row.get("title"),
            },
        }
        for rank, item in enumerate(gold_evidence, start=1)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "scifact",
        "run": {
            "name": "scifact_mini_oracle_evidence",
            "retriever": "oracle_evidence",
            "top_k": len(retrieved_passages),
            "model_profile": "gold_label_demo",
        },
        "example": {
            "id": str(claim["id"]),
            "query": claim["claim"],
            "gold_answer": label,
            "gold_evidence": gold_evidence,
            "metadata": {
                "claim_id": claim["id"],
                "cited_doc_ids": [str(doc_id) for doc_id in claim.get("cited_doc_ids", [])],
                "source_doc_id": str(doc_id),
                "source_title": corpus_row.get("title"),
            },
        },
        "retrieved_passages": retrieved_passages,
        "prediction": {
            "answer": label,
            "cited_passage_ids": [],
            "metadata": {"source": "gold_label_demo"},
        },
        "error": None,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Build tiny normalized v0.1 artifacts from real datasets."
    )
    parser.add_argument("dataset", choices=["pubmedqa", "scifact"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--hf-cache-dir", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--scifact-dir", type=Path, default=DEFAULT_SCIFACT_DIR)
    args = parser.parse_args()

    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / f"{args.dataset}_mini_v01.jsonl"

    if args.dataset == "pubmedqa":
        output_path = build_pubmedqa_sample(
            limit=args.limit,
            output_path=args.output,
            cache_dir=args.hf_cache_dir,
        )
    else:
        output_path = build_scifact_sample(
            limit=args.limit,
            output_path=args.output,
            data_dir=args.scifact_dir,
        )

    print(f"wrote={output_path}")


if __name__ == "__main__":
    _main()
