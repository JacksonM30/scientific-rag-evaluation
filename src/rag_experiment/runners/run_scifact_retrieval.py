"""Run pooled-corpus SciFact retrieval and write normalized v0.1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_experiment.data.hotpotqa import Passage
from rag_experiment.data.inspect_datasets import (
    DEFAULT_SCIFACT_DIR,
    _ensure_scifact_data,
    _read_jsonl,
)
from rag_experiment.retrieval.base import RetrievalResult, Retriever
from rag_experiment.runners.artifacts import error_record
from rag_experiment.runners.pooled_retrieval import (
    RETRIEVER_CHOICES,
    build_pooled_retriever,
    pooled_output_path,
)


SCHEMA_VERSION = "v0.1"


def run_scifact_retrieval(
    *,
    retriever_name: str,
    limit: int,
    top_k: int,
    output_path: Path,
    data_dir: Path,
    corpus_doc_limit: int,
) -> Path:
    """Build a pooled SciFact retrieval artifact from labeled train claims."""
    dataset_dir = _ensure_scifact_data(data_dir)
    corpus = {
        int(row["doc_id"]): row for row in _read_jsonl(dataset_dir / "corpus.jsonl")
    }
    claims = _select_labeled_claims(
        _read_jsonl(dataset_dir / "claims_train.jsonl"),
        corpus=corpus,
        limit=limit,
    )
    passages = _build_passage_pool(
        corpus=corpus,
        claims=claims,
        corpus_doc_limit=corpus_doc_limit,
    )
    retriever = build_pooled_retriever(retriever_name, passages, top_k=top_k)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for claim in claims:
            record = _build_record(
                claim=claim,
                corpus=corpus,
                retriever=retriever,
                retriever_name=retriever_name,
                top_k=top_k,
                limit=limit,
                corpus_doc_limit=corpus_doc_limit,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def _select_labeled_claims(
    claims: list[dict[str, Any]],
    *,
    corpus: dict[int, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for claim in claims:
        if _claim_label(claim) is None:
            continue
        if not _gold_evidence(claim, corpus):
            continue
        selected.append(claim)
        if len(selected) >= limit:
            break
    if not selected:
        raise ValueError("No labeled SciFact claims with usable evidence were loaded")
    return selected


def _build_passage_pool(
    *,
    corpus: dict[int, dict[str, Any]],
    claims: list[dict[str, Any]],
    corpus_doc_limit: int,
) -> list[Passage]:
    required_doc_ids = {
        int(item["doc_id"])
        for claim in claims
        for item in _gold_evidence(claim, corpus)
    }
    doc_ids = list(required_doc_ids)
    if corpus_doc_limit > 0:
        for doc_id in sorted(corpus):
            if len(doc_ids) >= corpus_doc_limit:
                break
            if doc_id not in required_doc_ids:
                doc_ids.append(doc_id)
    else:
        doc_ids = sorted(corpus)

    passages: list[Passage] = []
    for doc_id in doc_ids:
        row = corpus.get(doc_id)
        if row is None:
            continue
        title = str(row.get("title") or doc_id)
        for sentence_index, text in enumerate(row.get("abstract") or []):
            passages.append(
                Passage(
                    id=f"scifact::{doc_id}::{sentence_index}",
                    example_id=str(doc_id),
                    title=title,
                    sentence_index=sentence_index,
                    text=str(text),
                )
            )
    if not passages:
        raise ValueError("SciFact passage pool is empty")
    return passages


def _build_record(
    *,
    claim: dict[str, Any],
    corpus: dict[int, dict[str, Any]],
    retriever: Retriever,
    retriever_name: str,
    top_k: int,
    limit: int,
    corpus_doc_limit: int,
) -> dict[str, Any]:
    try:
        results = retriever.retrieve(str(claim["claim"]), top_k=top_k)
        return _record(
            claim=claim,
            corpus=corpus,
            retrieved_passages=[_retrieved_passage(result) for result in results],
            retriever_name=retriever_name,
            top_k=top_k,
            limit=limit,
            corpus_doc_limit=corpus_doc_limit,
            error=None,
        )
    except Exception as exc:
        return _record(
            claim=claim,
            corpus=corpus,
            retrieved_passages=[],
            retriever_name=retriever_name,
            top_k=top_k,
            limit=limit,
            corpus_doc_limit=corpus_doc_limit,
            error=error_record(exc),
        )


def _record(
    *,
    claim: dict[str, Any],
    corpus: dict[int, dict[str, Any]],
    retrieved_passages: list[dict[str, Any]],
    retriever_name: str,
    top_k: int,
    limit: int,
    corpus_doc_limit: int,
    error: dict[str, str] | None,
) -> dict[str, Any]:
    label = _claim_label(claim)
    evidence = _gold_evidence(claim, corpus)
    source_doc_ids = sorted({item["doc_id"] for item in evidence})

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "scifact",
        "run": {
            "name": f"scifact_{retriever_name}_pooled",
            "retriever": retriever_name,
            "top_k": top_k,
            "model_profile": "gold_label_demo",
            "sample_limit": limit,
            "corpus_doc_limit": corpus_doc_limit,
        },
        "example": {
            "id": str(claim["id"]),
            "query": claim["claim"],
            "gold_answer": label,
            "gold_evidence": evidence,
            "metadata": {
                "claim_id": claim["id"],
                "cited_doc_ids": [str(doc_id) for doc_id in claim.get("cited_doc_ids", [])],
                "source_doc_ids": source_doc_ids,
            },
        },
        "retrieved_passages": retrieved_passages,
        "prediction": {
            "answer": label,
            "cited_passage_ids": [],
            "metadata": {"source": "gold_label_demo"},
        },
        "error": error,
    }


def _claim_label(claim: dict[str, Any]) -> str | None:
    for evidence_items in (claim.get("evidence") or {}).values():
        for evidence_item in evidence_items:
            label = evidence_item.get("label")
            if label:
                return str(label)
    return None


def _gold_evidence(
    claim: dict[str, Any],
    corpus: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for doc_id_text, evidence_items in (claim.get("evidence") or {}).items():
        doc_id = int(doc_id_text)
        abstract = (corpus.get(doc_id) or {}).get("abstract") or []
        for evidence_item in evidence_items:
            for sentence_index in evidence_item.get("sentences") or []:
                sentence_index = int(sentence_index)
                key = (str(doc_id), sentence_index)
                if sentence_index >= len(abstract) or key in seen:
                    continue
                seen.add(key)
                evidence.append({"doc_id": str(doc_id), "sentence_index": sentence_index})
    return evidence


def _retrieved_passage(result: RetrievalResult) -> dict[str, Any]:
    passage = result.passage
    return {
        "passage_id": passage.id,
        "rank": result.rank,
        "score": result.score,
        "text": passage.text,
        "metadata": {
            "dataset": "scifact",
            "doc_id": passage.example_id,
            "sentence_index": passage.sentence_index,
            "title": passage.title,
        },
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SciFact retrieval over a pooled sentence corpus."
    )
    parser.add_argument("--retriever", choices=RETRIEVER_CHOICES, default="bm25")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--corpus-doc-limit",
        type=int,
        default=300,
        help="Maximum corpus docs in the pool; use 0 for the full SciFact corpus.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--scifact-dir", type=Path, default=DEFAULT_SCIFACT_DIR)
    args = parser.parse_args()

    output_path = args.output or pooled_output_path("scifact", args.retriever)
    output_path = run_scifact_retrieval(
        retriever_name=args.retriever,
        limit=args.limit,
        top_k=args.top_k,
        output_path=output_path,
        data_dir=args.scifact_dir,
        corpus_doc_limit=args.corpus_doc_limit,
    )
    print(f"wrote={output_path}")


if __name__ == "__main__":
    _main()
