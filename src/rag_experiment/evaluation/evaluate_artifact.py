"""Evaluate normalized v0.1 RAG artifacts."""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "v0.1"
PUBMEDQA_LABELS = ("yes", "no", "maybe")
SCIFACT_LABELS = {
    "support": "SUPPORT",
    "supports": "SUPPORT",
    "supported": "SUPPORT",
    "supporting": "SUPPORT",
    "contradict": "CONTRADICT",
    "contradicts": "CONTRADICT",
    "contradicted": "CONTRADICT",
    "contradiction": "CONTRADICT",
    "refute": "CONTRADICT",
    "refutes": "CONTRADICT",
    "refuted": "CONTRADICT",
    "not enough info": "NOT_ENOUGH_INFO",
    "not_enough_info": "NOT_ENOUGH_INFO",
    "nei": "NOT_ENOUGH_INFO",
    "unknown": "NOT_ENOUGH_INFO",
}


def evaluate_file(
    artifact_path: str | Path,
    *,
    dataset: str,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    records = list(_iter_records(path))
    if dataset == "pubmedqa":
        summary = _evaluate_pubmedqa(path, records)
    elif dataset == "scifact":
        summary = _evaluate_scifact(path, records)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    _print_summary(summary)
    if summary_json is not None:
        summary_path = Path(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def _iter_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            _validate_record(record, line_number=line_number)
            records.append(record)
    return records


def _validate_record(record: dict[str, Any], *, line_number: int) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"line {line_number}: expected schema_version={SCHEMA_VERSION!r}, "
            f"got {record.get('schema_version')!r}"
        )
    for key in ("dataset", "run", "example", "retrieved_passages", "prediction"):
        if key not in record:
            raise ValueError(f"line {line_number}: missing required key {key!r}")


def _evaluate_pubmedqa(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    answer_total = 0
    answer_correct = 0
    context_total = 0
    context_hit = 0
    context_recall_sum = 0.0
    skipped_context = 0
    citation = _empty_citation_stats()

    for record in records:
        _ensure_dataset(record, expected="pubmedqa")
        if record.get("error"):
            continue

        gold_answer = _normalize_pubmedqa_label(record["example"].get("gold_answer"))
        pred_answer = _normalize_pubmedqa_label(_prediction_answer(record))
        if gold_answer in PUBMEDQA_LABELS and pred_answer in PUBMEDQA_LABELS:
            answer_total += 1
            answer_correct += int(gold_answer == pred_answer)

        gold_contexts = _pubmedqa_context_keys(record["example"].get("gold_evidence", []))
        _update_citation_stats(
            citation,
            record=record,
            gold_keys=gold_contexts,
            passage_key_fn=_pubmedqa_passage_key,
        )
        if not gold_contexts:
            skipped_context += 1
            continue
        retrieved_contexts = _pubmedqa_context_keys(
            [passage.get("metadata") or {} for passage in record["retrieved_passages"]]
        )
        matched = gold_contexts & retrieved_contexts
        context_total += 1
        context_hit += int(bool(matched))
        context_recall_sum += len(matched) / len(gold_contexts)

    return _base_summary(path, "pubmedqa", records) | {
        "metrics": {
            "answer_accuracy": _rate(answer_correct, answer_total),
            "answer_correct": answer_correct,
            "answer_total": answer_total,
            "context_hit_at_k": _rate(context_hit, context_total),
            "context_hit_count": context_hit,
            "context_total": context_total,
            "context_recall_at_k": _average(context_recall_sum, context_total),
            "context_skipped": skipped_context,
            **_finalize_citation_stats(citation),
        },
    }


def _evaluate_scifact(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    label_total = 0
    label_correct = 0
    evidence_total = 0
    doc_hit = 0
    sentence_hit = 0
    all_hit = 0
    evidence_recall_sum = 0.0
    skipped_evidence = 0
    citation = _empty_citation_stats()

    for record in records:
        _ensure_dataset(record, expected="scifact")
        if record.get("error"):
            continue

        gold_label = _normalize_scifact_label(record["example"].get("gold_answer"))
        pred_label = _normalize_scifact_label(_prediction_answer(record))
        if gold_label is not None and pred_label is not None:
            label_total += 1
            label_correct += int(gold_label == pred_label)

        gold_evidence = _scifact_sentence_keys(record["example"].get("gold_evidence", []))
        _update_citation_stats(
            citation,
            record=record,
            gold_keys=gold_evidence,
            passage_key_fn=_scifact_passage_key,
        )
        if not gold_evidence:
            skipped_evidence += 1
            continue
        retrieved_evidence = _scifact_sentence_keys(
            [passage.get("metadata") or {} for passage in record["retrieved_passages"]]
        )
        matched_sentences = gold_evidence & retrieved_evidence
        gold_docs = {doc_id for doc_id, _ in gold_evidence}
        retrieved_docs = {doc_id for doc_id, _ in retrieved_evidence}

        evidence_total += 1
        doc_hit += int(bool(gold_docs & retrieved_docs))
        sentence_hit += int(bool(matched_sentences))
        all_hit += int(len(matched_sentences) == len(gold_evidence))
        evidence_recall_sum += len(matched_sentences) / len(gold_evidence)

    return _base_summary(path, "scifact", records) | {
        "metrics": {
            "label_accuracy": _rate(label_correct, label_total),
            "label_correct": label_correct,
            "label_total": label_total,
            "evidence_doc_hit_at_k": _rate(doc_hit, evidence_total),
            "evidence_doc_hit_count": doc_hit,
            "evidence_sentence_hit_at_k": _rate(sentence_hit, evidence_total),
            "evidence_sentence_hit_count": sentence_hit,
            "evidence_all_hit_at_k": _rate(all_hit, evidence_total),
            "evidence_all_hit_count": all_hit,
            "evidence_recall_at_k": _average(evidence_recall_sum, evidence_total),
            "evidence_total": evidence_total,
            "evidence_skipped": skipped_evidence,
            **_finalize_citation_stats(citation),
        },
    }


def _base_summary(path: Path, dataset: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    first = records[0] if records else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_path": str(path),
        "dataset": dataset,
        "run": first.get("run", {}),
        "row_count": len(records),
        "error_count": sum(1 for record in records if record.get("error")),
        "generation_error_count": sum(
            1 for record in records if record.get("generation_error")
        ),
    }


def _ensure_dataset(record: dict[str, Any], *, expected: str) -> None:
    if record.get("dataset") != expected:
        raise ValueError(f"Expected dataset={expected!r}, got {record.get('dataset')!r}")


def _prediction_answer(record: dict[str, Any]) -> Any:
    prediction = record.get("prediction") or {}
    return prediction.get("answer")


def _prediction_cited_ids(record: dict[str, Any]) -> list[str]:
    prediction = record.get("prediction") or {}
    cited_ids = prediction.get("cited_passage_ids")
    if cited_ids is None:
        parsed = record.get("parsed_answer") or {}
        cited_ids = parsed.get("cited_passage_ids")
    if not isinstance(cited_ids, list):
        return []
    return [str(cited_id) for cited_id in cited_ids]


def _normalize_pubmedqa_label(value: Any) -> str | None:
    text = _normalize_text(value)
    if text in PUBMEDQA_LABELS:
        return text
    for label in PUBMEDQA_LABELS:
        if re.search(rf"\b{label}\b", text):
            return label
    return None


def _normalize_scifact_label(value: Any) -> str | None:
    text = _normalize_text(value).replace("-", " ")
    return SCIFACT_LABELS.get(text)


def _normalize_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
    return " ".join(text.split())


def _pubmedqa_context_keys(items: list[dict[str, Any]]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for item in items:
        pubid = item.get("pubid")
        context_idx = item.get("context_idx")
        if pubid is None or context_idx is None:
            continue
        keys.add((str(pubid), int(context_idx)))
    return keys


def _pubmedqa_passage_key(passage: dict[str, Any]) -> tuple[str, int] | None:
    metadata = passage.get("metadata") or {}
    pubid = metadata.get("pubid")
    context_idx = metadata.get("context_idx")
    if pubid is None or context_idx is None:
        return None
    return (str(pubid), int(context_idx))


def _scifact_sentence_keys(items: list[dict[str, Any]]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for item in items:
        doc_id = item.get("doc_id")
        sentence_index = item.get("sentence_index")
        if doc_id is None or sentence_index is None:
            continue
        keys.add((str(doc_id), int(sentence_index)))
    return keys


def _scifact_passage_key(passage: dict[str, Any]) -> tuple[str, int] | None:
    metadata = passage.get("metadata") or {}
    doc_id = metadata.get("doc_id")
    sentence_index = metadata.get("sentence_index")
    if doc_id is None or sentence_index is None:
        return None
    return (str(doc_id), int(sentence_index))


def _empty_citation_stats() -> dict[str, int | float]:
    return {
        "citation_total": 0,
        "citation_no_citation_count": 0,
        "citation_valid_count": 0,
        "cited_passage_total": 0,
        "cited_passage_valid_count": 0,
        "citation_gold_total": 0,
        "citation_gold_hit_count": 0,
        "citation_gold_recall_sum": 0.0,
    }


def _update_citation_stats(
    stats: dict[str, int | float],
    *,
    record: dict[str, Any],
    gold_keys: set[tuple[str, int]],
    passage_key_fn: Any,
) -> None:
    retrieved_by_id = {
        passage.get("passage_id"): passage
        for passage in record.get("retrieved_passages", [])
        if passage.get("passage_id") is not None
    }
    cited_ids = _prediction_cited_ids(record)
    cited_passages = [
        retrieved_by_id[cited_id]
        for cited_id in cited_ids
        if cited_id in retrieved_by_id
    ]
    cited_keys = {
        key
        for key in (passage_key_fn(passage) for passage in cited_passages)
        if key is not None
    }

    stats["citation_total"] += 1
    stats["cited_passage_total"] += len(cited_ids)
    stats["cited_passage_valid_count"] += len(cited_passages)
    if not cited_ids:
        stats["citation_no_citation_count"] += 1
    if len(cited_passages) == len(cited_ids):
        stats["citation_valid_count"] += 1
    if gold_keys:
        matched_gold = gold_keys & cited_keys
        stats["citation_gold_total"] += 1
        stats["citation_gold_hit_count"] += int(bool(matched_gold))
        stats["citation_gold_recall_sum"] += len(matched_gold) / len(gold_keys)


def _finalize_citation_stats(stats: dict[str, int | float]) -> dict[str, Any]:
    citation_total = int(stats["citation_total"])
    cited_passage_total = int(stats["cited_passage_total"])
    citation_gold_total = int(stats["citation_gold_total"])
    return {
        "citation_valid_rate": _rate(int(stats["citation_valid_count"]), citation_total),
        "citation_valid_count": int(stats["citation_valid_count"]),
        "citation_total": citation_total,
        "citation_no_citation_rate": _rate(
            int(stats["citation_no_citation_count"]), citation_total
        ),
        "citation_no_citation_count": int(stats["citation_no_citation_count"]),
        "cited_passage_valid_rate": _rate(
            int(stats["cited_passage_valid_count"]), cited_passage_total
        ),
        "cited_passage_valid_count": int(stats["cited_passage_valid_count"]),
        "cited_passage_total": cited_passage_total,
        "citation_gold_hit_rate": _rate(
            int(stats["citation_gold_hit_count"]), citation_gold_total
        ),
        "citation_gold_hit_count": int(stats["citation_gold_hit_count"]),
        "citation_gold_total": citation_gold_total,
        "citation_gold_recall": _average(
            float(stats["citation_gold_recall_sum"]), citation_gold_total
        ),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _average(total: float, count: int) -> float | None:
    if count == 0:
        return None
    return round(total / count, 6)


def _print_summary(summary: dict[str, Any]) -> None:
    print("Normalized Artifact Evaluation")
    print("==============================")
    print(f"schema_version: {summary['schema_version']}")
    print(f"dataset: {summary['dataset']}")
    print(f"artifact: {summary['artifact_path']}")
    print(f"run: {summary['run']}")
    print(
        f"rows: {summary['row_count']}  errors: {summary['error_count']}  "
        f"generation_errors: {summary['generation_error_count']}"
    )
    print("metrics:")
    for key, value in summary["metrics"].items():
        print(f"  {key}: {value}")


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a normalized v0.1 PubMedQA or SciFact artifact."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--dataset", choices=["pubmedqa", "scifact"], required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    evaluate_file(
        args.artifact,
        dataset=args.dataset,
        summary_json=args.summary_json,
    )


if __name__ == "__main__":
    _main()
