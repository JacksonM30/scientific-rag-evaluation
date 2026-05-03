"""Inspect saved RAG generation JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


ARTICLES = {"a", "an", "the"}


def inspect_file(
    artifact_path: str | Path,
    *,
    limit: int | None = None,
    show_prompt: bool = False,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    """Print a readable inspection view and optionally write aggregate counts."""
    path = Path(artifact_path)
    records = list(_iter_records(path, limit=limit))
    inspected = [_inspect_record(record) for record in records]
    summary = _build_summary(path, inspected)

    _print_summary(summary)
    for item in inspected:
        _print_case(item, show_prompt=show_prompt)

    if summary_json is not None:
        summary_path = Path(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return summary


def _iter_records(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(records) >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            record["_line_number"] = line_number
            records.append(record)
    return records


def _inspect_record(record: dict[str, Any]) -> dict[str, Any]:
    example = record.get("example") or {}
    retrieved = record.get("retrieved_passages") or []
    parsed = record.get("parsed_answer") or {}
    cited_ids = record.get("cited_passage_ids") or parsed.get("cited_passage_ids") or []
    model_answer = parsed.get("answer") or record.get("raw_model_answer") or ""
    gold_answer = example.get("gold_answer") or ""

    answer_match = _normalize_answer(model_answer) == _normalize_answer(gold_answer)
    matched_support_count, total_support_count = _support_match_counts(example, retrieved)
    support_retrieved = (
        total_support_count == 0 or matched_support_count == total_support_count
    )
    citation_valid = _citation_valid(cited_ids, retrieved)
    labels = _failure_labels(
        record=record,
        answer_match=answer_match,
        support_retrieved=support_retrieved,
        citation_valid=citation_valid,
        cited_ids=cited_ids,
    )

    return {
        "record": record,
        "example_id": example.get("id", ""),
        "question": example.get("question", ""),
        "gold_answer": gold_answer,
        "model_answer": model_answer,
        "cited_passage_ids": cited_ids,
        "answer_match": answer_match,
        "support_retrieved": support_retrieved,
        "citation_valid": citation_valid,
        "matched_support_count": matched_support_count,
        "total_support_count": total_support_count,
        "failure_labels": labels,
    }


def _normalize_answer(value: Any) -> str:
    text = str(value).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in re.split(r"\s+", text) if token and token not in ARTICLES]
    return " ".join(tokens)


def _support_match_counts(
    example: dict[str, Any], retrieved: list[dict[str, Any]]
) -> tuple[int, int]:
    supporting_facts = {
        (fact.get("title"), fact.get("sentence_index"))
        for fact in example.get("supporting_facts", [])
    }
    retrieved_facts = {
        (passage.get("title"), passage.get("sentence_index"))
        for passage in retrieved
    }
    return len(supporting_facts & retrieved_facts), len(supporting_facts)


def _citation_valid(cited_ids: list[str], retrieved: list[dict[str, Any]]) -> bool:
    retrieved_ids = {passage.get("passage_id") for passage in retrieved}
    return all(cited_id in retrieved_ids for cited_id in cited_ids)


def _failure_labels(
    *,
    record: dict[str, Any],
    answer_match: bool,
    support_retrieved: bool,
    citation_valid: bool,
    cited_ids: list[str],
) -> list[str]:
    labels: list[str] = []
    if record.get("error"):
        labels.append("runtime_error")
    if not answer_match:
        labels.append("answer_mismatch")
    if not support_retrieved:
        labels.append("support_not_retrieved")
    if not citation_valid:
        labels.append("citation_invalid")
    if not cited_ids:
        labels.append("no_citations")
    return labels


def _build_summary(path: Path, inspected: list[dict[str, Any]]) -> dict[str, Any]:
    first_record = inspected[0]["record"] if inspected else {}
    config = first_record.get("config") or {}
    retriever = config.get("retriever") or {}
    generation = config.get("generation") or {}
    prompt = config.get("prompt") or {}
    label_counts = Counter(
        label for item in inspected for label in item["failure_labels"]
    )

    return {
        "artifact_path": str(path),
        "run_name": first_record.get("run_name"),
        "retriever": retriever.get("name"),
        "model_profile": generation.get("model_profile"),
        "prompt_id": prompt.get("id"),
        "top_k": retriever.get("top_k"),
        "row_count": len(inspected),
        "error_count": sum(1 for item in inspected if item["record"].get("error")),
        "answer_match_count": sum(1 for item in inspected if item["answer_match"]),
        "support_retrieved_count": sum(
            1 for item in inspected if item["support_retrieved"]
        ),
        "citation_valid_count": sum(1 for item in inspected if item["citation_valid"]),
        "failure_label_counts": dict(sorted(label_counts.items())),
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("Generation Artifact Inspection")
    print("=" * 30)
    print(f"artifact: {summary['artifact_path']}")
    print(f"run_name: {summary['run_name']}")
    print(f"retriever: {summary['retriever']}  top_k: {summary['top_k']}")
    print(f"model_profile: {summary['model_profile']}")
    print(f"prompt_id: {summary['prompt_id']}")
    print(f"rows: {summary['row_count']}  errors: {summary['error_count']}")
    print(
        "checks: "
        f"answer_match={summary['answer_match_count']}/{summary['row_count']}  "
        f"support_retrieved={summary['support_retrieved_count']}/{summary['row_count']}  "
        f"citation_valid={summary['citation_valid_count']}/{summary['row_count']}"
    )
    if summary["failure_label_counts"]:
        print(f"failure_labels: {summary['failure_label_counts']}")
    print()


def _print_case(item: dict[str, Any], *, show_prompt: bool) -> None:
    record = item["record"]
    example = record.get("example") or {}
    retrieved = record.get("retrieved_passages") or []

    print(f"Example {item['example_id']} (line {record.get('_line_number')})")
    print("-" * 30)
    print(f"question: {item['question']}")
    print(f"gold_answer: {item['gold_answer']}")
    print(f"model_answer: {item['model_answer']}")
    print(f"cited_passage_ids: {item['cited_passage_ids']}")
    print(
        "checks: "
        f"answer_match={item['answer_match']}  "
        f"support_retrieved={item['support_retrieved']} "
        f"({item['matched_support_count']}/{item['total_support_count']})  "
        f"citation_valid={item['citation_valid']}"
    )
    print(f"failure_labels: {item['failure_labels'] or ['none']}")
    if record.get("error"):
        print(f"error: {record['error']}")

    print("supporting_facts:")
    for fact in example.get("supporting_facts", []):
        print(f"  - {fact.get('title')}::{fact.get('sentence_index')}")

    print("retrieved_passages:")
    for passage in retrieved:
        print(
            f"  [{passage.get('rank')}] {passage.get('passage_id')} "
            f"| {passage.get('title')}::{passage.get('sentence_index')}"
        )
        print(f"      {passage.get('text')}")

    if show_prompt:
        print("rendered_messages:")
        for message in record.get("rendered_messages") or []:
            print(f"  {message.get('type')}:")
            print(_indent(str(message.get("content", "")), prefix="    "))

    print()


def _indent(text: str, *, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a saved RAG generation JSONL artifact."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for aggregate counts only.",
    )
    args = parser.parse_args()

    inspect_file(
        args.artifact,
        limit=args.limit,
        show_prompt=args.show_prompt,
        summary_json=args.summary_json,
    )


if __name__ == "__main__":
    _main()
