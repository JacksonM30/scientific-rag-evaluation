"""Build and compare small generation prompt A/B subsets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_PUBMEDQA_V1 = Path("outputs/generation/pubmedqa_dense_v4_generated_n100_v01.jsonl")
DEFAULT_SCIFACT_V1 = Path("outputs/generation/scifact_dense_v4_generated_n100_v01.jsonl")


def select_ab_examples(
    *,
    pubmedqa_v1: Path = DEFAULT_PUBMEDQA_V1,
    scifact_v1: Path = DEFAULT_SCIFACT_V1,
    output_dir: Path = Path("outputs/analysis/generation_ab"),
    per_category_limit: int = 10,
) -> dict[str, Any]:
    """Select deterministic example IDs for prompt A/B checks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pubmedqa_records = _load_jsonl(pubmedqa_v1)
    scifact_records = _load_jsonl(scifact_v1)
    selections = {
        "pubmedqa": _select_dataset_examples(
            "pubmedqa",
            pubmedqa_records,
            category_order=(
                "citation_invalid",
                "wrong_answer_but_gold_retrieved",
            ),
            per_category_limit=per_category_limit,
        ),
        "scifact": _select_dataset_examples(
            "scifact",
            scifact_records,
            category_order=(
                "generation_error",
                "citation_gold_miss",
                "wrong_answer_but_gold_retrieved",
            ),
            per_category_limit=per_category_limit,
        ),
    }

    summary: dict[str, Any] = {
        "selection_policy": {
            "per_category_limit": per_category_limit,
            "pubmedqa_categories": [
                "citation_invalid",
                "wrong_answer_but_gold_retrieved",
            ],
            "scifact_categories": [
                "generation_error",
                "citation_gold_miss",
                "wrong_answer_but_gold_retrieved",
            ],
        },
        "datasets": {},
    }
    for dataset, selected in selections.items():
        ids_path = output_dir / f"{dataset}_ab_selected_ids.txt"
        details_path = output_dir / f"{dataset}_ab_selected_cases.csv"
        ids_path.write_text("\n".join(selected["ids"]) + "\n", encoding="utf-8")
        _write_case_csv(details_path, selected["cases"])
        summary["datasets"][dataset] = {
            "ids_path": str(ids_path),
            "details_path": str(details_path),
            "selected_count": len(selected["ids"]),
            "category_counts": dict(Counter(selected["case_categories"])),
            "selected_ids": selected["ids"],
        }

    summary_path = output_dir / "ab_selection_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def compare_ab_outputs(
    *,
    dataset: str,
    v1_path: Path,
    v2_path: Path,
    v3_path: Path | None = None,
    output_csv: Path,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Compare v1, v2, and optionally v3 generated artifacts on the same IDs."""
    v1_by_id = _records_by_id(_load_jsonl(v1_path))
    v2_by_id = _records_by_id(_load_jsonl(v2_path))
    v3_by_id = _records_by_id(_load_jsonl(v3_path)) if v3_path is not None else None
    common_ids = [example_id for example_id in v2_by_id if example_id in v1_by_id]
    if v3_by_id is not None:
        common_ids = [example_id for example_id in common_ids if example_id in v3_by_id]

    rows: list[dict[str, Any]] = []
    for example_id in common_ids:
        v1 = v1_by_id[example_id]
        v2 = v2_by_id[example_id]
        v1_tag = tag_record(dataset, v1)
        v2_tag = tag_record(dataset, v2)
        row = {
            "id": example_id,
            "gold_answer": v1_tag["gold_answer"],
            "question": v1_tag["question"],
            "v1_answer": v1_tag["prediction"],
            "v2_answer": v2_tag["prediction"],
            "v1_answer_match": v1_tag["answer_match"],
            "v2_answer_match": v2_tag["answer_match"],
            "v1_generation_error": v1_tag["generation_error"],
            "v2_generation_error": v2_tag["generation_error"],
            "v1_citation_valid": v1_tag["citation_valid"],
            "v2_citation_valid": v2_tag["citation_valid"],
            "v1_cited_gold": v1_tag["cited_gold"],
            "v2_cited_gold": v2_tag["cited_gold"],
            "v1_cited_count": v1_tag["cited_count"],
            "v2_cited_count": v2_tag["cited_count"],
            "v1_evidence_summary": v1_tag["evidence_summary"],
            "v2_evidence_summary": v2_tag["evidence_summary"],
        }
        if v3_by_id is not None:
            v3_tag = tag_record(dataset, v3_by_id[example_id])
            row |= {
                "v3_answer": v3_tag["prediction"],
                "v3_answer_match": v3_tag["answer_match"],
                "v3_generation_error": v3_tag["generation_error"],
                "v3_citation_valid": v3_tag["citation_valid"],
                "v3_cited_gold": v3_tag["cited_gold"],
                "v3_cited_count": v3_tag["cited_count"],
                "v3_evidence_summary": v3_tag["evidence_summary"],
            }
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_case_csv(output_csv, rows)
    summary = _compare_summary(dataset, rows)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def compare_model_outputs(
    *,
    dataset: str,
    baseline_path: Path,
    candidate_path: Path,
    baseline_label: str,
    candidate_label: str,
    output_csv: Path,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Compare two generated artifacts on the same example IDs."""
    baseline_by_id = _records_by_id(_load_jsonl(baseline_path))
    candidate_by_id = _records_by_id(_load_jsonl(candidate_path))
    common_ids = [
        example_id for example_id in candidate_by_id if example_id in baseline_by_id
    ]

    rows: list[dict[str, Any]] = []
    for example_id in common_ids:
        baseline = tag_record(dataset, baseline_by_id[example_id])
        candidate = tag_record(dataset, candidate_by_id[example_id])
        rows.append(
            {
                "id": example_id,
                "gold_answer": baseline["gold_answer"],
                "question": baseline["question"],
                "baseline_label": baseline_label,
                "candidate_label": candidate_label,
                "baseline_answer": baseline["prediction"],
                "candidate_answer": candidate["prediction"],
                "baseline_answer_match": baseline["answer_match"],
                "candidate_answer_match": candidate["answer_match"],
                "baseline_generation_error": baseline["generation_error"],
                "candidate_generation_error": candidate["generation_error"],
                "baseline_citation_valid": baseline["citation_valid"],
                "candidate_citation_valid": candidate["citation_valid"],
                "baseline_cited_gold": baseline["cited_gold"],
                "candidate_cited_gold": candidate["cited_gold"],
                "baseline_cited_count": baseline["cited_count"],
                "candidate_cited_count": candidate["cited_count"],
                "baseline_evidence_summary": baseline["evidence_summary"],
                "candidate_evidence_summary": candidate["evidence_summary"],
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_case_csv(output_csv, rows)
    summary = _compare_model_summary(
        dataset=dataset,
        rows=rows,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
    )
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def tag_record(dataset: str, record: dict[str, Any]) -> dict[str, Any]:
    retrieved = record.get("retrieved_passages") or []
    retrieved_by_id = {passage.get("passage_id"): passage for passage in retrieved}
    cited_ids = _prediction_cited_ids(record)
    invalid_ids = [cited_id for cited_id in cited_ids if cited_id not in retrieved_by_id]
    cited_keys = {
        _passage_key(dataset, retrieved_by_id[cited_id])
        for cited_id in cited_ids
        if cited_id in retrieved_by_id
    }
    retrieved_keys = {_passage_key(dataset, passage) for passage in retrieved}
    gold = _gold_keys(dataset, record)
    gold_answer = (record.get("example") or {}).get("gold_answer")
    prediction = _prediction_answer(record)
    answer_match = _normalize_answer(dataset, prediction) == _normalize_answer(
        dataset, gold_answer
    )
    return {
        "id": str((record.get("example") or {}).get("id", "")),
        "question": (record.get("example") or {}).get("query"),
        "gold_answer": gold_answer,
        "prediction": prediction,
        "answer_match": answer_match,
        "generation_error": bool(record.get("generation_error")),
        "retrieved_gold": bool(gold & retrieved_keys),
        "cited_gold": bool(gold & cited_keys),
        "citation_valid": not invalid_ids,
        "no_citation": len(cited_ids) == 0,
        "cited_count": len(cited_ids),
        "invalid_cited_ids": ";".join(invalid_ids),
        "evidence_summary": _evidence_summary(record),
    }


def _select_dataset_examples(
    dataset: str,
    records: list[dict[str, Any]],
    *,
    category_order: tuple[str, ...],
    per_category_limit: int,
) -> dict[str, Any]:
    cases = [tag_record(dataset, record) for record in records]
    selected_ids: list[str] = []
    selected_cases: list[dict[str, Any]] = []
    selected_categories: list[str] = []

    for category in category_order:
        count = 0
        for case in cases:
            if count >= per_category_limit:
                break
            if case["id"] in selected_ids or not _case_matches(category, case):
                continue
            selected_ids.append(case["id"])
            selected_cases.append(case | {"selected_category": category})
            selected_categories.append(category)
            count += 1

    return {
        "ids": selected_ids,
        "cases": selected_cases,
        "case_categories": selected_categories,
    }


def _case_matches(category: str, case: dict[str, Any]) -> bool:
    if category == "generation_error":
        return bool(case["generation_error"])
    if category == "citation_invalid":
        return not bool(case["citation_valid"])
    if category == "citation_gold_miss":
        return bool(case["retrieved_gold"]) and not bool(case["cited_gold"])
    if category == "wrong_answer_but_gold_retrieved":
        return not bool(case["answer_match"]) and bool(case["retrieved_gold"])
    raise ValueError(f"Unknown A/B selection category: {category}")


def _compare_summary(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    summary = {
        "dataset": dataset,
        "row_count": total,
        "v1_answer_accuracy": _rate(
            sum(1 for row in rows if row["v1_answer_match"]), total
        ),
        "v2_answer_accuracy": _rate(
            sum(1 for row in rows if row["v2_answer_match"]), total
        ),
        "v1_generation_errors": sum(1 for row in rows if row["v1_generation_error"]),
        "v2_generation_errors": sum(1 for row in rows if row["v2_generation_error"]),
        "v1_citation_valid_rate": _rate(
            sum(1 for row in rows if row["v1_citation_valid"]), total
        ),
        "v2_citation_valid_rate": _rate(
            sum(1 for row in rows if row["v2_citation_valid"]), total
        ),
        "v1_citation_gold_hit_rate": _rate(
            sum(1 for row in rows if row["v1_cited_gold"]), total
        ),
        "v2_citation_gold_hit_rate": _rate(
            sum(1 for row in rows if row["v2_cited_gold"]), total
        ),
        "answer_regressions": sum(
            1
            for row in rows
            if row["v1_answer_match"] and not row["v2_answer_match"]
        ),
        "answer_fixes": sum(
            1
            for row in rows
            if not row["v1_answer_match"] and row["v2_answer_match"]
        ),
        "citation_gold_regressions": sum(
            1 for row in rows if row["v1_cited_gold"] and not row["v2_cited_gold"]
        ),
        "citation_gold_fixes": sum(
            1 for row in rows if not row["v1_cited_gold"] and row["v2_cited_gold"]
        ),
    }
    if rows and "v3_answer_match" in rows[0]:
        summary |= {
            "v3_answer_accuracy": _rate(
                sum(1 for row in rows if row["v3_answer_match"]), total
            ),
            "v3_generation_errors": sum(
                1 for row in rows if row["v3_generation_error"]
            ),
            "v3_citation_valid_rate": _rate(
                sum(1 for row in rows if row["v3_citation_valid"]), total
            ),
            "v3_citation_gold_hit_rate": _rate(
                sum(1 for row in rows if row["v3_cited_gold"]), total
            ),
            "v3_answer_regressions": sum(
                1
                for row in rows
                if row["v1_answer_match"] and not row["v3_answer_match"]
            ),
            "v3_answer_fixes": sum(
                1
                for row in rows
                if not row["v1_answer_match"] and row["v3_answer_match"]
            ),
            "v3_citation_gold_regressions": sum(
                1
                for row in rows
                if row["v1_cited_gold"] and not row["v3_cited_gold"]
            ),
            "v3_citation_gold_fixes": sum(
                1
                for row in rows
                if not row["v1_cited_gold"] and row["v3_cited_gold"]
            ),
        }
    return summary


def _compare_model_summary(
    *,
    dataset: str,
    rows: list[dict[str, Any]],
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    total = len(rows)
    return {
        "dataset": dataset,
        "row_count": total,
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_answer_accuracy": _rate(
            sum(1 for row in rows if row["baseline_answer_match"]), total
        ),
        "candidate_answer_accuracy": _rate(
            sum(1 for row in rows if row["candidate_answer_match"]), total
        ),
        "baseline_generation_errors": sum(
            1 for row in rows if row["baseline_generation_error"]
        ),
        "candidate_generation_errors": sum(
            1 for row in rows if row["candidate_generation_error"]
        ),
        "baseline_citation_valid_rate": _rate(
            sum(1 for row in rows if row["baseline_citation_valid"]), total
        ),
        "candidate_citation_valid_rate": _rate(
            sum(1 for row in rows if row["candidate_citation_valid"]), total
        ),
        "baseline_citation_gold_hit_rate": _rate(
            sum(1 for row in rows if row["baseline_cited_gold"]), total
        ),
        "candidate_citation_gold_hit_rate": _rate(
            sum(1 for row in rows if row["candidate_cited_gold"]), total
        ),
        "answer_regressions": sum(
            1
            for row in rows
            if row["baseline_answer_match"] and not row["candidate_answer_match"]
        ),
        "answer_fixes": sum(
            1
            for row in rows
            if not row["baseline_answer_match"] and row["candidate_answer_match"]
        ),
        "citation_gold_regressions": sum(
            1
            for row in rows
            if row["baseline_cited_gold"] and not row["candidate_cited_gold"]
        ),
        "citation_gold_fixes": sum(
            1
            for row in rows
            if not row["baseline_cited_gold"] and row["candidate_cited_gold"]
        ),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str((record.get("example") or {}).get("id", "")): record for record in records}


def _write_case_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_answer(record: dict[str, Any]) -> Any:
    return (record.get("prediction") or {}).get("answer")


def _prediction_cited_ids(record: dict[str, Any]) -> list[str]:
    cited_ids = (record.get("prediction") or {}).get("cited_passage_ids")
    if cited_ids is None:
        cited_ids = (record.get("parsed_answer") or {}).get("cited_passage_ids")
    if not isinstance(cited_ids, list):
        return []
    return [str(cited_id) for cited_id in cited_ids]


def _evidence_summary(record: dict[str, Any]) -> str:
    metadata = (record.get("prediction") or {}).get("metadata") or {}
    evidence_summary = metadata.get("evidence_summary")
    if isinstance(evidence_summary, str):
        return evidence_summary
    parsed = record.get("parsed_answer") or {}
    evidence_summary = parsed.get("evidence_summary")
    if isinstance(evidence_summary, str):
        return evidence_summary
    return ""


def _normalize_answer(dataset: str, value: Any) -> str:
    text = str(value or "").strip()
    if dataset == "pubmedqa":
        return text.lower()
    return text.upper().replace("-", "_").replace(" ", "_")


def _gold_keys(dataset: str, record: dict[str, Any]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for item in (record.get("example") or {}).get("gold_evidence", []):
        if dataset == "pubmedqa":
            keys.add((str(item.get("pubid")), int(item.get("context_idx"))))
        elif dataset == "scifact":
            keys.add((str(item.get("doc_id")), int(item.get("sentence_index"))))
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
    return keys


def _passage_key(dataset: str, passage: dict[str, Any]) -> tuple[str, int]:
    metadata = passage.get("metadata") or {}
    if dataset == "pubmedqa":
        return (str(metadata.get("pubid")), int(metadata.get("context_idx")))
    if dataset == "scifact":
        return (str(metadata.get("doc_id")), int(metadata.get("sentence_index")))
    raise ValueError(f"Unsupported dataset: {dataset}")


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Select generation prompt A/B examples or compare A/B outputs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--pubmedqa-v1", type=Path, default=DEFAULT_PUBMEDQA_V1)
    select_parser.add_argument("--scifact-v1", type=Path, default=DEFAULT_SCIFACT_V1)
    select_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/generation_ab"),
    )
    select_parser.add_argument("--per-category-limit", type=int, default=10)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--dataset", choices=("pubmedqa", "scifact"), required=True)
    compare_parser.add_argument("--v1", type=Path, required=True)
    compare_parser.add_argument("--v2", type=Path, required=True)
    compare_parser.add_argument("--v3", type=Path, default=None)
    compare_parser.add_argument("--output-csv", type=Path, required=True)
    compare_parser.add_argument("--output-json", type=Path, default=None)

    compare_model_parser = subparsers.add_parser("compare-models")
    compare_model_parser.add_argument(
        "--dataset", choices=("pubmedqa", "scifact"), required=True
    )
    compare_model_parser.add_argument("--baseline", type=Path, required=True)
    compare_model_parser.add_argument("--candidate", type=Path, required=True)
    compare_model_parser.add_argument("--baseline-label", required=True)
    compare_model_parser.add_argument("--candidate-label", required=True)
    compare_model_parser.add_argument("--output-csv", type=Path, required=True)
    compare_model_parser.add_argument("--output-json", type=Path, default=None)

    args = parser.parse_args()
    if args.command == "select":
        summary = select_ab_examples(
            pubmedqa_v1=args.pubmedqa_v1,
            scifact_v1=args.scifact_v1,
            output_dir=args.output_dir,
            per_category_limit=args.per_category_limit,
        )
    elif args.command == "compare":
        summary = compare_ab_outputs(
            dataset=args.dataset,
            v1_path=args.v1,
            v2_path=args.v2,
            v3_path=args.v3,
            output_csv=args.output_csv,
            output_json=args.output_json,
        )
    elif args.command == "compare-models":
        summary = compare_model_outputs(
            dataset=args.dataset,
            baseline_path=args.baseline,
            candidate_path=args.candidate,
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            output_csv=args.output_csv,
            output_json=args.output_json,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
