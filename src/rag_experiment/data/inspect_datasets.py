"""Download and inspect external datasets before normalizing them."""

from __future__ import annotations

import argparse
import json
import statistics
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HF_CACHE = PROJECT_ROOT / "data/raw/huggingface"
DEFAULT_SCIFACT_DIR = PROJECT_ROOT / "data/raw/scifact"
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def inspect_pubmedqa(*, cache_dir: Path, examples: int) -> dict[str, Any]:
    """Load and summarize the labeled PubMedQA subset."""
    from datasets import load_dataset

    dataset = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split="train",
        cache_dir=str(cache_dir),
    )
    final_decisions = Counter(str(row["final_decision"]) for row in dataset)
    context_lengths = [
        len((row.get("context") or {}).get("contexts") or []) for row in dataset
    ]

    return {
        "dataset": "pubmedqa",
        "source": "qiaojin/PubMedQA",
        "config": "pqa_labeled",
        "split": "train",
        "cache_dir": str(cache_dir),
        "row_count": len(dataset),
        "features": _jsonable_features(dataset.features),
        "label_counts": dict(sorted(final_decisions.items())),
        "context_sentence_counts": _numeric_summary(context_lengths),
        "example_rows": [_pubmedqa_example(row) for row in dataset.select(range(min(examples, len(dataset))))],
        "future_mapping": {
            "query": "question",
            "gold_answer": "final_decision",
            "passages": "context.contexts, one passage per abstract section/sentence block",
            "metadata": "pubid, context.labels, context.meshes, long_answer",
        },
        "notes": [
            "The labeled subset has 1000 expert-labeled rows.",
            "The answer space is yes/no/maybe via final_decision.",
            "There is no explicit gold evidence sentence id; evidence matching will likely be context-level or section-level.",
        ],
    }


def inspect_scifact(*, data_dir: Path, examples: int) -> dict[str, Any]:
    """Download the official SciFact tarball and summarize claims/corpus files."""
    dataset_dir = _ensure_scifact_data(data_dir)
    corpus = _read_jsonl(dataset_dir / "corpus.jsonl")
    claims_by_split = {
        "train": _read_jsonl(dataset_dir / "claims_train.jsonl"),
        "validation": _read_jsonl(dataset_dir / "claims_dev.jsonl"),
        "test": _read_jsonl(dataset_dir / "claims_test.jsonl"),
    }

    label_counts: Counter[str] = Counter()
    claims_with_evidence = 0
    evidence_sets = 0
    rationale_sentence_counts: list[int] = []
    for split in ("train", "validation"):
        for claim in claims_by_split[split]:
            evidence = claim.get("evidence") or {}
            if evidence:
                claims_with_evidence += 1
            for evidence_list in evidence.values():
                for evidence_item in evidence_list:
                    label_counts[str(evidence_item.get("label"))] += 1
                    evidence_sets += 1
                    rationale_sentence_counts.append(
                        len(evidence_item.get("sentences") or [])
                    )

    abstract_lengths = [len(row.get("abstract") or []) for row in corpus]

    return {
        "dataset": "scifact",
        "source": SCIFACT_URL,
        "local_data_dir": str(dataset_dir),
        "row_counts": {
            "corpus": len(corpus),
            "claims_train": len(claims_by_split["train"]),
            "claims_validation": len(claims_by_split["validation"]),
            "claims_test": len(claims_by_split["test"]),
        },
        "field_summary": {
            "corpus": sorted(corpus[0].keys()) if corpus else [],
            "claims": sorted(claims_by_split["train"][0].keys()) if claims_by_split["train"] else [],
        },
        "label_counts_train_validation": dict(sorted(label_counts.items())),
        "claims_with_evidence_train_validation": claims_with_evidence,
        "evidence_sets_train_validation": evidence_sets,
        "abstract_sentence_counts": _numeric_summary(abstract_lengths),
        "rationale_sentence_counts": _numeric_summary(rationale_sentence_counts),
        "example_corpus_rows": corpus[:examples],
        "example_claim_rows": {
            split: rows[:examples] for split, rows in claims_by_split.items()
        },
        "future_mapping": {
            "query": "claim",
            "gold_answer": "evidence label, usually SUPPORT or CONTRADICT for labeled evidence",
            "passages": "corpus abstracts split into sentence passages",
            "gold_evidence": "evidence doc id plus rationale sentence indices",
            "metadata": "claim id, cited_doc_ids, corpus doc_id, title, structured flag",
        },
        "notes": [
            "Train and validation claims include labels/evidence; test claims are unlabeled.",
            "A single claim can have multiple evidence documents or evidence sets.",
            "SciFact is claim verification, not ordinary QA, so generation prompts should ask for a label and cited evidence.",
        ],
    }


def _ensure_scifact_data(data_dir: Path) -> Path:
    archive_path = data_dir / "data.tar.gz"
    extract_dir = data_dir / "latest"
    dataset_dir = extract_dir / "data"
    if dataset_dir.exists():
        return dataset_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        urllib.request.urlretrieve(SCIFACT_URL, archive_path)

    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract(archive, extract_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Expected SciFact data directory at {dataset_dir}")
    return dataset_dir


def _safe_extract(archive: tarfile.TarFile, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    for member in archive.getmembers():
        member_path = (target_root / member.name).resolve()
        if target_root not in (member_path, *member_path.parents):
            raise ValueError(f"Unsafe path in SciFact archive: {member.name}")
    archive.extractall(target_root)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _pubmedqa_example(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("context") or {}
    contexts = context.get("contexts") or []
    return {
        "pubid": row.get("pubid"),
        "question": row.get("question"),
        "final_decision": row.get("final_decision"),
        "long_answer": row.get("long_answer"),
        "context_labels": context.get("labels") or [],
        "context_count": len(contexts),
        "first_context": contexts[0] if contexts else None,
    }


def _numeric_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 3),
        "median": statistics.median(values),
    }


def _jsonable_features(features: Any) -> dict[str, Any]:
    return json.loads(json.dumps(features.to_dict()))


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"Dataset: {summary['dataset']}")
    print("=" * (9 + len(summary["dataset"])))
    if summary["dataset"] == "pubmedqa":
        print(f"source: {summary['source']} / {summary['config']}")
        print(f"split: {summary['split']}")
        print(f"rows: {summary['row_count']}")
        print(f"label_counts: {summary['label_counts']}")
        print(f"context_sentence_counts: {summary['context_sentence_counts']}")
        print("features:")
        print(json.dumps(summary["features"], indent=2, ensure_ascii=False))
        print("examples:")
        print(json.dumps(summary["example_rows"], indent=2, ensure_ascii=False))
    else:
        print(f"source: {summary['source']}")
        print(f"local_data_dir: {summary['local_data_dir']}")
        print(f"row_counts: {summary['row_counts']}")
        print(f"field_summary: {summary['field_summary']}")
        print(f"label_counts_train_validation: {summary['label_counts_train_validation']}")
        print(f"abstract_sentence_counts: {summary['abstract_sentence_counts']}")
        print(f"rationale_sentence_counts: {summary['rationale_sentence_counts']}")
        print("example_claim_rows:")
        print(json.dumps(summary["example_claim_rows"], indent=2, ensure_ascii=False))
        print("example_corpus_rows:")
        print(json.dumps(summary["example_corpus_rows"], indent=2, ensure_ascii=False))
    print("future_mapping:")
    print(json.dumps(summary["future_mapping"], indent=2, ensure_ascii=False))
    print("notes:")
    for note in summary["notes"]:
        print(f"- {note}")
    print()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and inspect PubMedQA/SciFact schemas before normalization."
    )
    parser.add_argument("dataset", choices=["pubmedqa", "scifact", "all"])
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--hf-cache-dir", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--scifact-dir", type=Path, default=DEFAULT_SCIFACT_DIR)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    if args.dataset in {"pubmedqa", "all"}:
        summaries.append(inspect_pubmedqa(cache_dir=args.hf_cache_dir, examples=args.examples))
    if args.dataset in {"scifact", "all"}:
        summaries.append(inspect_scifact(data_dir=args.scifact_dir, examples=args.examples))

    for summary in summaries:
        _print_summary(summary)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload: Any = summaries[0] if len(summaries) == 1 else summaries
        args.save_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    _main()
