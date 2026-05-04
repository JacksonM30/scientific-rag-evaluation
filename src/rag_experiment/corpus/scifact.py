"""SciFact corpus helpers for pooled retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_experiment.data.hotpotqa import Passage
from rag_experiment.data.inspect_datasets import _ensure_scifact_data, _read_jsonl


def load_scifact_data(
    *,
    data_dir: Path,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Load SciFact corpus rows and train claims."""
    dataset_dir = _ensure_scifact_data(data_dir)
    corpus = {
        int(row["doc_id"]): row for row in _read_jsonl(dataset_dir / "corpus.jsonl")
    }
    claims = _read_jsonl(dataset_dir / "claims_train.jsonl")
    if not corpus:
        raise ValueError("SciFact corpus is empty")
    if not claims:
        raise ValueError("SciFact train claims are empty")
    return corpus, claims


def select_labeled_claims(
    claims: list[dict[str, Any]],
    *,
    corpus: dict[int, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Select deterministic labeled SciFact claims with usable rationale evidence."""
    selected: list[dict[str, Any]] = []
    for claim in claims:
        if scifact_label(claim) is None:
            continue
        if not scifact_gold_evidence(claim, corpus):
            continue
        selected.append(claim)
        if len(selected) >= limit:
            break
    if not selected:
        raise ValueError("No labeled SciFact claims with usable evidence were loaded")
    return selected


def select_scifact_corpus_doc_ids(
    *,
    corpus: dict[int, dict[str, Any]],
    claims: list[dict[str, Any]],
    corpus_doc_limit: int,
) -> list[int]:
    """Select corpus doc ids; corpus_doc_limit=0 means all corpus docs."""
    if corpus_doc_limit < 0:
        raise ValueError("corpus_doc_limit must be >= 0")
    required_doc_ids = {
        int(item["doc_id"])
        for claim in claims
        for item in scifact_gold_evidence(claim, corpus)
    }
    if corpus_doc_limit == 0:
        return sorted(corpus)

    doc_ids = list(required_doc_ids)
    for doc_id in sorted(corpus):
        if len(doc_ids) >= corpus_doc_limit:
            break
        if doc_id not in required_doc_ids:
            doc_ids.append(doc_id)
    if not doc_ids:
        raise ValueError("No SciFact corpus docs selected")
    return doc_ids


def build_scifact_passages(
    *,
    corpus: dict[int, dict[str, Any]],
    doc_ids: list[int],
) -> list[Passage]:
    """Convert selected SciFact abstracts into sentence-level passages."""
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


def scifact_label(claim: dict[str, Any]) -> str | None:
    """Return the first available claim-verification label."""
    for evidence_items in (claim.get("evidence") or {}).values():
        for evidence_item in evidence_items:
            label = evidence_item.get("label")
            if label:
                return str(label)
    return None


def scifact_gold_evidence(
    claim: dict[str, Any],
    corpus: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return unique rationale sentence keys for a SciFact claim."""
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
