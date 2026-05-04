"""Run pooled-corpus SciFact retrieval and write normalized v0.1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_experiment.corpus.scifact import (
    build_scifact_passages,
    load_scifact_data,
    scifact_gold_evidence,
    scifact_label,
    select_labeled_claims,
    select_scifact_corpus_doc_ids,
)
from rag_experiment.data.inspect_datasets import DEFAULT_SCIFACT_DIR
from rag_experiment.retrieval.base import RetrievalResult, Retriever
from rag_experiment.runners.artifacts import PROJECT_ROOT, error_record
from rag_experiment.runners.pooled_retrieval import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    RETRIEVER_CHOICES,
    build_pooled_retriever,
    embedding_run_metadata,
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
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensions: int | None = DEFAULT_EMBEDDING_DIMENSIONS,
    embedding_cache_dir: Path | None = PROJECT_ROOT / "outputs/embedding_cache",
) -> Path:
    """Build a pooled SciFact retrieval artifact from labeled train claims."""
    corpus, train_claims = load_scifact_data(data_dir=data_dir)
    claims = select_labeled_claims(
        train_claims,
        corpus=corpus,
        limit=limit,
    )
    doc_ids = select_scifact_corpus_doc_ids(
        corpus=corpus,
        claims=claims,
        corpus_doc_limit=corpus_doc_limit,
    )
    passages = build_scifact_passages(corpus=corpus, doc_ids=doc_ids)
    retriever = build_pooled_retriever(
        retriever_name,
        passages,
        top_k=top_k,
        embedding_model_name=embedding_model,
        embedding_dimensions=embedding_dimensions,
        embedding_cache_dir=embedding_cache_dir,
        embedding_cache_namespace=(
            f"scifact_docs{corpus_doc_limit}_doccount{len(doc_ids)}_passages{len(passages)}"
        ),
    )
    embedding_metadata = embedding_run_metadata(retriever)

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
                corpus_doc_count=len(doc_ids),
                passage_count=len(passages),
                embedding_metadata=embedding_metadata,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def _build_record(
    *,
    claim: dict[str, Any],
    corpus: dict[int, dict[str, Any]],
    retriever: Retriever,
    retriever_name: str,
    top_k: int,
    limit: int,
    corpus_doc_limit: int,
    corpus_doc_count: int,
    passage_count: int,
    embedding_metadata: dict[str, Any],
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
            corpus_doc_count=corpus_doc_count,
            passage_count=passage_count,
            embedding_metadata=embedding_metadata,
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
            corpus_doc_count=corpus_doc_count,
            passage_count=passage_count,
            embedding_metadata=embedding_metadata,
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
    corpus_doc_count: int,
    passage_count: int,
    embedding_metadata: dict[str, Any],
    error: dict[str, str] | None,
) -> dict[str, Any]:
    label = scifact_label(claim)
    evidence = scifact_gold_evidence(claim, corpus)
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
            "corpus_doc_count": corpus_doc_count,
            "passage_count": passage_count,
        }
        | embedding_metadata,
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
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSIONS,
        help="Embedding vector dimensions for models that support it; use 0 for provider default.",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/embedding_cache",
    )
    parser.add_argument("--no-embedding-cache", action="store_true")
    args = parser.parse_args()

    output_path = args.output or pooled_output_path("scifact", args.retriever)
    output_path = run_scifact_retrieval(
        retriever_name=args.retriever,
        limit=args.limit,
        top_k=args.top_k,
        output_path=output_path,
        data_dir=args.scifact_dir,
        corpus_doc_limit=args.corpus_doc_limit,
        embedding_model=args.embedding_model,
        embedding_dimensions=(
            args.embedding_dimensions if args.embedding_dimensions != 0 else None
        ),
        embedding_cache_dir=None if args.no_embedding_cache else args.embedding_cache_dir,
    )
    print(f"wrote={output_path}")


if __name__ == "__main__":
    _main()
