"""Shared helpers for pooled-corpus retrieval runners."""

from __future__ import annotations

from rag_experiment.data.hotpotqa import Passage
from rag_experiment.model_clients.embeddings import get_embedding_model
from rag_experiment.retrieval.base import Retriever
from rag_experiment.retrieval.bm25 import BM25Retriever
from rag_experiment.retrieval.dense import DenseRetriever
from rag_experiment.retrieval.hybrid import HybridRetriever


RETRIEVER_CHOICES = ("bm25", "dense", "hybrid")


def build_pooled_retriever(
    name: str,
    passages: list[Passage],
    *,
    top_k: int,
    candidate_k: int | None = None,
) -> Retriever:
    """Build a retriever for a pooled passage list."""
    if name == "bm25":
        return BM25Retriever(passages, top_k=top_k)

    if name == "dense":
        return DenseRetriever(
            passages,
            top_k=top_k,
            embedding_model=get_embedding_model(),
        )

    if name == "hybrid":
        effective_candidate_k = min(
            len(passages),
            candidate_k if candidate_k is not None else max(top_k * 4, top_k),
        )
        bm25_retriever = BM25Retriever(passages, top_k=effective_candidate_k)
        dense_retriever = DenseRetriever(
            passages,
            top_k=effective_candidate_k,
            embedding_model=get_embedding_model(),
        )
        return HybridRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            top_k=top_k,
            candidate_k=effective_candidate_k,
        )

    available = ", ".join(RETRIEVER_CHOICES)
    raise ValueError(f"Unknown retriever {name!r}. Available: {available}")


def pooled_output_path(dataset: str, retriever_name: str):
    from rag_experiment.runners.artifacts import PROJECT_ROOT

    return PROJECT_ROOT / f"outputs/retrieval/{dataset}_{retriever_name}_pooled_v01.jsonl"
