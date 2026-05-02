"""Factory for config-driven retrievers."""

from __future__ import annotations

from typing import Any

from rag_experiment.data.hotpotqa import Passage
from rag_experiment.model_clients.embeddings import get_embedding_model
from rag_experiment.retrieval.base import Retriever
from rag_experiment.retrieval.bm25 import BM25Retriever
from rag_experiment.retrieval.dense import DenseRetriever
from rag_experiment.retrieval.hybrid import HybridRetriever


def build_retriever(config: dict[str, Any], passages: list[Passage]) -> Retriever:
    name = config.get("name", "bm25")
    top_k = int(config.get("top_k", 3))

    if name == "bm25":
        return BM25Retriever(passages, top_k=top_k)

    if name == "dense":
        return DenseRetriever(
            passages,
            top_k=top_k,
            embedding_model=_build_embedding_model(config),
        )

    if name == "hybrid":
        hybrid_config = config.get("hybrid", {})
        candidate_k = _candidate_k(config, len(passages))
        bm25_retriever = BM25Retriever(
            passages,
            top_k=candidate_k,
        )
        dense_retriever = DenseRetriever(
            passages,
            top_k=candidate_k,
            embedding_model=_build_embedding_model(config),
        )
        return HybridRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            top_k=top_k,
            rrf_k=int(hybrid_config.get("rrf_k", 60)),
            candidate_k=candidate_k,
            weights=_hybrid_weights(hybrid_config),
        )

    raise ValueError("Unknown retriever name {!r}. Use bm25, dense, or hybrid.".format(name))


def _build_embedding_model(config: dict[str, Any]):
    embedding_config = config.get("embedding", {})
    provider = embedding_config.get("provider", "dashscope")
    kwargs = {
        key: value
        for key, value in embedding_config.items()
        if key != "provider" and value is not None
    }
    return get_embedding_model(provider=provider, **kwargs)


def _candidate_k(config: dict[str, Any], passage_count: int) -> int:
    top_k = int(config.get("top_k", 3))
    hybrid_config = config.get("hybrid", {})
    configured = hybrid_config.get("candidate_k")
    if configured is not None:
        return min(passage_count, int(configured))
    return min(passage_count, max(top_k * 4, top_k))


def _hybrid_weights(hybrid_config: dict[str, Any]) -> tuple[float, float]:
    weights = hybrid_config.get("weights", [0.5, 0.5])
    if len(weights) != 2:
        raise ValueError("hybrid.weights must contain exactly two values: BM25 and dense")
    return (float(weights[0]), float(weights[1]))
