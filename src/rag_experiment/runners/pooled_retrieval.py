"""Shared helpers for pooled-corpus retrieval runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_experiment.data.hotpotqa import Passage
from rag_experiment.model_clients.providers import (
    DASHSCOPE_OPENAI_BASE_URL,
    EMBEDDING_PROVIDERS,
)
from rag_experiment.model_clients.embeddings import get_embedding_model
from rag_experiment.retrieval.base import Retriever
from rag_experiment.retrieval.bm25 import BM25Retriever
from rag_experiment.retrieval.dense import DenseRetriever
from rag_experiment.retrieval.embedding_cache import EmbeddingCacheSpec
from rag_experiment.retrieval.hybrid import HybridRetriever


RETRIEVER_CHOICES = ("bm25", "dense", "hybrid")
DEFAULT_EMBEDDING_PROVIDER = "dashscope"
DEFAULT_EMBEDDING_MODEL = EMBEDDING_PROVIDERS[DEFAULT_EMBEDDING_PROVIDER][
    "default_model"
]
DEFAULT_EMBEDDING_DIMENSIONS = EMBEDDING_PROVIDERS[DEFAULT_EMBEDDING_PROVIDER].get(
    "default_dimensions"
)


def build_pooled_retriever(
    name: str,
    passages: list[Passage],
    *,
    top_k: int,
    candidate_k: int | None = None,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensions: int | None = DEFAULT_EMBEDDING_DIMENSIONS,
    embedding_cache_dir: Path | None = None,
    embedding_cache_namespace: str = "pooled",
) -> Retriever:
    """Build a retriever for a pooled passage list."""
    if name == "bm25":
        return BM25Retriever(passages, top_k=top_k)

    if name == "dense":
        embedding_model = get_embedding_model(
            model=embedding_model_name,
            dimensions=embedding_dimensions,
        )
        return DenseRetriever(
            passages,
            top_k=top_k,
            embedding_model=embedding_model,
            embedding_cache_dir=embedding_cache_dir,
            embedding_cache_spec=_cache_spec(
                namespace=embedding_cache_namespace,
                model=embedding_model_name,
                dimensions=embedding_dimensions,
            ),
        )

    if name == "hybrid":
        effective_candidate_k = min(
            len(passages),
            candidate_k if candidate_k is not None else max(top_k * 4, top_k),
        )
        bm25_retriever = BM25Retriever(passages, top_k=effective_candidate_k)
        embedding_model = get_embedding_model(
            model=embedding_model_name,
            dimensions=embedding_dimensions,
        )
        dense_retriever = DenseRetriever(
            passages,
            top_k=effective_candidate_k,
            embedding_model=embedding_model,
            embedding_cache_dir=embedding_cache_dir,
            embedding_cache_spec=_cache_spec(
                namespace=embedding_cache_namespace,
                model=embedding_model_name,
                dimensions=embedding_dimensions,
            ),
        )
        return HybridRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            top_k=top_k,
            candidate_k=effective_candidate_k,
        )

    available = ", ".join(RETRIEVER_CHOICES)
    raise ValueError(f"Unknown retriever {name!r}. Available: {available}")


def embedding_run_metadata(retriever: Retriever) -> dict[str, Any]:
    """Return embedding metadata for dense-capable pooled retrievers."""
    dense_retriever = None
    if isinstance(retriever, DenseRetriever):
        dense_retriever = retriever
    elif isinstance(retriever, HybridRetriever):
        dense_retriever = retriever.dense_retriever

    if dense_retriever is None:
        return {}
    return dense_retriever.cache_info.as_run_metadata()


def pooled_output_path(dataset: str, retriever_name: str):
    from rag_experiment.runners.artifacts import PROJECT_ROOT

    return PROJECT_ROOT / f"outputs/retrieval/{dataset}_{retriever_name}_pooled_v01.jsonl"


def _cache_spec(
    *,
    namespace: str,
    model: str,
    dimensions: int | None,
) -> EmbeddingCacheSpec:
    return EmbeddingCacheSpec(
        namespace=namespace,
        provider=DEFAULT_EMBEDDING_PROVIDER,
        model=model,
        dimensions=dimensions,
        base_url=DASHSCOPE_OPENAI_BASE_URL,
    )
