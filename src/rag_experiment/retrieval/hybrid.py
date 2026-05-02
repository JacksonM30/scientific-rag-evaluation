"""Hybrid retrieval with LangChain ensemble fusion."""

from __future__ import annotations

from langchain_classic.retrievers import EnsembleRetriever

from rag_experiment.retrieval.base import (
    RetrievalResult,
    passage_from_metadata,
)
from rag_experiment.retrieval.bm25 import BM25Retriever
from rag_experiment.retrieval.dense import DenseRetriever


class HybridRetriever:
    """Combine already-built retrievers with LangChain weighted RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        *,
        top_k: int = 3,
        candidate_k: int | None = None,
        weights: tuple[float, float] = (0.5, 0.5),
        rrf_k: int = 60,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.top_k = top_k
        self.candidate_k = candidate_k or max(top_k * 4, top_k)
        self.weights = weights
        self.rrf_k = rrf_k

    def retrieve(
        self, query: str, *, top_k: int | None = None
    ) -> list[RetrievalResult]:
        effective_top_k = top_k if top_k is not None else self.top_k
        candidate_k = max(self.candidate_k, effective_top_k)
        documents = self._build_ensemble(candidate_k).invoke(query)[:effective_top_k]

        return [
            RetrievalResult(
                passage=passage_from_metadata(document.metadata, document.page_content),
                score=None,
                rank=rank,
                metadata={
                    "retriever": "hybrid",
                    "fusion": "langchain_ensemble_rrf",
                    "weights": list(self.weights),
                    "rrf_k": self.rrf_k,
                },
            )
            for rank, document in enumerate(documents, start=1)
        ]

    def _build_ensemble(self, candidate_k: int) -> EnsembleRetriever:
        return EnsembleRetriever(
            retrievers=[
                self.bm25_retriever.as_langchain_retriever(top_k=candidate_k),
                self.dense_retriever.as_langchain_retriever(top_k=candidate_k),
            ],
            weights=list(self.weights),
            c=self.rrf_k,
            id_key="id",
        )
