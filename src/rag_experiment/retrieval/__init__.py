"""Retrieval adapters."""
from rag_experiment.retrieval.base import RetrievalResult
from rag_experiment.retrieval.bm25 import BM25Retriever
from rag_experiment.retrieval.dense import DenseRetriever
from rag_experiment.retrieval.factory import build_retriever
from rag_experiment.retrieval.hybrid import HybridRetriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "RetrievalResult",
    "build_retriever",
]
