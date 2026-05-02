"""Dense retrieval with LangChain in-memory vectors and API embeddings."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from rag_experiment.data.hotpotqa import Passage
from rag_experiment.model_clients.embeddings import get_embedding_model
from rag_experiment.retrieval.base import (
    RetrievalResult,
    passage_from_metadata,
    passage_to_metadata,
)


class DenseRetriever:
    """Dense retriever for small experiment runs."""

    def __init__(
        self,
        passages: list[Passage],
        *,
        embedding_model: Embeddings | None = None,
        top_k: int = 3,
    ) -> None:
        if not passages:
            raise ValueError("DenseRetriever requires at least one passage")
        self.passages = passages
        self.top_k = top_k
        self.embedding_model = embedding_model or get_embedding_model()
        self._vectorstore = self._build_vectorstore(passages)

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievalResult]:
        effective_top_k = top_k if top_k is not None else self.top_k
        documents_and_scores = self._vectorstore.similarity_search_with_score(
            query,
            k=effective_top_k,
        )
        return [
            RetrievalResult(
                passage=passage_from_metadata(document.metadata, document.page_content),
                score=float(score),
                rank=rank,
                metadata={
                    "retriever": "dense",
                    "score_kind": "cosine_similarity",
                },
            )
            for rank, (document, score) in enumerate(documents_and_scores, start=1)
        ]

    def as_langchain_retriever(self, *, top_k: int | None = None):
        effective_top_k = top_k if top_k is not None else self.top_k
        return self._vectorstore.as_retriever(search_kwargs={"k": effective_top_k})

    def _build_vectorstore(self, passages: list[Passage]) -> InMemoryVectorStore:
        return InMemoryVectorStore.from_texts(
            texts=[passage.text for passage in passages],
            embedding=self.embedding_model,
            metadatas=[passage_to_metadata(passage) for passage in passages],
        )
