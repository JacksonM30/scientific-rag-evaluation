"""Shared retrieval types and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rag_experiment.data.hotpotqa import Passage


@dataclass(frozen=True)
class RetrievalResult:
    passage: Passage
    score: float | None
    rank: int
    metadata: dict = field(default_factory=dict)


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievalResult]:
        """Return ranked retrieval results for a query."""


def passage_to_metadata(passage: Passage) -> dict:
    return {
        "id": passage.id,
        "example_id": passage.example_id,
        "title": passage.title,
        "sentence_index": passage.sentence_index,
    }


def passage_from_metadata(metadata: dict, text: str) -> Passage:
    return Passage(
        id=str(metadata["id"]),
        example_id=str(metadata["example_id"]),
        title=str(metadata["title"]),
        sentence_index=int(metadata["sentence_index"]),
        text=text,
    )
