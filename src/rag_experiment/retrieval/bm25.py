"""LangChain-backed BM25 retrieval."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever

from rag_experiment.data.hotpotqa import Passage, load_hotpot_jsonl


RANK_BM25_INSTALL_HELP = (
    "LangChain BM25 requires the optional `rank-bm25` package.\n"
    "Install it in the project environment:\n\n"
    "  conda activate LLM\n"
    "  python -m pip install rank-bm25\n"
)


@dataclass(frozen=True)
class RetrievalResult:
    passage: Passage
    score: float | None
    rank: int


class BM25Retriever:
    """Thin adapter around LangChain's BM25 retriever."""

    def __init__(self, passages: list[Passage], *, top_k: int = 3) -> None:
        if not passages:
            raise ValueError("BM25Retriever requires at least one passage")
        self.passages = passages
        self.top_k = top_k
        self._retriever = self._build_retriever(passages, top_k=top_k)

    def search(self, query: str, *, top_k: int = 3) -> list[RetrievalResult]:
        self._retriever.k = top_k
        documents = self._retriever.invoke(query)
        return [
            RetrievalResult(
                passage=_passage_from_metadata(document.metadata, document.page_content),
                score=None,
                rank=rank,
            )
            for rank, document in enumerate(documents, start=1)
        ]

    def _build_retriever(
        self, passages: list[Passage], *, top_k: int
    ) -> LangChainBM25Retriever:
        if find_spec("rank_bm25") is None:
            raise RuntimeError(RANK_BM25_INSTALL_HELP)

        texts = [passage.text for passage in passages]
        ids = [passage.id for passage in passages]
        metadatas = [
            {
                "id": passage.id,
                "example_id": passage.example_id,
                "title": passage.title,
                "sentence_index": passage.sentence_index,
            }
            for passage in passages
        ]
        return LangChainBM25Retriever.from_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            k=top_k,
        )


def _passage_from_metadata(metadata: dict, text: str) -> Passage:
    return Passage(
        id=str(metadata["id"]),
        example_id=str(metadata["example_id"]),
        title=str(metadata["title"]),
        sentence_index=int(metadata["sentence_index"]),
        text=text,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Preview BM25 retrieval over HotpotQA-style JSONL.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    examples = load_hotpot_jsonl(args.path)
    example = examples[args.example_index]
    retriever = BM25Retriever(example.passages())
    results = retriever.search(example.question, top_k=args.top_k)

    print(f"question={example.question}")
    print(f"answer={example.answer}")
    for result in results:
        passage = result.passage
        print(f"{result.rank}. title={passage.title} sent={passage.sentence_index}")
        print(f"   {passage.text}")


if __name__ == "__main__":
    _main()
