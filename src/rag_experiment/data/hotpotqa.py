"""Dataset loading utilities for HotpotQA-style examples."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Passage:
    """A retrievable text unit from a titled context."""

    id: str
    example_id: str
    title: str
    sentence_index: int
    text: str


@dataclass(frozen=True)
class HotpotExample:
    """A normalized HotpotQA-style question with titled context."""

    id: str
    question: str
    answer: str
    context: tuple[tuple[str, tuple[str, ...]], ...]
    supporting_facts: tuple[tuple[str, int], ...]
    type: str | None = None
    level: str | None = None

    def passages(self) -> list[Passage]:
        passages: list[Passage] = []
        for title, sentences in self.context:
            for sentence_index, text in enumerate(sentences):
                passage_id = f"{self.id}::{title}::{sentence_index}"
                passages.append(
                    Passage(
                        id=passage_id,
                        example_id=self.id,
                        title=title,
                        sentence_index=sentence_index,
                        text=text,
                    )
                )
        return passages


def load_hotpot_jsonl(path: str | Path, *, limit: int | None = None) -> list[HotpotExample]:
    """Load HotpotQA-style JSONL examples from disk."""

    source = Path(path)
    examples: list[HotpotExample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
                examples.append(_parse_hotpot_example(raw))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"{source}:{line_number}: invalid HotpotQA-style row: {exc}") from exc
            if limit is not None and len(examples) >= limit:
                break
    return examples


def _parse_hotpot_example(raw: dict[str, Any]) -> HotpotExample:
    context = tuple(_parse_context(raw["context"]))
    supporting_facts = tuple(_parse_supporting_facts(raw.get("supporting_facts", [])))
    return HotpotExample(
        id=str(raw["id"]),
        question=str(raw["question"]),
        answer=str(raw["answer"]),
        type=str(raw["type"]) if raw.get("type") is not None else None,
        level=str(raw["level"]) if raw.get("level") is not None else None,
        context=context,
        supporting_facts=supporting_facts,
    )


def _parse_context(raw_context: Iterable[Any]) -> Iterable[tuple[str, tuple[str, ...]]]:
    for item in raw_context:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError(f"context item must be [title, sentences], got {item!r}")
        title, sentences = item
        if not isinstance(sentences, list | tuple):
            raise ValueError(f"context sentences for {title!r} must be a list")
        yield str(title), tuple(str(sentence) for sentence in sentences)


def _parse_supporting_facts(raw_facts: Iterable[Any]) -> Iterable[tuple[str, int]]:
    for item in raw_facts:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError(f"supporting fact must be [title, sentence_index], got {item!r}")
        title, sentence_index = item
        yield str(title), int(sentence_index)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize HotpotQA-style JSONL.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = load_hotpot_jsonl(args.path, limit=args.limit)
    passage_count = sum(len(example.passages()) for example in examples)
    print(f"examples={len(examples)}")
    print(f"passages={passage_count}")
    if examples:
        first = examples[0]
        print(f"first_id={first.id}")
        print(f"first_question={first.question}")


if __name__ == "__main__":
    _main()
