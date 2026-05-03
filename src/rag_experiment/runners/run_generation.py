"""Real RAG generation runner."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_experiment.data.hotpotqa import HotpotExample, load_hotpot_jsonl
from rag_experiment.generation.parsing import parse_answer_json
from rag_experiment.generation.prompts import PromptDefinition, format_retrieved_context, get_prompt
from rag_experiment.model_clients import get_llm_profile
from rag_experiment.retrieval.factory import build_retriever
from rag_experiment.runners.artifacts import (
    error_record,
    example_record,
    jsonable_config,
    load_json,
    resolve_config_paths,
    resolve_path,
    retrieval_record,
)


def run_config(config_path: str | Path) -> Path:
    config_file = resolve_path(config_path)
    config = load_json(config_file)
    resolved_config = resolve_config_paths(config)

    if resolved_config["generation"].get("dry_run", True):
        raise ValueError("This runner requires generation.dry_run=false.")

    examples = load_hotpot_jsonl(
        resolved_config["dataset"]["path"],
        limit=resolved_config["dataset"].get("limit"),
    )
    output_path = resolved_config["output"]["path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = get_prompt(resolved_config["prompt"]["id"])
    model_profile = resolved_config["generation"]["model_profile"]
    model_overrides = resolved_config["generation"].get("model_overrides", {})
    model = get_llm_profile(model_profile, **model_overrides)
    started_at = datetime.now(timezone.utc).isoformat()

    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            record = _build_generation_record(
                example=example,
                config=resolved_config,
                prompt=prompt,
                model=model,
                started_at=started_at,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def _build_generation_record(
    *,
    example: HotpotExample,
    config: dict[str, Any],
    prompt: PromptDefinition,
    model: Any,
    started_at: str,
) -> dict[str, Any]:
    retrieved_passages: list[dict[str, Any]] = []
    rendered_messages: list[dict[str, str]] = []
    raw_model_answer: str | None = None
    parsed_answer: dict[str, Any] | None = None
    cited_passage_ids: list[str] = []

    try:
        top_k = int(config["retriever"]["top_k"])
        retriever = build_retriever(config["retriever"], example.passages())
        results = retriever.retrieve(example.question, top_k=top_k)
        retrieved_passages = [retrieval_record(result) for result in results]
        context = format_retrieved_context(results)
        rendered_messages = prompt.render(question=example.question, context=context)
        langchain_messages = prompt.render_langchain(
            question=example.question,
            context=context,
        )

        response = model.invoke(langchain_messages)
        raw_model_answer = _response_content_as_text(response.content)
        parsed_answer = parse_answer_json(raw_model_answer)
        cited_passage_ids = parsed_answer["cited_passage_ids"]

        return _record(
            example=example,
            config=config,
            prompt=prompt,
            started_at=started_at,
            retrieved_passages=retrieved_passages,
            rendered_messages=rendered_messages,
            raw_model_answer=raw_model_answer,
            parsed_answer=parsed_answer,
            cited_passage_ids=cited_passage_ids,
            error=None,
        )
    except Exception as exc:
        return _record(
            example=example,
            config=config,
            prompt=prompt,
            started_at=started_at,
            retrieved_passages=retrieved_passages,
            rendered_messages=rendered_messages,
            raw_model_answer=raw_model_answer,
            parsed_answer=parsed_answer,
            cited_passage_ids=cited_passage_ids,
            error=error_record(exc),
        )


def _record(
    *,
    example: HotpotExample,
    config: dict[str, Any],
    prompt: PromptDefinition,
    started_at: str,
    retrieved_passages: list[dict[str, Any]],
    rendered_messages: list[dict[str, str]],
    raw_model_answer: str | None,
    parsed_answer: dict[str, Any] | None,
    cited_passage_ids: list[str],
    error: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "run_name": config["run_name"],
        "created_at": started_at,
        "dry_run": False,
        "config": jsonable_config(config),
        "example": example_record(example),
        "retrieved_passages": retrieved_passages,
        "prompt": prompt.as_dict(),
        "rendered_messages": rendered_messages,
        "raw_model_answer": raw_model_answer,
        "parsed_answer": parsed_answer,
        "cited_passage_ids": cited_passage_ids,
        "error": error,
    }


def _response_content_as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    return str(content)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run a real RAG generation pass.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    output_path = run_config(args.config)
    print(f"wrote={output_path}")


if __name__ == "__main__":
    _main()
