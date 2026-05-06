"""Run generation from saved pooled-corpus retrieval artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_experiment.generation.parsing import parse_answer_json
from rag_experiment.generation.prompts import (
    PromptDefinition,
    default_prompt_id_for_dataset,
    format_artifact_retrieved_context,
    get_prompt,
)
from rag_experiment.model_clients import get_llm_profile
from rag_experiment.runners.artifacts import PROJECT_ROOT, error_record


SCHEMA_VERSION = "v0.1"
DATASET_CHOICES = ("pubmedqa", "scifact")
DEFAULT_MODEL_PROFILE = "rag_qwen_generation_v1"


def run_pooled_generation(
    *,
    input_path: Path,
    output_path: Path,
    dataset: str,
    model_profile: str = DEFAULT_MODEL_PROFILE,
    prompt_id: str | None = None,
    limit: int | None = None,
    example_ids: set[str] | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> Path:
    """Generate answers for a normalized pooled retrieval artifact."""
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"Unsupported dataset {dataset!r}. Available: {DATASET_CHOICES}")

    prompt = get_prompt(prompt_id or default_prompt_id_for_dataset(dataset))
    model = get_llm_profile(model_profile, **(model_overrides or {}))
    records = list(
        _iter_input_records(
            input_path,
            dataset=dataset,
            limit=limit,
            example_ids=example_ids,
        )
    )
    started_at = datetime.now(timezone.utc).isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            generated = _build_generated_record(
                record=record,
                input_path=input_path,
                dataset=dataset,
                prompt=prompt,
                model=model,
                model_profile=model_profile,
                model_overrides=model_overrides or {},
                started_at=started_at,
            )
            handle.write(json.dumps(generated, ensure_ascii=False) + "\n")

    return output_path


def _iter_input_records(
    path: Path,
    *,
    dataset: str,
    limit: int | None,
    example_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            _validate_input_record(record, dataset=dataset, line_number=line_number)
            example_id = str((record.get("example") or {}).get("id", ""))
            if example_ids is not None and example_id not in example_ids:
                continue
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if example_ids is not None and not records:
        raise ValueError("No records matched the selected example ids")
    return records


def _validate_input_record(
    record: dict[str, Any], *, dataset: str, line_number: int
) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"line {line_number}: expected schema_version={SCHEMA_VERSION!r}, "
            f"got {record.get('schema_version')!r}"
        )
    if record.get("dataset") != dataset:
        raise ValueError(
            f"line {line_number}: expected dataset={dataset!r}, "
            f"got {record.get('dataset')!r}"
        )
    for key in ("run", "example", "retrieved_passages"):
        if key not in record:
            raise ValueError(f"line {line_number}: missing required key {key!r}")


def _build_generated_record(
    *,
    record: dict[str, Any],
    input_path: Path,
    dataset: str,
    prompt: PromptDefinition,
    model: Any,
    model_profile: str,
    model_overrides: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    retrieved_passages = record.get("retrieved_passages") or []
    rendered_messages: list[dict[str, str]] = []
    raw_model_answer: str | None = None
    parsed_answer: dict[str, Any] | None = None
    prediction = {
        "answer": None,
        "cited_passage_ids": [],
        "metadata": {"source": "model"},
    }
    error = record.get("error")
    generation_error = None

    if error is None:
        try:
            question = str((record.get("example") or {}).get("query", ""))
            context = format_artifact_retrieved_context(retrieved_passages)
            rendered_messages = prompt.render(question=question, context=context)
            langchain_messages = prompt.render_langchain(
                question=question,
                context=context,
            )
            response = model.invoke(langchain_messages)
            raw_model_answer = _response_content_as_text(response.content)
            parsed_answer = parse_answer_json(raw_model_answer)
            metadata = {"source": "model"}
            if "evidence_summary" in parsed_answer:
                metadata["evidence_summary"] = parsed_answer["evidence_summary"]
            prediction = {
                "answer": _normalize_dataset_answer(dataset, parsed_answer["answer"]),
                "cited_passage_ids": parsed_answer["cited_passage_ids"],
                "metadata": metadata,
            }
        except Exception as exc:
            generation_error = error_record(exc)

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "run": _generation_run_metadata(
            record=record,
            input_path=input_path,
            prompt=prompt,
            model_profile=model_profile,
            model_overrides=model_overrides,
            started_at=started_at,
        ),
        "example": record["example"],
        "retrieved_passages": retrieved_passages,
        "prediction": prediction,
        "prompt": prompt.as_dict(),
        "rendered_messages": rendered_messages,
        "raw_model_answer": raw_model_answer,
        "parsed_answer": parsed_answer,
        "generation_error": generation_error,
        "error": error,
    }


def _generation_run_metadata(
    *,
    record: dict[str, Any],
    input_path: Path,
    prompt: PromptDefinition,
    model_profile: str,
    model_overrides: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    run = dict(record.get("run") or {})
    run["model_profile"] = model_profile
    run["prompt_id"] = prompt.id
    run["generation_input_path"] = str(input_path)
    run["generated_at"] = started_at
    if model_overrides:
        run["model_overrides"] = model_overrides
    return run


def _normalize_dataset_answer(dataset: str, answer: Any) -> str:
    text = str(answer or "").strip()
    if dataset == "pubmedqa":
        normalized = text.lower()
        if normalized in {"yes", "no", "maybe"}:
            return normalized
    elif dataset == "scifact":
        normalized = text.upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "SUPPORTS": "SUPPORT",
            "SUPPORTED": "SUPPORT",
            "REFUTE": "CONTRADICT",
            "REFUTES": "CONTRADICT",
            "REFUTED": "CONTRADICT",
            "CONTRADICTS": "CONTRADICT",
            "CONTRADICTION": "CONTRADICT",
            "NEI": "NOT_ENOUGH_INFO",
            "UNKNOWN": "NOT_ENOUGH_INFO",
        }
        canonical = aliases.get(normalized, normalized)
        if canonical in {"SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"}:
            return canonical

    raise ValueError(f"Unsupported {dataset} answer label: {answer!r}")


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


def _load_example_ids(
    *,
    example_ids: list[str] | None,
    example_ids_file: Path | None,
) -> set[str] | None:
    selected = {example_id.strip() for example_id in example_ids or [] if example_id.strip()}
    if example_ids_file is not None:
        with example_ids_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    selected.add(stripped)
    return selected or None


def _default_output_path(input_path: Path, dataset: str) -> Path:
    stem = input_path.stem.replace("_pooled", "_generated")
    if not stem.startswith(f"{dataset}_"):
        stem = f"{dataset}_{stem}"
    return PROJECT_ROOT / f"outputs/generation/{stem}.jsonl"


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run generation from a saved pooled retrieval artifact."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--model-profile", default=DEFAULT_MODEL_PROFILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--example-id",
        action="append",
        default=None,
        help="Run generation only for this example id. May be repeated.",
    )
    parser.add_argument(
        "--example-ids-file",
        type=Path,
        default=None,
        help="Optional newline-delimited example ids to generate.",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_path = args.output or _default_output_path(input_path, args.dataset)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    example_ids_file = args.example_ids_file
    if example_ids_file is not None and not example_ids_file.is_absolute():
        example_ids_file = PROJECT_ROOT / example_ids_file

    written = run_pooled_generation(
        input_path=input_path,
        output_path=output_path,
        dataset=args.dataset,
        prompt_id=args.prompt_id,
        model_profile=args.model_profile,
        limit=args.limit,
        example_ids=_load_example_ids(
            example_ids=args.example_id,
            example_ids_file=example_ids_file,
        ),
    )
    print(f"wrote={written}")


if __name__ == "__main__":
    _main()
