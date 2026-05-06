from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from rag_experiment.generation.prompts import get_prompt
from rag_experiment.runners.run_pooled_generation import (
    _iter_input_records,
    _load_example_ids,
    _normalize_dataset_answer,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(example_id: str) -> dict:
    return {
        "schema_version": "v0.1",
        "dataset": "pubmedqa",
        "run": {"name": "test"},
        "example": {"id": example_id, "query": "q"},
        "retrieved_passages": [],
        "prediction": {"answer": "yes", "cited_passage_ids": []},
    }


class PooledGenerationTests(unittest.TestCase):
    def test_pubmedqa_v2_prompt_is_registered(self) -> None:
        prompt = get_prompt("pubmedqa_rag_json_v2")

        self.assertEqual(prompt.id, "pubmedqa_rag_json_v2")
        self.assertIn("claim implied by the question", prompt.system_template)
        self.assertIn("Choose cited_passage_ids first", prompt.system_template)
        self.assertIn(
            "The answer must be the conclusion supported by those cited passages",
            prompt.system_template,
        )
        self.assertIn(
            "Do not cite passages that are merely related",
            prompt.system_template,
        )
        self.assertIn("smallest set of passage IDs", prompt.system_template)

    def test_scifact_v2_prompt_is_registered(self) -> None:
        prompt = get_prompt("scifact_rag_json_v2")

        self.assertEqual(prompt.id, "scifact_rag_json_v2")
        self.assertIn("directly supports the claim", prompt.system_template)
        self.assertIn("Choose cited_passage_ids first", prompt.system_template)
        self.assertIn(
            "The label must be the conclusion supported by those cited passages",
            prompt.system_template,
        )
        self.assertIn(
            "Do not cite passages that are merely related",
            prompt.system_template,
        )
        self.assertIn(
            "Do not use NOT_ENOUGH_INFO as a default fallback",
            prompt.system_template,
        )

    def test_pubmedqa_v3_debug_prompt_is_registered(self) -> None:
        prompt = get_prompt("pubmedqa_rag_json_v3_debug")

        self.assertEqual(prompt.id, "pubmedqa_rag_json_v3_debug")
        self.assertIn("Choose cited_passage_ids first", prompt.system_template)
        self.assertIn("one-sentence evidence_summary", prompt.system_template)
        self.assertIn(
            "cited_passage_ids, evidence_summary, answer", prompt.system_template
        )
        self.assertIn('"evidence_summary": "one sentence"', prompt.human_template)

    def test_scifact_v3_debug_prompt_is_registered(self) -> None:
        prompt = get_prompt("scifact_rag_json_v3_debug")

        self.assertEqual(prompt.id, "scifact_rag_json_v3_debug")
        self.assertIn("Choose cited_passage_ids first", prompt.system_template)
        self.assertIn("one-sentence evidence_summary", prompt.system_template)
        self.assertIn(
            "cited_passage_ids, evidence_summary, answer", prompt.system_template
        )
        self.assertIn("SUPPORT|CONTRADICT|NOT_ENOUGH_INFO", prompt.human_template)

    def test_iter_input_records_filters_selected_example_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "artifact.jsonl"
            _write_jsonl(artifact, [_record("a"), _record("b"), _record("c")])

            records = _iter_input_records(
                artifact,
                dataset="pubmedqa",
                limit=None,
                example_ids={"b", "c"},
            )

        self.assertEqual([record["example"]["id"] for record in records], ["b", "c"])

    def test_iter_input_records_applies_limit_after_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "artifact.jsonl"
            _write_jsonl(artifact, [_record("a"), _record("b"), _record("c")])

            records = _iter_input_records(
                artifact,
                dataset="pubmedqa",
                limit=1,
                example_ids={"b", "c"},
            )

        self.assertEqual([record["example"]["id"] for record in records], ["b"])

    def test_load_example_ids_merges_cli_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ids_file = Path(temp_dir) / "ids.txt"
            ids_file.write_text("# debug cases\nb\n\nc\n", encoding="utf-8")

            self.assertEqual(
                _load_example_ids(example_ids=["a"], example_ids_file=ids_file),
                {"a", "b", "c"},
            )

    def test_normalize_pubmedqa_answer_accepts_only_known_labels(self) -> None:
        self.assertEqual(_normalize_dataset_answer("pubmedqa", " YES "), "yes")
        self.assertEqual(_normalize_dataset_answer("pubmedqa", "Maybe"), "maybe")
        with self.assertRaises(ValueError):
            _normalize_dataset_answer("pubmedqa", "unclear")


if __name__ == "__main__":
    unittest.main()
