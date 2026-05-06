from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from rag_experiment.analysis.generation_ab import (
    compare_ab_outputs,
    compare_model_outputs,
    select_ab_examples,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _pubmedqa_record(
    example_id: str,
    *,
    gold_answer: str = "yes",
    prediction: str = "maybe",
    cited_ids: list[str] | None = None,
    evidence_summary: str | None = None,
) -> dict:
    metadata = {"source": "model"}
    parsed_answer = {
        "answer": prediction,
        "cited_passage_ids": cited_ids
        if cited_ids is not None
        else [f"pubmedqa::{example_id}::0"],
    }
    if evidence_summary is not None:
        metadata["evidence_summary"] = evidence_summary
        parsed_answer["evidence_summary"] = evidence_summary
    return {
        "schema_version": "v0.1",
        "dataset": "pubmedqa",
        "run": {"name": "test"},
        "example": {
            "id": example_id,
            "query": "q",
            "gold_answer": gold_answer,
            "gold_evidence": [{"pubid": example_id, "context_idx": 0}],
        },
        "retrieved_passages": [
            {
                "passage_id": f"pubmedqa::{example_id}::0",
                "rank": 1,
                "text": "gold",
                "metadata": {
                    "dataset": "pubmedqa",
                    "pubid": example_id,
                    "context_idx": 0,
                },
            }
        ],
        "prediction": {
            "answer": prediction,
            "cited_passage_ids": parsed_answer["cited_passage_ids"],
            "metadata": metadata,
        },
        "parsed_answer": parsed_answer,
        "generation_error": None,
        "error": None,
    }


def _scifact_record(
    example_id: str,
    *,
    gold_answer: str = "SUPPORT",
    prediction: str | None = "CONTRADICT",
    cited_ids: list[str] | None = None,
    generation_error: dict | None = None,
) -> dict:
    return {
        "schema_version": "v0.1",
        "dataset": "scifact",
        "run": {"name": "test"},
        "example": {
            "id": example_id,
            "query": "claim",
            "gold_answer": gold_answer,
            "gold_evidence": [{"doc_id": example_id, "sentence_index": 0}],
        },
        "retrieved_passages": [
            {
                "passage_id": f"scifact::{example_id}::0",
                "rank": 1,
                "text": "gold",
                "metadata": {
                    "dataset": "scifact",
                    "doc_id": example_id,
                    "sentence_index": 0,
                },
            }
        ],
        "prediction": {
            "answer": prediction,
            "cited_passage_ids": cited_ids
            if cited_ids is not None
            else [f"scifact::{example_id}::0"],
        },
        "generation_error": generation_error,
        "error": None,
    }


class GenerationABTests(unittest.TestCase):
    def test_select_ab_examples_writes_fixed_id_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pubmedqa = root / "pubmedqa.jsonl"
            scifact = root / "scifact.jsonl"
            _write_jsonl(
                pubmedqa,
                [
                    _pubmedqa_record("p1", cited_ids=["not-in-retrieved"]),
                    _pubmedqa_record("p2"),
                ],
            )
            _write_jsonl(
                scifact,
                [
                    _scifact_record(
                        "s1",
                        prediction=None,
                        cited_ids=[],
                        generation_error={"type": "ValueError"},
                    ),
                    _scifact_record("s2", cited_ids=[]),
                ],
            )

            summary = select_ab_examples(
                pubmedqa_v1=pubmedqa,
                scifact_v1=scifact,
                output_dir=root / "ab",
                per_category_limit=1,
            )

            self.assertEqual(summary["datasets"]["pubmedqa"]["selected_ids"], ["p1", "p2"])
            self.assertEqual(summary["datasets"]["scifact"]["selected_ids"], ["s1", "s2"])
            self.assertTrue((root / "ab/pubmedqa_ab_selected_ids.txt").exists())
            self.assertTrue((root / "ab/scifact_ab_selected_cases.csv").exists())

    def test_compare_ab_outputs_counts_fixes_and_regressions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v1 = root / "v1.jsonl"
            v2 = root / "v2.jsonl"
            _write_jsonl(
                v1,
                [
                    _pubmedqa_record("p1", prediction="maybe"),
                    _pubmedqa_record("p2", prediction="yes"),
                ],
            )
            _write_jsonl(
                v2,
                [
                    _pubmedqa_record("p1", prediction="yes"),
                    _pubmedqa_record("p2", prediction="maybe"),
                ],
            )

            summary = compare_ab_outputs(
                dataset="pubmedqa",
                v1_path=v1,
                v2_path=v2,
                output_csv=root / "compare.csv",
                output_json=root / "compare.json",
            )

            self.assertEqual(summary["row_count"], 2)
            self.assertEqual(summary["answer_fixes"], 1)
            self.assertEqual(summary["answer_regressions"], 1)
            self.assertTrue((root / "compare.csv").exists())

    def test_compare_ab_outputs_can_include_v3_evidence_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v1 = root / "v1.jsonl"
            v2 = root / "v2.jsonl"
            v3 = root / "v3.jsonl"
            _write_jsonl(v1, [_pubmedqa_record("p1", prediction="maybe")])
            _write_jsonl(v2, [_pubmedqa_record("p1", prediction="maybe")])
            _write_jsonl(
                v3,
                [
                    _pubmedqa_record(
                        "p1",
                        prediction="yes",
                        evidence_summary="The cited passage supports the claim.",
                    )
                ],
            )

            summary = compare_ab_outputs(
                dataset="pubmedqa",
                v1_path=v1,
                v2_path=v2,
                v3_path=v3,
                output_csv=root / "compare.csv",
                output_json=root / "compare.json",
            )

            csv_text = (root / "compare.csv").read_text(encoding="utf-8")
            self.assertEqual(summary["v3_answer_accuracy"], 1.0)
            self.assertEqual(summary["v3_answer_fixes"], 1)
            self.assertIn("v3_evidence_summary", csv_text)
            self.assertIn("The cited passage supports the claim.", csv_text)

    def test_compare_model_outputs_counts_candidate_fixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            _write_jsonl(baseline, [_pubmedqa_record("p1", prediction="maybe")])
            _write_jsonl(candidate, [_pubmedqa_record("p1", prediction="yes")])

            summary = compare_model_outputs(
                dataset="pubmedqa",
                baseline_path=baseline,
                candidate_path=candidate,
                baseline_label="qwen3-8b-v3",
                candidate_label="qwen3.5-flash-v3",
                output_csv=root / "compare_models.csv",
                output_json=root / "compare_models.json",
            )

            csv_text = (root / "compare_models.csv").read_text(encoding="utf-8")
            self.assertEqual(summary["baseline_answer_accuracy"], 0.0)
            self.assertEqual(summary["candidate_answer_accuracy"], 1.0)
            self.assertEqual(summary["answer_fixes"], 1)
            self.assertIn("qwen3.5-flash-v3", csv_text)


if __name__ == "__main__":
    unittest.main()
