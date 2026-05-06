from __future__ import annotations

import unittest

from rag_experiment.generation.parsing import parse_answer_json


class GenerationParsingTests(unittest.TestCase):
    def test_parse_answer_json_keeps_existing_schema(self) -> None:
        parsed = parse_answer_json(
            '{"answer": "yes", "cited_passage_ids": ["pubmedqa::1::0"]}'
        )

        self.assertEqual(
            parsed,
            {"answer": "yes", "cited_passage_ids": ["pubmedqa::1::0"]},
        )

    def test_parse_answer_json_preserves_evidence_summary_when_present(self) -> None:
        parsed = parse_answer_json(
            '{"cited_passage_ids": ["pubmedqa::1::0"], '
            '"evidence_summary": "The cited passage supports the claim.", '
            '"answer": "yes"}'
        )

        self.assertEqual(parsed["answer"], "yes")
        self.assertEqual(parsed["cited_passage_ids"], ["pubmedqa::1::0"])
        self.assertEqual(
            parsed["evidence_summary"], "The cited passage supports the claim."
        )

    def test_parse_answer_json_rejects_non_string_evidence_summary(self) -> None:
        with self.assertRaises(ValueError):
            parse_answer_json(
                '{"cited_passage_ids": ["pubmedqa::1::0"], '
                '"evidence_summary": ["not", "a", "string"], '
                '"answer": "yes"}'
            )


if __name__ == "__main__":
    unittest.main()
