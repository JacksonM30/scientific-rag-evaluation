from __future__ import annotations

import unittest

from rag_experiment.model_clients.profiles import PROFILES


class ModelProfileTests(unittest.TestCase):
    def test_qwen35_flash_v3_debug_profile_disables_thinking(self) -> None:
        profile = PROFILES["rag_qwen35_flash_v3_debug"]

        self.assertEqual(profile["provider"], "qwen")
        self.assertEqual(profile["model"], "qwen3.5-flash")
        self.assertEqual(profile["temperature"], 0)
        self.assertEqual(profile["max_tokens"], 1024)
        self.assertEqual(profile["extra_body"], {"enable_thinking": False})

    def test_qwen35_flash_thinking_v3_debug_profile_enables_thinking(self) -> None:
        profile = PROFILES["rag_qwen35_flash_thinking_v3_debug"]

        self.assertEqual(profile["provider"], "qwen")
        self.assertEqual(profile["model"], "qwen3.5-flash")
        self.assertEqual(profile["temperature"], 0)
        self.assertEqual(profile["max_tokens"], 1024)
        self.assertEqual(profile["extra_body"], {"enable_thinking": True})

    def test_report_qwen3_30b_a3b_profile_disables_thinking(self) -> None:
        profile = PROFILES["rag_qwen3_30b_a3b_v3_report"]

        self.assertEqual(profile["provider"], "qwen")
        self.assertEqual(profile["model"], "qwen3-30b-a3b")
        self.assertEqual(profile["temperature"], 0)
        self.assertEqual(profile["max_tokens"], 1024)
        self.assertEqual(profile["extra_body"], {"enable_thinking": False})

    def test_report_qwen3_30b_a3b_instruct_2507_profile_disables_thinking(self) -> None:
        profile = PROFILES["rag_qwen3_30b_a3b_instruct_2507_v3_report"]

        self.assertEqual(profile["provider"], "qwen")
        self.assertEqual(profile["model"], "qwen3-30b-a3b-instruct-2507")
        self.assertEqual(profile["temperature"], 0)
        self.assertEqual(profile["max_tokens"], 1024)
        self.assertEqual(profile["extra_body"], {"enable_thinking": False})


if __name__ == "__main__":
    unittest.main()
