from __future__ import annotations

import unittest

from eval.judge import extract_last_number, gsm8k_exact_match


class JudgeScoringTests(unittest.TestCase):
    def test_extract_last_number(self) -> None:
        text = "Let's solve it. Final answer: 72"
        self.assertEqual(extract_last_number(text), "72")

        float_text = "Result is 3.50"
        self.assertEqual(extract_last_number(float_text), "3.5")

    def test_gsm8k_exact_match(self) -> None:
        matched, ref, pred = gsm8k_exact_match("#### 72", "The answer is 72")
        self.assertTrue(matched)
        self.assertEqual(ref, "72")
        self.assertEqual(pred, "72")

        matched, _, _ = gsm8k_exact_match("#### 72", "The answer is 73")
        self.assertFalse(matched)


if __name__ == "__main__":
    unittest.main()
