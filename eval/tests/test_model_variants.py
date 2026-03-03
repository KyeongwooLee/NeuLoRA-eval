from __future__ import annotations

import unittest
from pathlib import Path

from eval.common import EvalConfig
from eval.modeling import HFChatGenerator, VariantExecutor


class _DummyRouter:
    def route(self, query: str, default_style: str = "direct"):
        _ = query
        return default_style, {default_style: 1.0}


class _DummyGen(HFChatGenerator):
    def __init__(self):
        pass


class ModelVariantTests(unittest.TestCase):
    def test_b1_random_style_is_deterministic(self) -> None:
        cfg = EvalConfig(
            mode="architecture",
            benchmarks=["gsm8k"],
            max_samples=5,
            seed=42,
            judge_model="gemini-2.5-flash",
            output_dir=Path("/tmp/eval_test_out"),
        )

        executor = VariantExecutor(
            config=cfg,
            generator=_DummyGen(),
            router=_DummyRouter(),
            seed=42,
        )

        first = executor._deterministic_random_style("s1", 1)
        second = executor._deterministic_random_style("s1", 1)
        third = executor._deterministic_random_style("s1", 2)

        self.assertEqual(first, second)
        self.assertIn(first, {"direct", "socratic", "scaffolding", "feedback"})
        self.assertIn(third, {"direct", "socratic", "scaffolding", "feedback"})


if __name__ == "__main__":
    unittest.main()
