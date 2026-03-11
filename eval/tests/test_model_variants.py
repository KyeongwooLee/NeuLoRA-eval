from __future__ import annotations

import unittest
from pathlib import Path

from eval.common import EvalConfig, EvalTurn
from eval.modeling import HFChatGenerator, VariantExecutor


class _DummyRouter:
    def route(self, query: str, default_style: str = "direct", **kwargs):
        _ = query
        _ = kwargs
        return default_style, {default_style: 1.0}


class _DummyGen(HFChatGenerator):
    def __init__(self):
        pass


class _RecordingGenerator:
    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        return "mock answer"


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

    def test_b_variant_uses_raw_prompt_and_variant_flag(self) -> None:
        cfg = EvalConfig(
            mode="architecture",
            benchmarks=["gsm8k"],
            max_samples=1,
            seed=42,
            judge_model="gemini-2.5-flash",
            output_dir=Path("/tmp/eval_test_out"),
        )
        generator = _RecordingGenerator()
        executor = VariantExecutor(
            config=cfg,
            generator=generator,  # type: ignore[arg-type]
            router=_DummyRouter(),
            seed=42,
        )
        turn = EvalTurn(
            benchmark="gsm8k",
            session_id="s1",
            turn_id=1,
            prompt="한국어로 바로 답해줘",
            reference="1",
            history=[],
            metadata={},
        )

        result = executor.run_architecture_turn(
            variant="B",
            turn=turn,
            generated_history=[],
            retrieved_docs=[],
        )

        self.assertEqual(result.model_id, executor.b_vanilla_model_name)
        self.assertIsNone(result.style)
        self.assertEqual(generator.calls[0]["variant"], "B")
        self.assertEqual(generator.calls[0]["user_prompt"], turn.prompt)


if __name__ == "__main__":
    unittest.main()
