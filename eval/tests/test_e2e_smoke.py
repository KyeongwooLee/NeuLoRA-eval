from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.common import BenchmarkData, EvalConfig, EvalTurn
from eval.runner import run_matrix


class _FakeGenerator:
    def __init__(self, hf_api_key: str):
        _ = hf_api_key

    def generate(self, **kwargs):
        _ = kwargs
        return "mock answer"


class _FakeJudge:
    def __init__(self, api_key: str, model_name: str):
        _ = api_key
        _ = model_name

    class _Result:
        def __init__(self, score: float):
            self.overall_score = score
            self.payload = {"score": score}

    def score_tutoring_turn(self, **kwargs):
        _ = kwargs
        return self._Result(80.0)

    def score_mtbench_turn(self, **kwargs):
        _ = kwargs
        return self._Result(8.0)


class _FakeEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name


class _FakeRouter:
    def __init__(self, router_model_path: Path, embedder: _FakeEmbedder):
        _ = router_model_path
        _ = embedder

    def route(self, query: str, default_style: str = "direct"):
        _ = query
        return default_style, {default_style: 1.0}


class _FakeCorpus:
    def __init__(self, persist_dir: Path, collection_name: str, embedder: _FakeEmbedder):
        _ = persist_dir
        _ = collection_name
        _ = embedder

    def build(self, docs, batch_size: int = 128):
        _ = batch_size
        return len(list(docs))

    def query(self, query: str, top_k: int = 5):
        _ = query
        _ = top_k
        return [{"text": "cached context", "metadata": {}}]


class E2ESmokeTests(unittest.TestCase):
    def _fake_load_benchmark(self, benchmark: str, config: EvalConfig) -> BenchmarkData:
        _ = config
        turns = [
            EvalTurn(
                benchmark=benchmark,
                session_id=f"{benchmark}_s1",
                turn_id=i,
                prompt=f"question {i}",
                reference="reference",
                history=[],
                metadata={},
            )
            for i in range(1, 6)
        ]
        docs = [{"id": f"{benchmark}_doc_{i}", "text": "doc text", "metadata": {}} for i in range(5)]
        return BenchmarkData(benchmark=benchmark, train_docs=docs, eval_turns=turns)

    def test_run_matrix_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = EvalConfig(
                mode="architecture",
                benchmarks=["comta"],
                max_samples=5,
                seed=42,
                judge_model="gemini-2.5-flash",
                output_dir=Path(tmp_dir),
            )

            with (
                patch("eval.runner.HFChatGenerator", _FakeGenerator),
                patch("eval.runner.GeminiJudge", _FakeJudge),
                patch("eval.runner.EmbeddingBackend", _FakeEmbedder),
                patch("eval.runner.StyleRouter", _FakeRouter),
                patch("eval.runner.ChromaCorpus", _FakeCorpus),
                patch("eval.runner.load_benchmark", self._fake_load_benchmark),
            ):
                summary = run_matrix(cfg)

            self.assertIn("runs", summary)
            self.assertGreaterEqual(len(summary["runs"]), 1)
            self.assertTrue(Path(summary["path"]).exists())


if __name__ == "__main__":
    unittest.main()
