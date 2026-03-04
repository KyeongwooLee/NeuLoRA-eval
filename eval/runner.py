from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .common import (
    ADAPTER_VARIANTS,
    ARCHITECTURE_VARIANTS,
    BenchmarkData,
    EvalConfig,
)
from .data_loading import load_benchmark, sample_turns
from .judge import GeminiJudge, gsm8k_exact_match, mean_or_zero
from .modeling import HFChatGenerator, VariantExecutor
from .retrieval import ChromaCorpus, EmbeddingBackend, StyleRouter


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _benchmark_output_path(output_dir: Path, mode: str, benchmark: str, variant: str) -> Path:
    sanitized = variant.replace("/", "_")
    return output_dir / mode / f"{benchmark}__{sanitized}.json"


def _summary_output_path(output_dir: Path, mode: str) -> Path:
    return output_dir / mode / "summary.json"


def _variant_list(mode: str) -> list[str]:
    if mode == "architecture":
        return list(ARCHITECTURE_VARIANTS)
    if mode == "adapters":
        return list(ADAPTER_VARIANTS)
    raise ValueError(f"Unsupported mode: {mode}")


def _required_loads(benchmarks: list[str]) -> list[str]:
    required = list(dict.fromkeys(benchmarks))
    if "mtbench" in benchmarks:
        for supporting in ("comta", "mathdial", "gsm8k"):
            if supporting not in required:
                required.append(supporting)
    return required


def _prepare_corpus_docs(benchmark: str, loaded_data: dict[str, BenchmarkData]) -> list[dict]:
    if benchmark != "mtbench":
        return loaded_data[benchmark].train_docs

    docs: list[dict] = []
    for source in ("comta", "mathdial", "gsm8k"):
        docs.extend(loaded_data[source].train_docs)
    return docs


def _build_run_metadata(config: EvalConfig, benchmark: str, variant: str, sample_count: int) -> dict[str, Any]:
    return {
        "timestamp_utc": _iso_now(),
        "mode": config.mode,
        "benchmark": benchmark,
        "model_variant": variant,
        "seed": config.seed,
        "sample_count": sample_count,
        "base_model_name": config.base_model_name,
        "embedding_model_name": config.embedding_model_name,
        "backend_url": config.backend_url,
        "llm_mode": "api",
        "judge_model": config.judge_model,
    }


def run_matrix(
    config: EvalConfig,
    *,
    generator: HFChatGenerator | None = None,
    judge: GeminiJudge | None = None,
) -> dict[str, Any]:
    config.ensure_dirs()
    os.environ["LLM_MODE"] = "api"

    if generator is None:
        try:
            generator = HFChatGenerator(
                hf_api_key=config.hf_api_key,
                backend_url=config.backend_url,
            )
        except TypeError:
            # Backward compatibility for test doubles or legacy generator signatures.
            generator = HFChatGenerator(hf_api_key=config.hf_api_key)
    if judge is None:
        judge = GeminiJudge(api_key=config.genai_api_key, model_name=config.judge_model)

    embedder = EmbeddingBackend(model_name=config.embedding_model_name)
    router = StyleRouter(config.project_root / "LangGraph" / "router_model.json", embedder=embedder)
    executor = VariantExecutor(config=config, generator=generator, router=router, seed=config.seed)

    loaded: dict[str, BenchmarkData] = {}
    for benchmark in _required_loads(config.benchmarks):
        loaded[benchmark] = load_benchmark(benchmark, config)

    variants = _variant_list(config.mode)
    run_results: list[dict[str, Any]] = []

    for benchmark in config.benchmarks:
        benchmark_data = loaded[benchmark]
        corpus_docs = _prepare_corpus_docs(benchmark, loaded)
        corpus_dir = config.chroma_root_dir / benchmark
        corpus = ChromaCorpus(
            persist_dir=corpus_dir,
            collection_name=f"{benchmark}_seed_{config.seed}",
            embedder=embedder,
        )
        corpus_size = corpus.build(corpus_docs)

        sampled_turns = sample_turns(benchmark_data.eval_turns, config.max_samples, config.seed)
        sampled_turns = sorted(sampled_turns, key=lambda item: (item.session_id, item.turn_id))

        for variant in variants:
            session_histories: dict[str, list[tuple[str, str]]] = {}
            items: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            scores: list[float] = []

            for turn in sampled_turns:
                session_history = session_histories.setdefault(turn.session_id, [])
                if benchmark in {"comta", "mathdial"} and not turn.reference:
                    session_history.append(("student", turn.prompt))
                    errors.append(
                        {
                            "session_id": turn.session_id,
                            "turn_id": turn.turn_id,
                            "error": "Skipped: reference is null for tutoring benchmark",
                        }
                    )
                    continue

                # Avoid double-RAG: backend /api/chat already performs retrieval.
                # Local Chroma retrieval is only needed for B2 LPITutor path.
                needs_local_retrieval = config.mode == "architecture" and variant == "B2"
                if needs_local_retrieval:
                    retrieval_query = f"{turn.prompt}\n\n{turn.metadata.get('question', '')}".strip()
                    retrieved_docs = corpus.query(retrieval_query, top_k=5)
                else:
                    retrieved_docs = []

                generation = None
                for _attempt in range(3):
                    if config.mode == "architecture":
                        generation = executor.run_architecture_turn(
                            variant=variant,
                            turn=turn,
                            generated_history=session_history,
                            retrieved_docs=retrieved_docs,
                        )
                    else:
                        generation = executor.run_adapter_turn(
                            adapter_style=variant,
                            turn=turn,
                            generated_history=session_history,
                            retrieved_docs=retrieved_docs,
                        )
                    if (generation.response or "").strip():
                        break

                assert generation is not None
                model_answer = generation.response
                score = 0.0
                judge_payload: dict[str, Any] = {}

                if generation.error:
                    errors.append(
                        {
                            "session_id": turn.session_id,
                            "turn_id": turn.turn_id,
                            "error": generation.error,
                        }
                    )

                if not (model_answer or "").strip():
                    errors.append(
                        {
                            "session_id": turn.session_id,
                            "turn_id": turn.turn_id,
                            "error": "Skipped: empty response after 3 attempts",
                        }
                    )
                    continue

                if benchmark in {"comta", "mathdial"}:
                    judged = judge.score_tutoring_turn(
                        benchmark=benchmark,
                        question=turn.prompt,
                        reference_answer=turn.reference or "",
                        model_answer=model_answer,
                    )
                    score = judged.overall_score
                    judge_payload = judged.payload
                elif benchmark == "mtbench":
                    history_text = "\n".join(
                        f"{role}: {text}" for role, text in session_history[-6:]
                    )
                    judged = judge.score_mtbench_turn(
                        question=turn.prompt,
                        model_answer=model_answer,
                        conversation_history=history_text,
                    )
                    score = judged.overall_score * 10.0
                    judge_payload = judged.payload
                elif benchmark == "gsm8k":
                    matched, ref_num, pred_num = gsm8k_exact_match(turn.reference or "", model_answer)
                    score = 100.0 if matched else 0.0
                    judge_payload = {
                        "metric": "exact_match_numeric",
                        "matched": matched,
                        "reference_number": ref_num,
                        "predicted_number": pred_num,
                    }
                else:
                    raise ValueError(f"Unsupported benchmark: {benchmark}")

                scores.append(score)
                items.append(
                    {
                        "benchmark": benchmark,
                        "session_id": turn.session_id,
                        "turn_id": turn.turn_id,
                        "prompt": turn.prompt,
                        "reference": turn.reference,
                        "response": model_answer,
                        "style": generation.style,
                        "model_id": generation.model_id,
                        "retrieval_hits": generation.retrieval_hits,
                        "score": round(score, 3),
                        "judge": judge_payload,
                        "metadata": turn.metadata,
                    }
                )

                session_history.append(("student", turn.prompt))
                session_history.append(("assistant", model_answer))

            benchmark_score = round(mean_or_zero(scores), 3)
            payload = {
                "metadata": _build_run_metadata(
                    config=config,
                    benchmark=benchmark,
                    variant=variant,
                    sample_count=len(items),
                ),
                "notes": benchmark_data.notes,
                "aggregate": {
                    "benchmark_score": benchmark_score,
                    "overall_score": benchmark_score,
                    "corpus_size": corpus_size,
                    "num_items": len(items),
                    "num_errors": len(errors),
                },
                "items": items,
                "errors": errors,
            }

            out_path = _benchmark_output_path(config.output_dir, config.mode, benchmark, variant)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            run_results.append(
                {
                    "benchmark": benchmark,
                    "variant": variant,
                    "benchmark_score": benchmark_score,
                    "path": str(out_path),
                }
            )

    by_variant: dict[str, list[float]] = {}
    for row in run_results:
        by_variant.setdefault(row["variant"], []).append(float(row["benchmark_score"]))

    macro_by_variant = {
        variant: round(mean(scores), 3) if scores else 0.0
        for variant, scores in by_variant.items()
    }

    comparison_table = sorted(
        [
            {
                "variant": variant,
                "macro_average_score": score,
            }
            for variant, score in macro_by_variant.items()
        ],
        key=lambda item: item["macro_average_score"],
        reverse=True,
    )

    summary = {
        "metadata": {
            "timestamp_utc": _iso_now(),
            "mode": config.mode,
            "benchmarks": config.benchmarks,
            "max_samples": config.max_samples,
            "seed": config.seed,
            "judge_model": config.judge_model,
            "base_model_name": config.base_model_name,
            "embedding_model_name": config.embedding_model_name,
            "backend_url": config.backend_url,
            "llm_mode": "api",
        },
        "runs": run_results,
        "macro_by_variant": macro_by_variant,
        "comparison_table": comparison_table,
    }

    summary_path = _summary_output_path(config.output_dir, config.mode)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["path"] = str(summary_path)
    return summary
