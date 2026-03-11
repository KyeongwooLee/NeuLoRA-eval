from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .common import (
        LLM_MODE_CHOICES,
        MODE_CHOICES,
        ROUTER_MODE_CHOICES,
        EvalConfig,
        parse_benchmarks,
        parse_models,
    )
    from .runner import run_matrix
except ImportError:  # pragma: no cover - direct script execution fallback
    from common import (
        LLM_MODE_CHOICES,
        MODE_CHOICES,
        ROUTER_MODE_CHOICES,
        EvalConfig,
        parse_benchmarks,
        parse_models,
    )
    from runner import run_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeuLoRA benchmark evaluation matrix.")
    parser.add_argument("--mode", type=str, default="architecture", choices=MODE_CHOICES)
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="comta,mathdial,mtbench,gsm8k",
        help="Comma-separated list. Supported: comta,mathdial,mtbench,gsm8k or all",
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", type=str, default="gemini-2.5-flash")
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://127.0.0.1:8800",
        help="NeuLoRA backend URL for /api/chat generation.",
    )
    parser.add_argument(
        "--llm-mode",
        type=str,
        choices=LLM_MODE_CHOICES,
        default=None,
        help="Override LLM_MODE env for eval metadata/runtime propagation (api or vessel).",
    )
    parser.add_argument(
        "--router",
        type=str,
        choices=ROUTER_MODE_CHOICES,
        default="cent",
        help="Style router mode: cent (centroid cosine) or mlp (2-layer classifier).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated architecture variants for --mode=models. Supported: B,B0,B1,B2,P",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_kwargs = dict(
        mode=args.mode,
        benchmarks=parse_benchmarks(args.benchmarks),
        max_samples=args.max_samples,
        seed=args.seed,
        judge_model=args.judge_model,
        backend_url=args.backend_url,
        router_mode=args.router,
        output_dir=args.output_dir,
        target_models=parse_models(args.models),
    )
    if args.mode == "models" and not config_kwargs["target_models"]:
        raise ValueError("--mode=models requires --models with at least one model ID.")
    if args.llm_mode:
        config_kwargs["llm_mode"] = args.llm_mode
    config = EvalConfig(**config_kwargs)

    summary = run_matrix(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
