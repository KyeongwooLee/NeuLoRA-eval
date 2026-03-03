from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .common import MODE_CHOICES, EvalConfig, parse_benchmarks
    from .runner import run_matrix
except ImportError:  # pragma: no cover - direct script execution fallback
    from common import MODE_CHOICES, EvalConfig, parse_benchmarks
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
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvalConfig(
        mode=args.mode,
        benchmarks=parse_benchmarks(args.benchmarks),
        max_samples=args.max_samples,
        seed=args.seed,
        judge_model=args.judge_model,
        backend_url=args.backend_url,
        output_dir=args.output_dir,
    )

    summary = run_matrix(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
