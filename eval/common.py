from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BENCHMARK_CHOICES: tuple[str, ...] = ("comta", "mathdial", "mtbench", "gsm8k")
MODE_CHOICES: tuple[str, ...] = ("architecture", "adapters", "models")
LLM_MODE_CHOICES: tuple[str, ...] = ("api", "vessel")
ROUTER_MODE_CHOICES: tuple[str, ...] = ("cent", "mlp")

STYLE_MODELS: dict[str, str] = {
    "direct": "RiverWon/NeuLoRA-direct",
    "socratic": "RiverWon/NeuLoRA-socratic",
    "scaffolding": "RiverWon/NeuLoRA-scaffolding",
    "feedback": "RiverWon/NeuLoRA-feedback",
}

ARCHITECTURE_VARIANTS: tuple[str, ...] = ("B", "B0", "B1", "B2", "P")
ADAPTER_VARIANTS: tuple[str, ...] = ("direct", "socratic", "scaffolding", "feedback")


@dataclass(frozen=True)
class EvalTurn:
    benchmark: str
    session_id: str
    turn_id: int
    prompt: str
    reference: str | None
    history: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkData:
    benchmark: str
    train_docs: list[dict]
    eval_turns: list[EvalTurn]
    notes: list[str] = field(default_factory=list)


@dataclass
class EvalConfig:
    mode: str
    benchmarks: list[str]
    max_samples: int
    seed: int
    judge_model: str
    output_dir: Path
    llm_mode: str = field(default_factory=lambda: os.getenv("LLM_MODE", "api"))
    router_mode: str = field(default_factory=lambda: os.getenv("ROUTER_MODE", "cent"))
    target_models: list[str] = field(default_factory=list)

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    base_model_name: str = field(default_factory=lambda: os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"))
    embedding_model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"))
    hf_api_key: str = field(default_factory=lambda: os.getenv("HF_API_KEY", ""))
    genai_api_key: str = field(default_factory=lambda: os.getenv("GENAI_API_KEY", ""))
    backend_url: str = field(default_factory=lambda: os.getenv("EVAL_BACKEND_URL", "http://127.0.0.1:8800"))

    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "cache")
    data_cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "cache" / "datasets")
    chroma_root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "cache" / "chroma")

    def ensure_dirs(self) -> None:
        self.router_mode = (self.router_mode or "cent").strip().lower()
        if self.router_mode not in ROUTER_MODE_CHOICES:
            raise ValueError(
                f"Unsupported router_mode: {self.router_mode}. "
                f"Supported: {', '.join(ROUTER_MODE_CHOICES)}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_root_dir.mkdir(parents=True, exist_ok=True)


def parse_benchmarks(value: str) -> list[str]:
    raw = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not raw:
        return list(BENCHMARK_CHOICES)
    if len(raw) == 1 and raw[0] == "all":
        return list(BENCHMARK_CHOICES)
    invalid = [token for token in raw if token not in BENCHMARK_CHOICES]
    if invalid:
        raise ValueError(f"Unsupported benchmark(s): {invalid}")
    return raw


def parse_models(value: str) -> list[str]:
    raw = [token.strip().upper() for token in value.split(",") if token.strip()]
    if not raw:
        return []
    invalid = [token for token in raw if token not in ARCHITECTURE_VARIANTS]
    if invalid:
        raise ValueError(
            f"Unsupported model variant(s): {invalid}. "
            f"Supported: {', '.join(ARCHITECTURE_VARIANTS)}"
        )
    return list(dict.fromkeys(raw))


def summarize_history(history: list[tuple[str, str]], max_turns: int = 6) -> str:
    if not history:
        return ""
    sliced = history[-max_turns:]
    lines = []
    for role, content in sliced:
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)
