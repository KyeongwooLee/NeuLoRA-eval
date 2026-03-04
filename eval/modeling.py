from __future__ import annotations

import json
import os
import random
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

from .common import STYLE_MODELS, EvalConfig, EvalTurn, summarize_history
from .retrieval import StyleRouter


STYLE_PROMPTS: dict[str, str] = {
    "direct": "Give a direct answer with key concepts and concise step-by-step reasoning.",
    "socratic": "Use a Socratic tutoring style. Ask guiding questions before giving conclusions.",
    "scaffolding": "Give partial hints and gradual scaffolding rather than a full solution at once.",
    "feedback": "Give formative feedback: identify strengths, then one concrete improvement step.",
}


@dataclass
class GenerationResult:
    response: str
    model_id: str
    style: str | None
    retrieval_hits: int
    error: str | None = None


class HFChatGenerator:
    def __init__(
        self,
        hf_api_key: str,
        *,
        backend_url: str,
        timeout: float = 120.0,
        disable_web_search: bool = True,
        disable_relevance_check: bool = True,
    ):
        # keep `hf_api_key` arg for backward compatibility with existing runner wiring.
        self.hf_api_key = hf_api_key
        self.backend_url = backend_url.rstrip("/")
        self.timeout = timeout
        self.disable_web_search = disable_web_search
        self.disable_relevance_check = disable_relevance_check
        if not self.backend_url:
            raise RuntimeError("backend_url is required for /api/chat generation.")

    def generate(
        self,
        *,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        history: list[tuple[str, str]],
        thread_id: str,
        forced_style: str | None = None,
        variant: str | None = None,
        max_tokens: int = 256,
    ) -> str:
        endpoint = f"{self.backend_url}/api/chat"
        payload = {
            "message": user_prompt,
            "thread_id": thread_id,
            "forced_style": forced_style,
            "variant": variant,
            "disable_web_search": self.disable_web_search,
            "disable_relevance_check": self.disable_relevance_check,
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            return str(parsed.get("answer", "") or "").strip()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"/api/chat HTTP {exc.code}: {body[:400]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"/api/chat URL error: {exc}") from exc


class VariantExecutor:
    def __init__(
        self,
        config: EvalConfig,
        generator: HFChatGenerator,
        router: StyleRouter,
        *,
        seed: int,
    ):
        self.config = config
        self.generator = generator
        self.router = router
        self.seed = seed
        self.random = random.Random(seed)
        self._lpitutor = self._load_lpitutor()

    def _load_lpitutor(self):
        lpi_root = self.config.project_root.parent / "LPITutor"
        if not lpi_root.exists():
            return None
        if str(lpi_root) not in sys.path:
            sys.path.insert(0, str(lpi_root))
        try:
            from response_generation import ResponseGenerator as LPITutorResponseGenerator

            return LPITutorResponseGenerator(
                model_name=self.config.base_model_name,
                embedding_model_name=self.config.embedding_model_name,
                hf_api_key=self.config.hf_api_key,
                hf_provider=os.getenv("HF_PROVIDER", "auto"),
            )
        except Exception:
            return None

    def _build_user_prompt(self, turn: EvalTurn) -> str:
        rules = ["Respond in English only."]
        if turn.benchmark == "gsm8k":
            rules.append("For GSM8K, end with: Final Answer: <number>.")
        rule_text = " ".join(rules)
        return f"{turn.prompt}\n\n{rule_text}"

    def _build_system_prompt(self, style: str | None, *, level: str | None = None) -> str:
        style_prompt = STYLE_PROMPTS.get(style or "direct", STYLE_PROMPTS["direct"])
        base = (
            "You are a helpful math tutor. Keep answers accurate, concise, and safe. "
            "Do not hallucinate facts beyond retrieved context when context is provided."
        )
        if level:
            base += f" Teach at {level} level."
        return f"{base} {style_prompt}".strip()

    def _deterministic_random_style(self, session_id: str, turn_id: int) -> str:
        choices = list(STYLE_MODELS.keys())
        local_rng = random.Random(f"{self.seed}:{session_id}:{turn_id}")
        return choices[local_rng.randrange(len(choices))]

    def run_architecture_turn(
        self,
        variant: str,
        turn: EvalTurn,
        generated_history: list[tuple[str, str]],
        retrieved_docs: list[dict],
    ) -> GenerationResult:
        user_prompt = self._build_user_prompt(turn)

        api_thread_id = f"eval-{variant}-{turn.benchmark}-{turn.session_id}"
        api_forced_style: str | None = None
        api_variant = variant

        if variant == "B0":
            model_id = self.config.base_model_name
            style = None
            system_prompt = self._build_system_prompt(style)
        elif variant == "B1":
            style = self._deterministic_random_style(turn.session_id, turn.turn_id)
            model_id = STYLE_MODELS[style]
            system_prompt = self._build_system_prompt(style)
            api_forced_style = style
        elif variant == "B2":
            style = "intermediate"
            model_id = self.config.base_model_name
            system_prompt = self._build_system_prompt("scaffolding", level="intermediate")
            if self._lpitutor is None:
                return GenerationResult(
                    response="",
                    model_id=model_id,
                    style=style,
                    retrieval_hits=len(retrieved_docs),
                    error="LPITutor path unavailable",
                )
            try:
                matches = [
                    {
                        "text": doc.get("text", ""),
                        "metadata": doc.get("metadata", {}),
                        "distance": doc.get("distance"),
                    }
                    for doc in retrieved_docs[:5]
                ]
                response = self._lpitutor.generate_response(
                    query=user_prompt,
                    matches=matches,
                    level="intermediate",
                )
                return GenerationResult(
                    response=(response or "").strip(),
                    model_id=model_id,
                    style=style,
                    retrieval_hits=len(retrieved_docs),
                )
            except Exception as error:
                full_history = list(turn.history) + list(generated_history)
                try:
                    fallback_response = self.generator.generate(
                        model_id=model_id,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        history=full_history,
                        thread_id=api_thread_id,
                        forced_style="scaffolding",
                        variant="B2",
                    )
                    return GenerationResult(
                        response=(fallback_response or "").strip(),
                        model_id=model_id,
                        style=style,
                        retrieval_hits=len(retrieved_docs),
                        error=f"LPITutor path failed: {error} (fallback=/api/chat 사용)",
                    )
                except Exception as fallback_error:
                    return GenerationResult(
                        response="",
                        model_id=model_id,
                        style=style,
                        retrieval_hits=len(retrieved_docs),
                        error=f"LPITutor path failed: {error}; fallback failed: {fallback_error}",
                    )
        elif variant == "P":
            style, _ = self.router.route(turn.prompt)
            model_id = STYLE_MODELS.get(style, STYLE_MODELS["direct"])
            system_prompt = self._build_system_prompt(style)
            api_variant = "P"
        else:
            raise ValueError(f"Unsupported architecture variant: {variant}")

        full_history = list(turn.history) + list(generated_history)
        try:
            response = self.generator.generate(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                history=full_history,
                thread_id=api_thread_id,
                forced_style=api_forced_style,
                variant=api_variant,
            )
            return GenerationResult(
                response=response,
                model_id=model_id,
                style=style,
                retrieval_hits=len(retrieved_docs),
            )
        except Exception as error:
            return GenerationResult(
                response="",
                model_id=model_id,
                style=style,
                retrieval_hits=len(retrieved_docs),
                error=str(error),
            )

    def run_adapter_turn(
        self,
        adapter_style: str,
        turn: EvalTurn,
        generated_history: list[tuple[str, str]],
        retrieved_docs: list[dict],
    ) -> GenerationResult:
        if adapter_style not in STYLE_MODELS:
            raise ValueError(f"Unsupported adapter style: {adapter_style}")

        model_id = STYLE_MODELS[adapter_style]
        api_thread_id = f"eval-adapter-{adapter_style}-{turn.benchmark}-{turn.session_id}"
        user_prompt = self._build_user_prompt(turn)
        full_history = list(turn.history) + list(generated_history)

        try:
            response = self.generator.generate(
                model_id=model_id,
                system_prompt=self._build_system_prompt(adapter_style),
                user_prompt=user_prompt,
                history=full_history,
                thread_id=api_thread_id,
                forced_style=adapter_style,
                variant="P",
            )
            return GenerationResult(
                response=response,
                model_id=model_id,
                style=adapter_style,
                retrieval_hits=len(retrieved_docs),
            )
        except Exception as error:
            return GenerationResult(
                response="",
                model_id=model_id,
                style=adapter_style,
                retrieval_hits=len(retrieved_docs),
                error=str(error),
            )


def build_debug_prompt(turn: EvalTurn, generated_history: list[tuple[str, str]]) -> str:
    history_text = summarize_history(list(turn.history) + list(generated_history))
    return (
        f"Session={turn.session_id} Turn={turn.turn_id}\n"
        f"Prompt={turn.prompt}\n"
        f"History={history_text}"
    )
