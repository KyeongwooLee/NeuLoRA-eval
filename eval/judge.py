from __future__ import annotations

import json
import random
import re
import threading
import time
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from statistics import mean
from typing import Any


TUTORING_CRITERIA: list[dict[str, Any]] = [
    {
        "name": "reference_correctness",
        "description": "Conceptual/logical alignment with the reference answer",
        "weight": 0.45,
    },
    {
        "name": "pedagogical_quality",
        "description": "Tutoring quality (guidance, hints, structure) and learning support",
        "weight": 0.25,
    },
    {
        "name": "clarity",
        "description": "Clarity of explanation, stepwise progression, and ease of understanding",
        "weight": 0.20,
    },
    {
        "name": "safety_hallucination_control",
        "description": "Suppression of exaggeration, hallucination, and unsupported claims",
        "weight": 0.10,
    },
]


@dataclass
class JudgeResult:
    overall_score: float
    payload: dict[str, Any]


class GeminiJudge:
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise RuntimeError("GENAI_API_KEY is required for Gemini judge.")

        from google import genai
        from google.genai import types

        self.model_name = model_name
        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._min_interval_sec = 0.5
        self._rate_lock = threading.Lock()
        self._next_allowed_at = 0.0
        self._max_retries = 6
        self._backoff_base_sec = 1.0
        self._backoff_cap_sec = 30.0
        self._jitter_max_sec = 0.3

    @staticmethod
    def _json_loads_safe(text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _wait_rate_limit_slot(self) -> None:
        sleep_sec = 0.0
        with self._rate_lock:
            now = time.monotonic()
            if now < self._next_allowed_at:
                sleep_sec = self._next_allowed_at - now
            slot_start = max(now, self._next_allowed_at)
            self._next_allowed_at = slot_start + self._min_interval_sec
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        for attr in ("status_code", "code", "status"):
            value = getattr(exc, attr, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)

        response = getattr(exc, "response", None)
        if response is not None:
            for attr in ("status_code", "status", "code"):
                value = getattr(response, attr, None)
                if isinstance(value, int):
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)
        return None

    @staticmethod
    def _extract_retry_after_seconds(exc: Exception) -> float | None:
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except Exception:
                pass

        headers = getattr(exc, "headers", None)
        if headers is None:
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", None) if response is not None else None
        if not headers:
            return None

        raw = None
        if hasattr(headers, "get"):
            raw = headers.get("Retry-After") or headers.get("retry-after")
        elif isinstance(headers, dict):
            raw = headers.get("Retry-After") or headers.get("retry-after")

        if not raw:
            return None

        try:
            return max(0.0, float(raw))
        except Exception:
            pass

        try:
            dt = parsedate_to_datetime(str(raw))
            if dt is None:
                return None
            now = time.time()
            return max(0.0, dt.timestamp() - now)
        except Exception:
            return None

    def _is_retryable(self, exc: Exception) -> bool:
        code = self._extract_status_code(exc)
        if code in {429, 500, 502, 503, 504}:
            return True

        msg = str(exc).lower()
        retry_signals = (
            "too many requests",
            "rate limit",
            "resource has been exhausted",
            "temporarily unavailable",
            "service unavailable",
            "deadline exceeded",
            "timed out",
            "timeout",
            "connection reset",
        )
        return any(signal in msg for signal in retry_signals)

    def _generate_json(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._wait_rate_limit_slot()
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                        thinking_config=self._types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                return self._json_loads_safe(response.text or "{}")
            except Exception as exc:
                last_error = exc
                if not self._is_retryable(exc) or attempt >= self._max_retries:
                    raise

                retry_after = self._extract_retry_after_seconds(exc) or 0.0
                backoff = min(self._backoff_cap_sec, self._backoff_base_sec * (2 ** attempt))
                jitter = random.uniform(0.0, self._jitter_max_sec)
                time.sleep(max(retry_after, backoff + jitter))

        if last_error is not None:
            raise last_error
        return {}

    def score_tutoring_turn(
        self,
        *,
        benchmark: str,
        question: str,
        reference_answer: str,
        model_answer: str,
    ) -> JudgeResult:
        prompt = f"""
You are an evaluator for a tutoring LLM.
Benchmark: {benchmark}

Question:
{question}

Reference tutor answer:
{reference_answer}

Model tutor answer:
{model_answer}

Score each criterion from 1 to 5 and return JSON only.
Criteria:
{json.dumps(TUTORING_CRITERIA, ensure_ascii=False)}

Output schema:
{{
  "criterion_scores": [
    {{"name": "reference_correctness", "score": 1, "reason": "..."}},
    {{"name": "pedagogical_quality", "score": 1, "reason": "..."}},
    {{"name": "clarity", "score": 1, "reason": "..."}},
    {{"name": "safety_hallucination_control", "score": 1, "reason": "..."}}
  ],
  "major_issues": ["..."],
  "improvement_tip": "..."
}}
""".strip()

        payload = self._generate_json(prompt)
        score_map = {
            str(item.get("name")): item
            for item in (payload.get("criterion_scores") or [])
            if isinstance(item, dict)
        }

        normalized: list[dict[str, Any]] = []
        weighted = 0.0
        for criterion in TUTORING_CRITERIA:
            name = criterion["name"]
            item = score_map.get(name, {})
            try:
                score = float(item.get("score", 1))
            except Exception:
                score = 1.0
            score = max(1.0, min(5.0, score))
            weighted += criterion["weight"] * score
            normalized.append(
                {
                    "name": name,
                    "score": score,
                    "reason": str(item.get("reason", "")).strip() or "No reason provided.",
                }
            )

        overall = (weighted / 5.0) * 100.0
        payload["criterion_scores"] = normalized
        payload["overall_score"] = round(overall, 3)
        return JudgeResult(overall_score=round(overall, 3), payload=payload)

    def score_mtbench_turn(
        self,
        *,
        question: str,
        model_answer: str,
        conversation_history: str,
    ) -> JudgeResult:
        prompt = f"""
You are evaluating one MT-Bench turn.

Conversation history (may be empty):
{conversation_history}

Current user turn:
{question}

Model answer:
{model_answer}

Give one score from 1 to 10 for helpfulness, relevance, coherence, and instruction-following.
Return JSON only:
{{"score": 1, "reason": "short reason"}}
""".strip()

        payload = self._generate_json(prompt)
        try:
            score = float(payload.get("score", 1))
        except Exception:
            score = 1.0
        score = max(1.0, min(10.0, score))
        payload["score"] = score
        payload.setdefault("reason", "")
        return JudgeResult(overall_score=score, payload=payload)


def extract_last_number(text: str) -> str | None:
    if not text:
        return None
    matches = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if not matches:
        return None
    value = matches[-1]
    try:
        numeric = float(value)
    except ValueError:
        return value
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def gsm8k_exact_match(reference: str, prediction: str) -> tuple[bool, str | None, str | None]:
    ref = extract_last_number(reference or "")
    pred = extract_last_number(prediction or "")
    return bool(ref and pred and ref == pred), ref, pred


def mean_or_zero(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0
