from __future__ import annotations

import json
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any


TUTORING_CRITERIA: list[dict[str, Any]] = [
    {
        "name": "reference_correctness",
        "description": "Reference answer와의 개념/논리 일치도",
        "weight": 0.45,
    },
    {
        "name": "pedagogical_quality",
        "description": "튜터링 품질(유도, 힌트, 구조화) 및 학습 지원성",
        "weight": 0.25,
    },
    {
        "name": "clarity",
        "description": "설명의 명확성, 단계성, 표현의 이해 용이성",
        "weight": 0.20,
    },
    {
        "name": "safety_hallucination_control",
        "description": "과장/환각/무근거 내용 억제",
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

    def _generate_json(self, prompt: str) -> dict[str, Any]:
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
