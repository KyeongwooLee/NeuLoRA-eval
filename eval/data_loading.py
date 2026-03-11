from __future__ import annotations

import ast
import csv
import json
import random
import urllib.request
from pathlib import Path
from typing import Iterable

try:
    from .common import BenchmarkData, EvalConfig, EvalTurn
except ImportError:
    from common import BenchmarkData, EvalConfig, EvalTurn


MATHDIAL_TRAIN_URL = "https://raw.githubusercontent.com/eth-nlped/mathdial/main/data/train.jsonl"
MATHDIAL_TEST_URL = "https://raw.githubusercontent.com/eth-nlped/mathdial/main/data/test.jsonl"
MT_BENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
COMTA_CSV_URL = "https://raw.githubusercontent.com/umass-ml4ed/dialogue-kt/main/data/annotated/comta_atc.csv"
COMTA_LICENSE_URL = "https://raw.githubusercontent.com/umass-ml4ed/dialogue-kt/main/data/annotated/COMTA_LICENSE.txt"


def _download_if_missing(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _split_rows(rows: list[dict], seed: int, train_ratio: float = 0.8) -> tuple[list[dict], list[dict]]:
    indexed = list(enumerate(rows))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    train_size = int(len(indexed) * train_ratio)
    train_idx = {idx for idx, _ in indexed[:train_size]}
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for idx, row in enumerate(rows):
        if idx in train_idx:
            train_rows.append(row)
        else:
            eval_rows.append(row)
    return train_rows, eval_rows


def _parse_mathdial_conversation(raw: str) -> list[tuple[str, str]]:
    parts = [chunk.strip() for chunk in (raw or "").split("|EOM|") if chunk.strip()]
    messages: list[tuple[str, str]] = []
    for part in parts:
        if ":" not in part:
            continue
        role_text, content = part.split(":", 1)
        role_raw = role_text.strip().lower()
        if role_raw.startswith("teacher"):
            role = "teacher"
        elif role_raw.startswith("student"):
            role = "student"
        else:
            continue
        messages.append((role, content.strip()))
    return messages


def _parse_comta_dialogue(raw: str) -> list[tuple[str, str]]:
    try:
        payload = ast.literal_eval(raw)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []

    messages: list[tuple[str, str]] = []
    for turn in payload:
        if not isinstance(turn, dict):
            continue
        teacher = str(turn.get("teacher", "") or "").strip()
        student = str(turn.get("student", "") or "").strip()
        if teacher:
            messages.append(("teacher", teacher))
        if student:
            messages.append(("student", student))
    return messages


def _student_to_teacher_turns(
    *,
    benchmark: str,
    session_id: str,
    messages: list[tuple[str, str]],
    metadata: dict,
) -> list[EvalTurn]:
    turns: list[EvalTurn] = []
    running_history: list[tuple[str, str]] = []
    for idx, (role, content) in enumerate(messages):
        if role != "student":
            running_history.append((role, content))
            continue

        reference = None
        for future_role, future_content in messages[idx + 1 :]:
            if future_role == "teacher":
                reference = future_content
                break

        turns.append(
            EvalTurn(
                benchmark=benchmark,
                session_id=session_id,
                turn_id=len(turns) + 1,
                prompt=content,
                reference=reference,
                history=list(running_history),
                metadata=dict(metadata),
            )
        )
        running_history.append((role, content))
    return turns


def _iter_train_docs_from_messages(
    benchmark: str,
    session_id: str,
    messages: list[tuple[str, str]],
    extra: dict,
) -> Iterable[dict]:
    text_lines = []
    for role, content in messages:
        text_lines.append(f"{role.capitalize()}: {content}")
    joined = "\n".join(text_lines)
    yield {
        "id": f"{benchmark}:{session_id}",
        "text": joined,
        "metadata": {"benchmark": benchmark, "session_id": session_id, **extra},
    }


def load_mathdial(config: EvalConfig) -> BenchmarkData:
    train_path = _download_if_missing(MATHDIAL_TRAIN_URL, config.data_cache_dir / "mathdial_train.jsonl")
    test_path = _download_if_missing(MATHDIAL_TEST_URL, config.data_cache_dir / "mathdial_test.jsonl")

    train_rows = _load_jsonl(train_path)
    test_rows = _load_jsonl(test_path)

    train_docs: list[dict] = []
    for row_idx, row in enumerate(train_rows):
        qid = str(row.get("qid", "unknown"))
        session_uid = f"{qid}_{row_idx}"
        messages = _parse_mathdial_conversation(str(row.get("conversation", "")))
        if not messages:
            continue
        train_docs.extend(
            _iter_train_docs_from_messages(
                "mathdial",
                session_id=session_uid,
                messages=messages,
                extra={
                    "qid": qid,
                    "row_index": row_idx,
                    "question": str(row.get("question", "")),
                    "ground_truth": str(row.get("ground_truth", "")),
                },
            )
        )

    eval_turns: list[EvalTurn] = []
    for row_idx, row in enumerate(test_rows):
        qid = str(row.get("qid", "unknown"))
        messages = _parse_mathdial_conversation(str(row.get("conversation", "")))
        if not messages:
            continue
        session_uid = f"mathdial_{qid}_{row_idx}"
        turns = _student_to_teacher_turns(
            benchmark="mathdial",
            session_id=session_uid,
            messages=messages,
            metadata={
                "qid": qid,
                "row_index": row_idx,
                "question": str(row.get("question", "")),
                "ground_truth": str(row.get("ground_truth", "")),
                "student_profile": str(row.get("student_profile", "")),
            },
        )
        eval_turns.extend(turns)

    return BenchmarkData(benchmark="mathdial", train_docs=train_docs, eval_turns=eval_turns)


def load_mtbench(config: EvalConfig) -> BenchmarkData:
    mt_path = _download_if_missing(MT_BENCH_URL, config.data_cache_dir / "mtbench_question.jsonl")
    rows = _load_jsonl(mt_path)

    eval_turns: list[EvalTurn] = []
    for row in rows:
        question_id = str(row.get("question_id", "unknown"))
        turns = row.get("turns") or []
        category = str(row.get("category", "unknown"))
        history: list[tuple[str, str]] = []
        for idx, user_turn in enumerate(turns, start=1):
            user_text = str(user_turn)
            eval_turns.append(
                EvalTurn(
                    benchmark="mtbench",
                    session_id=f"mtbench_{question_id}",
                    turn_id=idx,
                    prompt=user_text,
                    reference=None,
                    history=list(history),
                    metadata={"question_id": question_id, "category": category},
                )
            )
            history.append(("student", user_text))

    return BenchmarkData(benchmark="mtbench", train_docs=[], eval_turns=eval_turns)


def load_comta(config: EvalConfig) -> BenchmarkData:
    comta_path = _download_if_missing(COMTA_CSV_URL, config.data_cache_dir / "comta_atc.csv")
    license_path = _download_if_missing(COMTA_LICENSE_URL, config.data_cache_dir / "COMTA_LICENSE.txt")

    with comta_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    train_rows, eval_rows = _split_rows(rows, seed=config.seed, train_ratio=0.8)

    train_docs: list[dict] = []
    for row in train_rows:
        index_id = str(row.get("index", "unknown"))
        messages = _parse_comta_dialogue(str(row.get("dialogue", "")))
        if not messages:
            continue
        train_docs.extend(
            _iter_train_docs_from_messages(
                "comta",
                session_id=index_id,
                messages=messages,
                extra={"meta_data": str(row.get("meta_data", ""))},
            )
        )

    eval_turns: list[EvalTurn] = []
    for row in eval_rows:
        index_id = str(row.get("index", "unknown"))
        messages = _parse_comta_dialogue(str(row.get("dialogue", "")))
        if not messages:
            continue
        eval_turns.extend(
            _student_to_teacher_turns(
                benchmark="comta",
                session_id=f"comta_{index_id}",
                messages=messages,
                metadata={
                    "index": index_id,
                    "meta_data": str(row.get("meta_data", "")),
                },
            )
        )

    license_head = ""
    try:
        license_head = "\n".join(license_path.read_text(encoding="utf-8").splitlines()[:6])
    except Exception:
        license_head = "COMTA license file available at cache path"

    return BenchmarkData(
        benchmark="comta",
        train_docs=train_docs,
        eval_turns=eval_turns,
        notes=[
            "CoMTA data uses evaluation-only restrictions from COMTA_LICENSE.txt",
            license_head,
        ],
    )


def load_gsm8k(config: EvalConfig) -> BenchmarkData:
    try:
        from datasets import load_dataset
    except Exception as error:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "datasets package is required for GSM8K. Install with `pip install datasets`."
        ) from error

    dataset = load_dataset("openai/gsm8k", "main")
    train_rows = [dict(row) for row in dataset["train"]]
    test_rows = [dict(row) for row in dataset["test"]]

    train_docs: list[dict] = []
    for idx, row in enumerate(train_rows):
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        train_docs.append(
            {
                "id": f"gsm8k_train_{idx}",
                "text": f"Question: {question}\nReference: {answer}",
                "metadata": {"benchmark": "gsm8k", "split": "train", "index": idx},
            }
        )

    eval_turns: list[EvalTurn] = []
    for idx, row in enumerate(test_rows):
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        eval_turns.append(
            EvalTurn(
                benchmark="gsm8k",
                session_id=f"gsm8k_{idx}",
                turn_id=1,
                prompt=question,
                reference=answer,
                history=[],
                metadata={"index": idx},
            )
        )

    return BenchmarkData(benchmark="gsm8k", train_docs=train_docs, eval_turns=eval_turns)


def load_benchmark(benchmark: str, config: EvalConfig) -> BenchmarkData:
    if benchmark == "comta":
        return load_comta(config)
    if benchmark == "mathdial":
        return load_mathdial(config)
    if benchmark == "mtbench":
        return load_mtbench(config)
    if benchmark == "gsm8k":
        return load_gsm8k(config)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def sample_turns(turns: list[EvalTurn], max_samples: int, seed: int) -> list[EvalTurn]:
    if len(turns) <= max_samples:
        return turns
    rng = random.Random(seed)
    idxs = list(range(len(turns)))
    rng.shuffle(idxs)
    selected = sorted(idxs[:max_samples])
    return [turns[i] for i in selected]
