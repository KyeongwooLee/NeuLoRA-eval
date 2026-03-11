from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from router_utils import build_router_input
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from router_utils import build_router_input


@dataclass
class StyleTurn:
    style: str
    conversation_id: str
    turn_index: int
    current_query: str
    assistant_response: str
    source: str
    raw_history: list[tuple[str, str]] | None = None


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _remove_label_hints(text: str) -> str:
    raw = str(text or "")
    patterns = [
        r"Respond in Socratic style:[^\n]*",
        r"Provide concise formative feedback with one concrete next step\.?",
        r"Feedback type:[^\n]*",
        r"Tutor goal:[^\n]*",
        r"Solve directly with clear steps\.?",
        r"QuestionId:\s*\S+",
        r"Turn:\s*\d+",
    ]
    for pattern in patterns:
        raw = re.sub(pattern, " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def _extract_current_query(instruction: str) -> str:
    raw = str(instruction or "").strip()
    if "Problem:" in raw:
        raw = raw.split("Problem:", 1)[1]
    if "Student answer:" in raw:
        raw = raw.split("Student answer:", 1)[1]
    if "Student:" in raw:
        raw = raw.split("Student:", 1)[1]
    return _remove_label_hints(raw)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _build_direct(rows: list[dict]) -> list[StyleTurn]:
    turns: list[StyleTurn] = []
    for idx, row in enumerate(rows):
        turns.append(
            StyleTurn(
                style="direct",
                conversation_id=f"direct:{idx}",
                turn_index=0,
                current_query=_extract_current_query(str(row.get("instruction", ""))),
                assistant_response=str(row.get("response", "")).strip(),
                source=str(row.get("source", "")),
            )
        )
    return turns


def _build_socratic(rows: list[dict]) -> list[StyleTurn]:
    turns: list[StyleTurn] = []
    for idx, row in enumerate(rows):
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        problem_id = str(meta.get("problem_id", "unknown_problem")).strip()
        dialogue_id = str(meta.get("dialogue_id", f"row_{idx}")).strip()
        turn_index = _to_int(meta.get("turn_index"), idx)
        turns.append(
            StyleTurn(
                style="socratic",
                conversation_id=f"socratic:{problem_id}:{dialogue_id}",
                turn_index=turn_index,
                current_query=_extract_current_query(str(row.get("instruction", ""))),
                assistant_response=str(row.get("response", "")).strip(),
                source=str(row.get("source", "")),
            )
        )
    return turns


def _build_scaffolding(rows: list[dict]) -> list[StyleTurn]:
    turns: list[StyleTurn] = []
    for idx, row in enumerate(rows):
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        question_id = str(meta.get("question_id_dq", "unknown_question")).strip()
        intervention = str(meta.get("intervention_id", "unknown_intervention")).strip()
        turn_index = _to_int(meta.get("student_msg_seq"), idx)
        turns.append(
            StyleTurn(
                style="scaffolding",
                conversation_id=f"scaffolding:{question_id}:{intervention}",
                turn_index=turn_index,
                current_query=_extract_current_query(str(row.get("instruction", ""))),
                assistant_response=str(row.get("response", "")).strip(),
                source=str(row.get("source", "")),
            )
        )
    return turns


def _parse_feedback_key(sample_id: str, fallback_idx: int) -> tuple[str, int]:
    sample = str(sample_id or "").strip()
    if not sample:
        return f"row_{fallback_idx}", fallback_idx
    if "@" in sample:
        left, right = sample.rsplit("@", 1)
        return left, _to_int(right, fallback_idx)
    return sample, fallback_idx


def _load_raw_single(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else {}


def _extract_pair_history(raw_entry: dict | None) -> list[tuple[str, str]]:
    if not isinstance(raw_entry, dict):
        return []
    history = raw_entry.get("history")
    if not isinstance(history, list):
        return []
    pairs: list[tuple[str, str]] = []
    for item in history:
        if not isinstance(item, list) or len(item) < 2:
            continue
        user_text = str(item[0] or "").strip()
        assistant_text = str(item[1] or "").strip()
        if not user_text and not assistant_text:
            continue
        pairs.append((user_text, assistant_text))
    return pairs


def _build_feedback(rows: list[dict], raw_single_path: Path) -> list[StyleTurn]:
    raw_map = _load_raw_single(raw_single_path)
    turns: list[StyleTurn] = []
    for idx, row in enumerate(rows):
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        sample_id = str(meta.get("sample_id", "")).strip()
        conv_key, turn_index = _parse_feedback_key(sample_id, idx)
        raw_history = _extract_pair_history(raw_map.get(sample_id))
        turns.append(
            StyleTurn(
                style="feedback",
                conversation_id=f"feedback:{conv_key}",
                turn_index=turn_index,
                current_query=_extract_current_query(str(row.get("instruction", ""))),
                assistant_response=str(row.get("response", "")).strip(),
                source=str(row.get("source", "")),
                raw_history=raw_history,
            )
        )
    return turns


def _history_dicts_from_pairs(conversation_id: str, pairs: list[tuple[str, str]]) -> list[dict]:
    history: list[dict] = []
    idx = 0
    for user_text, assistant_text in pairs:
        if user_text:
            history.append(
                {
                    "conversation_id": conversation_id,
                    "turn_index": idx,
                    "role": "user",
                    "content": user_text,
                }
            )
            idx += 1
        if assistant_text:
            history.append(
                {
                    "conversation_id": conversation_id,
                    "turn_index": idx,
                    "role": "assistant",
                    "content": assistant_text,
                }
            )
            idx += 1
    return history


def _serialize_style_turns(turns: list[StyleTurn], max_history_turns: int) -> list[dict]:
    by_conversation: dict[str, list[StyleTurn]] = defaultdict(list)
    for turn in turns:
        by_conversation[turn.conversation_id].append(turn)

    rows: list[dict] = []
    for conversation_id, items in by_conversation.items():
        ordered = sorted(items, key=lambda item: item.turn_index)
        for pos, item in enumerate(ordered):
            if item.raw_history:
                # feedback style can ship richer raw history; prefer it when available.
                history = _history_dicts_from_pairs(conversation_id, item.raw_history)
            else:
                prior_pairs = [(p.current_query, p.assistant_response) for p in ordered[:pos]]
                history = _history_dicts_from_pairs(conversation_id, prior_pairs)

            text = build_router_input(
                current_query=item.current_query,
                history=history,
                conversation_id=conversation_id,
                max_turns=max_history_turns,
            )
            rows.append(
                {
                    "text": text,
                    "label": item.style,
                    "conversation_id": conversation_id,
                    "turn_index": item.turn_index,
                    "source": item.source,
                }
            )
    rows.sort(key=lambda row: (row["conversation_id"], int(row["turn_index"])))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build session-safe multi-turn router training data from Multi-LoRA datasets."
    )
    parser.add_argument(
        "--multi-lora-dir",
        type=Path,
        default=root / "Multi-LoRA",
        help="Path to Multi-LoRA repository root.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "cache" / "router" / "router_train_multiturn.jsonl",
    )
    parser.add_argument("--max-history-turns", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = args.multi_lora_dir / "data" / "processed"
    raw_dir = args.multi_lora_dir / "data" / "raw"

    style_rows = {
        "direct": _read_jsonl(processed_dir / "train_direct.jsonl"),
        "socratic": _read_jsonl(processed_dir / "train_socratic.jsonl"),
        "scaffolding": _read_jsonl(processed_dir / "train_scaffolding.jsonl"),
        "feedback": _read_jsonl(processed_dir / "train_feedback.jsonl"),
    }

    turns: list[StyleTurn] = []
    turns.extend(_build_direct(style_rows["direct"]))
    turns.extend(_build_socratic(style_rows["socratic"]))
    turns.extend(_build_scaffolding(style_rows["scaffolding"]))
    turns.extend(_build_feedback(style_rows["feedback"], raw_dir / "SocraTeach_single.json"))

    rows = _serialize_style_turns(turns, max_history_turns=max(1, args.max_history_turns))
    _write_jsonl(args.output_path, rows)

    stats = defaultdict(int)
    for row in rows:
        stats[row["label"]] += 1

    payload = {
        "output_path": str(args.output_path),
        "num_rows": len(rows),
        "label_counts": dict(stats),
        "max_history_turns": max(1, args.max_history_turns),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
