from __future__ import annotations

from typing import Any


_ROLE_MAP: dict[str, str] = {
    "user": "User",
    "student": "User",
    "human": "User",
    "assistant": "Assistant",
    "teacher": "Assistant",
    "tutor": "Assistant",
    "ai": "Assistant",
}


def _normalize_role(value: Any) -> str | None:
    key = str(value or "").strip().lower()
    return _ROLE_MAP.get(key)


def build_router_input(
    current_query: str,
    history: list[dict] | None,
    conversation_id: str | None,
    max_turns: int = 6,
    logger=None,
) -> str:
    conv_id = str(conversation_id or "").strip()
    cleaned_query = str(current_query or "").strip()
    if not cleaned_query:
        cleaned_query = "(empty)"

    usable: list[dict] = []
    source = list(history or [])
    if source:
        for row in source:
            row_conv = str(row.get("conversation_id", "")).strip()
            if conv_id and row_conv and row_conv != conv_id:
                if logger:
                    logger.warning(
                        "router_input: dropped cross-conversation history row "
                        f"(expected={conv_id}, got={row_conv})"
                    )
                continue
            usable.append(row)

    if any("turn_index" not in item for item in usable):
        if logger:
            logger.warning("router_input: history dropped because turn_index is missing")
        usable = []

    usable.sort(key=lambda item: int(item.get("turn_index", 0)))
    max_pairs = max(1, int(max_turns))
    sliced = usable[-(max_pairs * 2) :]

    lines: list[str] = []
    for item in sliced:
        role = _normalize_role(item.get("role"))
        if role is None:
            continue
        text = str(item.get("content", "")).strip()
        if not text:
            continue
        lines.append(f"{role}: {text}")

    history_block = "\n".join(lines) if lines else "(none)"
    return f"History:\n{history_block}\n\nCurrent user query:\n{cleaned_query}"
