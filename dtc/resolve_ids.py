from __future__ import annotations

import json
from typing import Dict, List, Tuple


def prompt_text_to_ids(tok, prompt_text: str) -> List[int]:
    text = prompt_text.strip()
    if text.startswith("["):
        try:
            messages = json.loads(text)
        except Exception:
            messages = None
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            from dtc.sampling_utils import apply_chat

            rendered = apply_chat(tok, messages, enable_thinking=False)
            return list(tok(rendered, add_special_tokens=False)["input_ids"])
    return list(tok(prompt_text, add_special_tokens=False)["input_ids"])


def resolve_ids(row: Dict, tok) -> Tuple[List[int], List[int]]:
    prompt_ids = row.get("prompt_token_ids")
    completion_ids = row.get("completion_token_ids")
    if (
        isinstance(prompt_ids, list)
        and isinstance(completion_ids, list)
        and prompt_ids
        and completion_ids
    ):
        return list(prompt_ids), list(completion_ids)

    prompt_text = row.get("prompt_text")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("Row missing prompt_text or token ids")
    completion_text = row.get("generate", "")
    if not isinstance(completion_text, str):
        completion_text = str(completion_text)
    p_ids = prompt_text_to_ids(tok, prompt_text)
    c_ids = list(tok(completion_text, add_special_tokens=False)["input_ids"])
    if not c_ids:
        raise ValueError("Empty completion after retokenize")
    return p_ids, c_ids
