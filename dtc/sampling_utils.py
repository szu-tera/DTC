from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dtc.config import SYSTEM_PROMPT


def apply_chat(
    tokenizer,
    messages: List[Dict[str, str]],
    enable_thinking: bool = False,
) -> str:
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)
    parts = []
    for m in messages:
        parts.append(f"{m.get('role','user')}: {m.get('content','')}")
    return "\n".join(parts)


def build_stop_words() -> List[str]:
    return [
        "</s>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
        "<end_of_turn>",
        "<turn|>",
        "<eos>",
        "<|return|>",
    ]


def generation_config_eos_ids(model_dir: Optional[Union[str, Path]]) -> List[int]:
    if not model_dir:
        return []
    gc_path = Path(model_dir) / "generation_config.json"
    if not gc_path.is_file():
        return []
    try:
        data = json.loads(gc_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    eos = data.get("eos_token_id")
    if isinstance(eos, int) and eos >= 0:
        return [eos]
    if isinstance(eos, list):
        return [int(x) for x in eos if isinstance(x, int) and x >= 0]
    return []


def resolve_stop_token_ids(tokenizer, model_dir: Optional[Union[str, Path]] = None) -> List[int]:
    ids: set[int] = set()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, int) and eos_id >= 0:
        ids.add(eos_id)
    for d in [model_dir, getattr(tokenizer, "name_or_path", None)]:
        if d:
            for eid in generation_config_eos_ids(d):
                ids.add(eid)
    for tok in build_stop_words():
        try:
            enc = tokenizer(tok, add_special_tokens=False)
            input_ids = enc.get("input_ids", []) or []
        except Exception:
            continue
        if len(input_ids) == 1:
            ids.add(input_ids[0])
    return sorted(ids)


def strip_think_block(text: str) -> str:
    end_tag = "</think>"
    if end_tag in (text or ""):
        return text.split(end_tag, 1)[1].lstrip()
    return text or ""


def default_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
