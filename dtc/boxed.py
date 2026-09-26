from __future__ import annotations

from typing import List, Optional, Tuple

import regex


def char_span_of_boxed(text: str) -> Optional[Tuple[int, int]]:
    pattern = regex.compile(r"\\boxed\{((?:[^{}]++|\{(?1)\})++)\}")
    last = None
    for m in pattern.finditer(text):
        last = m.span(1)
    return last


def boxed_answer_positions_in_completion(tokenizer, completion_text: str) -> Optional[List[int]]:
    span = char_span_of_boxed(completion_text)
    if span is None:
        return None
    span_cs, span_ce = span
    enc = tokenizer(
        completion_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = enc.get("offset_mapping")
    if offsets is None:
        return None
    pos: List[int] = []
    for i, (a, b) in enumerate(offsets):
        if b <= span_cs or a >= span_ce:
            continue
        pos.append(i)
    return pos if pos else None
