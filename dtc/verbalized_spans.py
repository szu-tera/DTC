"""Map verbalized JSON answer strings to completion token indices."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from dtc.sampling_utils import strip_think_block
from dtc.verbalized import (
    METHOD_VERBALIZED_CONFIDENCE,
    METHOD_VERBALIZED_DISTRIBUTION,
    METHOD_VERBALIZED_TOPK,
    METRIC_SCORE,
    parse_verbalized_output,
)


def last_top_level_json_span(text: str) -> Optional[Tuple[int, int, Any]]:
    cleaned = text or ""
    decoder = json.JSONDecoder()
    last: Optional[Tuple[int, int, Any]] = None
    i = 0
    n = len(cleaned)
    while i < n:
        ch = cleaned[i]
        if ch not in "{[":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(cleaned[i:])
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, (dict, list)):
            last = (i, i + end, obj)
            i += end
            continue
        i += 1
    return last


def _json_string_literal_inner_span(json_text: str, value: str) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    literal = json.dumps(value, ensure_ascii=False)
    if len(literal) < 2 or literal[0] != '"' or literal[-1] != '"':
        return None
    idx = json_text.rfind(literal)
    if idx < 0:
        needle = f'"{value}"'
        idx = json_text.rfind(needle)
        if idx < 0:
            return None
        return (idx + 1, idx + 1 + len(value))
    return (idx + 1, idx + len(literal) - 1)


def _resolve_selected_answer(
    text: str,
    *,
    method: Optional[str],
    selected_answer: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if selected_answer is not None and str(selected_answer).strip():
        return method, str(selected_answer).strip()
    if not method:
        return None, None
    try:
        parsed = parse_verbalized_output(method, text)
    except Exception:
        return method, None
    ans = parsed.get("answer")
    if ans is None:
        return method, None
    return method, str(ans).strip()


def char_spans_verbalized_reason_ans(
    text: str,
    *,
    method: Optional[str] = None,
    selected_answer: Optional[str] = None,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    cleaned = strip_think_block(text or "")
    js = last_top_level_json_span(cleaned)
    if js is None:
        return None, None
    json_start, json_end, _obj = js
    reason_span = (0, json_start) if json_start > 0 else None

    _m, answer = _resolve_selected_answer(
        cleaned, method=method, selected_answer=selected_answer
    )
    ans_span = None
    if answer:
        rel = _json_string_literal_inner_span(cleaned[json_start:json_end], answer)
        if rel is not None:
            ans_span = (json_start + rel[0], json_start + rel[1])
    return reason_span, ans_span


def _token_indices_overlapping_span(
    offsets: List[Tuple[int, int]],
    span: Optional[Tuple[int, int]],
) -> List[int]:
    if span is None or not offsets:
        return []
    cs, ce = span
    if ce <= cs:
        return []
    pos: List[int] = []
    for i, (a, b) in enumerate(offsets):
        if b <= cs or a >= ce:
            continue
        pos.append(i)
    return pos


def _token_indices_strictly_before(
    offsets: List[Tuple[int, int]],
    cut: int,
) -> List[int]:
    pos: List[int] = []
    for i, (a, b) in enumerate(offsets):
        if b <= cut and b > a:
            pos.append(i)
    return pos


def verbalized_reason_ans_token_indices(
    tokenizer,
    completion_text: str,
    *,
    method: Optional[str] = None,
    selected_answer: Optional[str] = None,
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    text = strip_think_block(completion_text or "")
    if not text.strip():
        return None, None

    reason_span, ans_span = char_spans_verbalized_reason_ans(
        text, method=method, selected_answer=selected_answer
    )
    if reason_span is None and ans_span is None:
        return None, None

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = enc.get("offset_mapping")
    if offsets is None:
        return None, None
    offsets_list = [(int(a), int(b)) for a, b in offsets]

    reason_idx: Optional[List[int]] = None
    ans_idx: Optional[List[int]] = None
    if reason_span is not None:
        reason_idx = _token_indices_strictly_before(offsets_list, reason_span[1])
    if ans_span is not None:
        ans_idx = _token_indices_overlapping_span(offsets_list, ans_span)

    return (reason_idx if reason_idx else None), (ans_idx if ans_idx else None)


def resolve_verbalized_selected_answer_from_row(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    method = row.get("verbalized_method")
    if not isinstance(method, str) or not method.strip():
        return None, None
    method = method.strip()
    conf = row.get("confidence")
    if isinstance(conf, dict):
        block = conf.get(method)
        if isinstance(block, dict):
            ans = block.get("answer")
            if ans is not None and str(ans).strip():
                return method, str(ans).strip()
            cands = block.get("candidates")
            if isinstance(cands, list) and method in (
                METHOD_VERBALIZED_TOPK,
                METHOD_VERBALIZED_DISTRIBUTION,
            ):
                best_a, best_c = None, None
                for c in cands:
                    if not isinstance(c, dict):
                        continue
                    a = c.get("candidate")
                    sc = c.get("confidence")
                    if a is None or sc is None:
                        continue
                    try:
                        scf = float(sc)
                    except (TypeError, ValueError):
                        continue
                    if best_c is None or scf > best_c:
                        best_c, best_a = scf, str(a).strip()
                if best_a:
                    return method, best_a
            if method == METHOD_VERBALIZED_CONFIDENCE and METRIC_SCORE in block:
                pass
    return method, None
