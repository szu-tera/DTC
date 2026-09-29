"""Wang et al. 2025 verbalized black-box prompts and JSON parsing (math-adapted)."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from dtc.sampling_utils import strip_think_block

METHOD_VERBALIZED_CONFIDENCE = "verbalized_confidence"
METHOD_VERBALIZED_TOPK = "verbalized_topk"
METHOD_VERBALIZED_DISTRIBUTION = "verbalized_distribution"

METHOD_CHOICES = (
    METHOD_VERBALIZED_CONFIDENCE,
    METHOD_VERBALIZED_TOPK,
    METHOD_VERBALIZED_DISTRIBUTION,
)

METRIC_SCORE = "score"
DEFAULT_TOPK = 2

_MATH_ANSWER_CONSTRAINT = (
    "Your answer must be a mathematical value or short exact answer "
    "(e.g., an integer, fraction, radical, coordinate, or simplified expression), "
    "not a full sentence. If the question specifies a final report form "
    "(e.g., m+n or m-n), use that form."
)
_MATH_EACH_ANSWER_CONSTRAINT = (
    "Each answer must be a mathematical value or short exact answer "
    "(e.g., an integer, fraction, radical, coordinate, or simplified expression), "
    "not a full sentence. If the question specifies a final report form "
    "(e.g., m+n or m-n), use that form."
)

PROMPT_VERBALIZED_CONFIDENCE = """\
Question: {question}
Reason step-by-step to formulate your final answer. {answer_constraint} Then, reason about the confidence in your answer. Conclude by providing a JSON object that states the final answer and your estimated confidence in it:
{{
"final_answer": "Your final answer",
"confidence": "0-1"
}}
"""


def _prompt_verbalized_topk(question: str, k: int) -> str:
    k = max(1, int(k))
    items = []
    ordinal = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
    for i in range(k):
        label = ordinal[i] if i < len(ordinal) else f"{i + 1}-th"
        items.append(
            '{\n'
            f'"candidate": "{label} most likely answer",\n'
            '"confidence": "0-1"\n'
            "}"
        )
    array_body = ",\n".join(items)
    return (
        f"Question: {question}\n"
        f"Reason step-by-step to formulate {k} best guesses and probability that each is correct. "
        f"{_MATH_EACH_ANSWER_CONSTRAINT} Your final output must be a JSON array:\n"
        f"[\n{array_body}\n]\n"
    )


PROMPT_VERBALIZED_DISTRIBUTION = """\
Question: {question}
Reason step-by-step to formulate your answer. You may propose multiple possible answers (fewer than five). {each_answer_constraint} Always include "None of the above" as a possible answer. Reason about the confidence in each possible answer. Your final output must be a JSON array where the confidence scores form a probability distribution (they must sum to 1.0):
[
{{
"candidate": "Candidate 1",
"confidence": "0-1"
}},
{{
"candidate": "Candidate 2",
"confidence": "0-1"
}},
...
{{
"candidate": "None of the above",
"confidence": "0-1"
}}
]
"""

_NONE_OF_THE_ABOVE = re.compile(r"^\s*none\s+of\s+the\s+above\s*$", re.IGNORECASE)


def build_verbalized_user_prompt(method: str, question: str, *, top_k: int = DEFAULT_TOPK) -> str:
    q = question.strip()
    if method == METHOD_VERBALIZED_CONFIDENCE:
        return PROMPT_VERBALIZED_CONFIDENCE.format(
            question=q, answer_constraint=_MATH_ANSWER_CONSTRAINT
        )
    if method == METHOD_VERBALIZED_TOPK:
        return _prompt_verbalized_topk(q, top_k)
    if method == METHOD_VERBALIZED_DISTRIBUTION:
        return PROMPT_VERBALIZED_DISTRIBUTION.format(
            question=q, each_answer_constraint=_MATH_EACH_ANSWER_CONSTRAINT
        )
    raise ValueError(f"未知 verbalized method: {method}")


def _parse_confidence_value(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        v = float(raw)
    else:
        s = str(raw).strip().rstrip("%")
        if not s:
            return None
        try:
            v = float(s)
        except ValueError:
            return None
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    if not 0.0 <= v <= 1.0:
        return None
    return v


def _candidate_text(item: Dict[str, Any]) -> Optional[str]:
    for key in ("final_answer", "candidate", "option", "answer"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if val is not None and not isinstance(val, (dict, list)):
            s = str(val).strip()
            if s:
                return s
    return None


def _scan_json_values(text: str) -> List[Any]:
    decoder = json.JSONDecoder()
    found: List[Any] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, (dict, list)):
            found.append(obj)
            i += end
            continue
        i += 1
    return found


def _extract_json_payload(text: str) -> Any:
    cleaned = strip_think_block(text or "").strip()
    if not cleaned:
        raise ValueError("空生成文本")

    cands = _scan_json_values(cleaned)
    if cands:
        return cands[-1]

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, (dict, list)):
            return obj
    except json.JSONDecodeError:
        pass

    fences = list(re.finditer(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE))
    for m in reversed(fences):
        block = m.group(1).strip()
        try:
            obj = json.loads(block)
            if isinstance(obj, (dict, list)):
                return obj
        except json.JSONDecodeError:
            pass
        block_cands = _scan_json_values(block)
        if block_cands:
            return block_cands[-1]

    raise ValueError("未找到可解析 JSON")


def selected_answer_as_pred(answer: str) -> str:
    return strip_think_block(answer or "").strip()


def _pick_best_candidate(
    items: List[Dict[str, Any]],
    *,
    allow_none_of_the_above: bool = True,
) -> Tuple[Optional[str], Optional[float], List[Dict[str, Any]]]:
    parsed: List[Dict[str, Any]] = []
    best_ans: Optional[str] = None
    best_conf: Optional[float] = None
    for item in items:
        if not isinstance(item, dict):
            continue
        ans = _candidate_text(item)
        conf = _parse_confidence_value(item.get("confidence"))
        entry = {"candidate": ans, "confidence": conf}
        parsed.append(entry)
        if ans is None or conf is None:
            continue
        if not allow_none_of_the_above and _NONE_OF_THE_ABOVE.match(ans):
            continue
        if best_conf is None or conf > best_conf:
            best_conf = conf
            best_ans = ans
    return best_ans, best_conf, parsed


def parse_verbalized_output(
    method: str,
    response_text: str,
) -> Dict[str, Any]:
    payload = _extract_json_payload(response_text)
    out: Dict[str, Any] = {"raw_json": payload}

    if method == METHOD_VERBALIZED_CONFIDENCE:
        if not isinstance(payload, dict):
            raise ValueError("verbalized_confidence 期望 JSON object")
        ans = _candidate_text(payload) or (
            str(payload["final_answer"]).strip() if payload.get("final_answer") is not None else None
        )
        score = _parse_confidence_value(payload.get("confidence"))
        if ans is None or score is None:
            raise ValueError("verbalized_confidence 缺少 final_answer/confidence")
        out["answer"] = ans
        out[METRIC_SCORE] = score
        return out

    if method in (METHOD_VERBALIZED_TOPK, METHOD_VERBALIZED_DISTRIBUTION):
        if not isinstance(payload, list):
            raise ValueError(f"{method} 期望 JSON array")
        ans, score, cands = _pick_best_candidate(payload, allow_none_of_the_above=True)
        if method == METHOD_VERBALIZED_DISTRIBUTION and ans is not None and _NONE_OF_THE_ABOVE.match(ans):
            ans2, score2, _ = _pick_best_candidate(payload, allow_none_of_the_above=False)
            if ans2 is not None:
                ans, score = ans2, score2
        if ans is None or score is None:
            raise ValueError(f"{method} 无法从候选中选出有效答案/置信度")
        out["answer"] = ans
        out[METRIC_SCORE] = score
        out["candidates"] = cands
        return out

    raise ValueError(f"未知 verbalized method: {method}")
