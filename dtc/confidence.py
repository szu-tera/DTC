from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np

from dtc.config import CLIN_A, CLIN_B, CLIN_N, PROD_K


def count_divergent_reasoning_tokens(
    per_token: List[Dict[str, Any]],
    *,
    theta: float,
    ans_indices: Set[int],
) -> int:
    m = 0
    for tok in per_token:
        if not isinstance(tok, dict):
            continue
        idx = tok.get("index")
        js = tok.get("js_divergence")
        if idx is None or js is None:
            continue
        if int(idx) in ans_indices:
            continue
        if float(js) > theta:
            m += 1
    return m


def mean_prob_on_trajectory(
    per_token: List[Dict[str, Any]],
    *,
    prob_key: str,
) -> Optional[float]:
    vals: List[float] = []
    for tok in per_token:
        if not isinstance(tok, dict):
            continue
        p = tok.get(prob_key)
        if p is None:
            continue
        pf = float(p)
        if np.isfinite(pf):
            vals.append(pf)
    if not vals:
        return None
    return float(np.mean(vals))


def dtc_lin(m: int, *, n: int = CLIN_N, a: float = CLIN_A, b: float = CLIN_B) -> float:
    ratio = min(float(m) / float(n), 1.0)
    return float(a - (a - b) * ratio)


def dtc_prod(c_mean: float, m: int, *, k: int = PROD_K) -> float:
    n = int(m + k)
    if c_mean <= 0.0:
        return 0.0
    if c_mean >= 1.0:
        return 1.0
    return float(np.exp(n * np.log(c_mean)))


def scores_from_scored_row(
    row: Dict[str, Any],
    *,
    theta: float,
    prob_key: str,
) -> Optional[tuple[float, float]]:
    block = row.get("dtc")
    if not isinstance(block, dict):
        return None
    per_token = block.get("per_token")
    if not isinstance(per_token, list) or not per_token:
        return None
    raw_idx = block.get("boxed_token_indices")
    ans_indices: Set[int] = {int(i) for i in raw_idx} if isinstance(raw_idx, list) else set()
    m = count_divergent_reasoning_tokens(per_token, theta=theta, ans_indices=ans_indices)
    c_mean = mean_prob_on_trajectory(per_token, prob_key=prob_key)
    if c_mean is None:
        return None
    return dtc_lin(m), dtc_prod(c_mean, m)
