#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtc.confidence import scores_from_scored_row  # noqa: E402
from dtc.config import THETA_BLACKBOX, whitebox_theta  # noqa: E402
from dtc.io_utils import load_jsonl  # noqa: E402
from dtc.metrics import accuracy_percent, compute_metrics_balanced  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Compute Acc / ECE / AUROC for DTClin and DTCprod")
    p.add_argument("--input", required=True, help="Scored jsonl from score.py")
    p.add_argument("--theta", type=float, default=None, help="Override JS threshold")
    p.add_argument("--metric-seed", type=int, default=0)
    p.add_argument("--generator-name", default=None, help="For whitebox theta if rows lack dtc_theta")
    return p.parse_args()


def row_has_answer(row: Dict[str, Any]) -> bool:
    ans = row.get("pred_answer_extracted")
    if ans is None:
        return False
    return str(ans).strip() != ""


def row_truncated(row: Dict[str, Any]) -> bool:
    if row.get("truncated") is True:
        return True
    return row.get("finish_reason") == "length"


def filter_uq_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept = [r for r in rows if row_has_answer(r) and not row_truncated(r)]
    return kept


def resolve_theta(row: Dict[str, Any], override: Optional[float], generator_name: Optional[str]) -> float:
    if override is not None:
        return float(override)
    if row.get("dtc_theta") is not None:
        return float(row["dtc_theta"])
    setting = row.get("dtc_setting", "whitebox")
    if setting == "blackbox":
        return THETA_BLACKBOX
    if generator_name:
        return whitebox_theta(generator_name)
    path = row.get("instruct_model_path", "")
    return whitebox_theta(str(path))


def eval_estimator(rows: List[Dict[str, Any]], *, estimator: str, theta_override, generator_name):
    probs: List[float] = []
    labels: List[int] = []
    idx = 0 if estimator == "lin" else 1
    for row in rows:
        theta = resolve_theta(row, theta_override, generator_name)
        prob_key = str(row.get("dtc_prob_key", "prob_ref"))
        scored = scores_from_scored_row(row, theta=theta, prob_key=prob_key)
        if scored is None:
            continue
        conf = scored[idx]
        probs.append(conf)
        labels.append(1 if row.get("correct") else 0)
    if not probs:
        return None
    return np.array(probs, dtype=float), np.array(labels, dtype=int)


def main():
    args = parse_args()
    all_rows = load_jsonl(Path(args.input))
    acc_labels = np.array([1 if r.get("correct") else 0 for r in all_rows], dtype=int)
    acc = accuracy_percent(acc_labels)

    uq_rows = filter_uq_rows(all_rows)
    print(f"[info] samples={len(all_rows)} uq_eval={len(uq_rows)} acc={acc:.2f}%", flush=True)

    for name, est in (("DTClin", "lin"), ("DTCprod", "prod")):
        pair = eval_estimator(
            uq_rows,
            estimator=est,
            theta_override=args.theta,
            generator_name=args.generator_name,
        )
        if pair is None:
            print(f"{name}: no valid scores")
            continue
        pv, lv = pair
        metrics = compute_metrics_balanced(
            pv,
            lv,
            seed=args.metric_seed,
            n_bins=20,
            n_passes=5,
            max_num_instances=500,
        )
        print(
            f"{name}: ECE={100 * metrics['ece_mean']:.2f}±{100 * metrics['ece_std']:.2f} "
            f"AUC={100 * metrics['auc_mean']:.2f}±{100 * metrics['auc_std']:.2f} "
            f"(n/class={metrics['n_per_class']})"
        )


if __name__ == "__main__":
    main()
