"""
用于在已有预测结果上计算 pass@k。

输入格式：每行是一个样本的 JSON，对应字段包含：
- 预测："pred"（list）或 "preds"（list）或 "prediction"/"output"（str/list）
- 标准答案：可用字段 ["gt", "gt_ans", "answer", "label", "gold", "reference"]。
    若未提供且传入 --data_name，会调用 parser.parse_ground_truth 从原始样本字段推导。

示例：
        python pass_at_k.py \
                --pred_file /path/to/preds.jsonl \
                --k 1 5 10 \
                --data_name math
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent / "Qwen2.5-Math" / "evaluation"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grader import math_equal  # type: ignore
from parser import parse_ground_truth, extract_answer  # type: ignore


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pick_preds(sample: Dict[str, Any]) -> List[str]:
    for key in ["pred", "preds", "prediction", "output"]:
        if key in sample:
            preds = sample[key]
            if isinstance(preds, str):
                return [preds]
            if isinstance(preds, list):
                return [str(p) for p in preds]
    raise ValueError("No predictions found (expected keys: pred/preds/prediction/output)")


def pick_gt(sample: Dict[str, Any], data_name: str | None) -> str:
    for key in ["gt", "gt_ans", "answer", "label", "gold", "reference", "ground_truth"]:
        if key in sample:
            return str(sample[key])
    if data_name:
        try:
            _, gt_ans = parse_ground_truth(sample, data_name)
            return str(gt_ans)
        except Exception as exc:  # pragma: no cover - best effort fallback
            raise ValueError(f"Cannot derive ground truth for sample idx={sample.get('idx')} ({exc})")
    raise ValueError("No ground truth found (provide --data_name or include gt fields)")


def normalize_pred(pred: str, data_name: str | None) -> str:
    # 与项目 parser 的抽取逻辑保持一致，尽量提取最终答案
    return extract_answer(str(pred), data_name or "math")


def compute_scores(samples: List[Dict[str, Any]], data_name: str | None) -> List[List[bool]]:
    score_rows: List[List[bool]] = []
    for sample in samples:
        preds = pick_preds(sample)
        gt = pick_gt(sample, data_name)
        row: List[bool] = []
        for p in preds:
            norm_p = normalize_pred(p, data_name)
            row.append(bool(math_equal(norm_p, gt)))
        score_rows.append(row)
    return score_rows


def pass_at_k_bootstrap(score_rows: List[List[bool]], k_list: List[int], runs: int, rng: np.random.Generator) -> Dict[int, Dict[str, float]]:
    """对每个 k 做多次重采样求均值和标准差。"""
    res: Dict[int, Dict[str, float]] = {}
    total = len(score_rows)
    if total == 0:
        return {k: {"mean": 0.0, "std": 0.0} for k in k_list}

    score_arr = np.array(score_rows, dtype=object)
    for k in k_list:
        vals = []
        for _ in range(runs):
            idx = rng.choice(total, size=total, replace=True)
            sampled = score_arr[idx]
            hits = sum(1 for row in sampled if any(row[:k]))
            vals.append(hits / total * 100)
        vals = np.array(vals, dtype=float)
        res[k] = {
            "mean": round(float(vals.mean()), 2),
            "std": round(float(vals.std(ddof=0)), 2),
        }
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Path to predictions JSONL")
    parser.add_argument("--k", type=int, nargs="+", default=[1], help="k 值列表，例如 --k 1 5 10")
    parser.add_argument("--data_name", type=str, default=None, help="数据集名；当文件中没有 gt 时用于解析标准答案")
    parser.add_argument("--runs", type=int, default=3, help="重采样次数，默认 3 次")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，可选")
    args = parser.parse_args()

    pred_path = Path(args.pred_file)
    samples = list(load_jsonl(pred_path))
    score_rows = compute_scores(samples, args.data_name)

    rng = np.random.default_rng(args.seed)
    metrics = pass_at_k_bootstrap(score_rows, args.k, args.runs, rng)

    print(f"Loaded samples: {len(samples)} from {pred_path}")
    for k in sorted(metrics):
        m, s = metrics[k]["mean"], metrics[k]["std"]
        print(f"pass@{k}: {m:.2f}% ± {s:.2f}% (runs={args.runs})")


if __name__ == "__main__":
    main()
