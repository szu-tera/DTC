from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dtc.config import EVAL_ROOT


def ensure_eval_imports() -> None:
    root = str(EVAL_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_dataset_examples(
    data_name: str,
    *,
    split: str = "test",
    limit: int = 0,
) -> List[Dict[str, Any]]:
    ensure_eval_imports()
    from data_loader import load_data  # type: ignore

    data_dir = str(EVAL_ROOT / "data")
    examples = load_data(data_name, split, data_dir)
    if limit and limit > 0:
        examples = examples[:limit]
    return examples


def grade_rows(rows: List[Dict[str, Any]], data_name: str) -> None:
    ensure_eval_imports()
    from parser import parse_ground_truth  # type: ignore

    from concurrent.futures import TimeoutError

    from pebble import ProcessPool
    from tqdm import tqdm

    for sample in rows:
        sample["gt_cot"], sample["gt"] = parse_ground_truth(sample, data_name)

    from grader import math_equal_process  # type: ignore

    params = [
        (idx, sample["pred_answer_extracted"], sample["gt"])
        for idx, sample in enumerate(rows)
    ]
    results: List[bool] = []
    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=10)
        it = future.result()
        with tqdm(total=len(rows), desc="Math grade") as bar:
            for _ in rows:
                try:
                    results.append(bool(next(it)))
                except StopIteration:
                    results.append(False)
                except TimeoutError:
                    results.append(False)
                bar.update(1)

    for sample, ok in zip(rows, results):
        sample["correct"] = ok
        sample.pop("gt", None)
        sample.pop("gt_cot", None)
