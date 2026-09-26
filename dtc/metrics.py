from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def expected_calibration_error_from_split(
    correct_probs: List[float],
    incorrect_probs: List[float],
    n_bins: int = 20,
) -> float:
    y_true = np.array([1] * len(correct_probs) + [0] * len(incorrect_probs), dtype=float)
    y_prob = np.array(correct_probs + incorrect_probs, dtype=float)
    n_samples = y_prob.size
    if n_samples == 0:
        return float("nan")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    sum_of_probs = np.bincount(bin_indices, weights=y_prob, minlength=n_bins)
    sum_of_positives = np.bincount(bin_indices, weights=y_true, minlength=n_bins)

    nonzero = bin_counts > 0
    fraction_of_positives = sum_of_positives[nonzero] / bin_counts[nonzero]
    mean_predicted_value = sum_of_probs[nonzero] / bin_counts[nonzero]
    bin_weights = bin_counts[nonzero] / n_samples
    return float(np.sum(bin_weights * np.abs(fraction_of_positives - mean_predicted_value)))


def compute_metrics_balanced(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    max_num_instances: int = 500,
    n_passes: int = 5,
    seed: int = 0,
    n_bins: int = 20,
) -> Dict[str, float]:
    labels = labels.astype(int)
    probs = probs.astype(float)
    correct_idx = np.flatnonzero(labels == 1)
    incorrect_idx = np.flatnonzero(labels == 0)
    n_inst = min(max_num_instances, len(correct_idx), len(incorrect_idx))
    if n_inst == 0:
        nan = float("nan")
        return {
            "auc_mean": nan,
            "auc_std": nan,
            "ece_mean": nan,
            "ece_std": nan,
            "brier_mean": nan,
            "brier_std": nan,
            "n_per_class": 0,
            "n_passes": n_passes,
        }

    aucs: List[float] = []
    eces: List[float] = []
    briers: List[float] = []
    for i in range(n_passes):
        rng = np.random.RandomState(seed + i)
        c_idx = rng.choice(correct_idx, size=n_inst, replace=False)
        ic_idx = rng.choice(incorrect_idx, size=n_inst, replace=False)
        c_probs = probs[c_idx].tolist()
        ic_probs = probs[ic_idx].tolist()
        y = np.array([1] * n_inst + [0] * n_inst)
        p = np.array(c_probs + ic_probs)
        try:
            aucs.append(float(roc_auc_score(y, p)))
        except ValueError:
            aucs.append(float("nan"))
        eces.append(expected_calibration_error_from_split(c_probs, ic_probs, n_bins=n_bins))
        try:
            briers.append(float(brier_score_loss(y, p)))
        except ValueError:
            briers.append(float("nan"))

    def _ms(xs: List[float]) -> Tuple[float, float]:
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    auc_m, auc_s = _ms(aucs)
    ece_m, ece_s = _ms(eces)
    bri_m, bri_s = _ms(briers)
    return {
        "auc_mean": auc_m,
        "auc_std": auc_s,
        "ece_mean": ece_m,
        "ece_std": ece_s,
        "brier_mean": bri_m,
        "brier_std": bri_s,
        "n_per_class": n_inst,
        "n_passes": n_passes,
    }


def accuracy_percent(labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    return float(100.0 * np.mean(labels.astype(float)))
