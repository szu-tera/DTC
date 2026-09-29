from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Paper operating points
CLIN_A = 0.95
CLIN_B = 0.05
CLIN_N = 10
PROD_K = 4

THETA_QWEN25 = 0.50
THETA_QWEN3 = 0.60
THETA_GEMMA3 = 0.85
THETA_BLACKBOX = 0.70

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

WHITE_BOX_DATASETS = ("math500", "amc23", "aime24", "aime25")
BLACK_BOX_DATASETS = ("aime24", "aime25", "hmmt_25", "hmmt_feb_2026")

VERBALIZED_METHODS = (
    "verbalized_confidence",
    "verbalized_topk",
    "verbalized_distribution",
)


def verbalized_sample_path(method: str, dataset: str, model: str) -> Path:
    if method not in VERBALIZED_METHODS:
        raise ValueError(f"未知 verbalized method: {method}")
    return OUTPUTS_DIR / "sample" / f"on_{method}" / f"{dataset}_{_basename(model)}.jsonl"

SAMPLES_PER_QUESTION = {
    "math500": 8,
    "amc23": 64,
    "aime24": 64,
    "aime25": 64,
    "hmmt_25": 64,
    "hmmt_feb_2026": 64,
}

# Black-box API (DeepSeek-V3.2): 32 per question
SAMPLES_PER_QUESTION_API = 32

DTC_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = DTC_ROOT / "evaluation" / "Qwen2.5-Math" / "evaluation"
DATA_DIR = EVAL_ROOT / "data"
OUTPUTS_DIR = DTC_ROOT / "outputs"


@dataclass(frozen=True)
class SamplingParams:
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    max_tokens: int
    num_samples: int


def _basename(model: str) -> str:
    p = Path(model.strip())
    if p.exists() or "/" in model or "\\" in model:
        return p.name
    return model.strip()


def is_gemma3_it(model: str) -> bool:
    name = _basename(model).lower()
    return "gemma-3" in name and "-it" in name


def is_qwen3(model: str) -> bool:
    name = _basename(model)
    return bool(re.search(r"Qwen3", name, flags=re.IGNORECASE))


def is_qwen25(model: str) -> bool:
    name = _basename(model)
    return bool(re.search(r"Qwen2\.5", name, flags=re.IGNORECASE))


def is_qwen3_instruct_2507(model: str) -> bool:
    name = _basename(model)
    return bool(re.search(r"Qwen3-.*Instruct-2507", name, flags=re.IGNORECASE))


def is_deepseek_api(model: str) -> bool:
    name = _basename(model).lower()
    return "deepseek" in name


def whitebox_theta(generator: str) -> float:
    if is_gemma3_it(generator):
        return THETA_GEMMA3
    if is_qwen3(generator):
        return THETA_QWEN3
    if is_qwen25(generator):
        return THETA_QWEN25
    return THETA_QWEN25


def default_aux_for_generator(generator: str) -> str:
    """Same-family small auxiliary (user must place weights under MODELS_DIR)."""
    if is_gemma3_it(generator):
        return "gemma-3-4b-it"
    if is_qwen3(generator):
        return "Qwen3-1.7B"
    return "Qwen2.5-1.5B-Instruct"


def sampling_params(
    model: str,
    dataset: str,
    *,
    backend: str = "vllm",
) -> SamplingParams:
    backend = (backend or "vllm").lower()
    if backend == "api" or is_deepseek_api(model):
        n = SAMPLES_PER_QUESTION_API
        return SamplingParams(
            temperature=1.0,
            top_p=0.95,
            top_k=-1,
            presence_penalty=0.0,
            max_tokens=16384,
            num_samples=n,
        )
    if is_qwen3_instruct_2507(model):
        n = SAMPLES_PER_QUESTION.get(dataset, 64)
        return SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            presence_penalty=1.0,
            max_tokens=16384,
            num_samples=n,
        )
    n = SAMPLES_PER_QUESTION.get(dataset, 8)
    return SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=-1,
        presence_penalty=0.0,
        max_tokens=8192,
        num_samples=n,
    )


def resolve_model_path(model: str, models_dir: Optional[Path] = None) -> str:
    """Local path: MODELS_DIR / basename if exists, else treat as HF id or absolute path."""
    import os

    raw = model.strip()
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return str(p.resolve())
    if p.exists():
        return str(p.resolve())
    root = models_dir or Path(os.environ.get("MODELS_DIR", "/mnt/data_oss/models"))
    cand = root / _basename(raw)
    if cand.exists():
        return str(cand.resolve())
    return raw
